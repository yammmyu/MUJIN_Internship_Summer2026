"""
Renders a single RGB texture as a head-stable 3D quad inside the projection
layer's swapchain image.

Why a 3D quad rather than a fullscreen NDC quad?
The :class:`xr.utils.gl.ContextObject` we use is configured with a ``LOCAL``
reference space so the cube renderer can position controller cubes in world
coordinates. In that mode the OpenXR runtime treats the rendered content as
world-anchored and may reproject it for late head motion. A fullscreen NDC
quad would shift visibly when the head moves; instead we place the quad
exactly at ``(0, 0, -d)`` in *view* space, sized to fill the FOV. Algebraically
the view.pose factor cancels out:

    mvp = proj @ view_matrix @ (view.pose @ quad_in_view)
        = proj @ quad_in_view

so the quad always lands at the head-relative position the runtime expects.
The reprojection step then "follows the head" exactly, giving a stable virtual
screen even with low video frame rates.

Aspect ratio is preserved by letterbox/pillarbox-shrinking the quad inside the
FOV rectangle so the texture isn't stretched.
"""

import ctypes
import inspect
import logging
import math
from typing import Optional

import numpy as np
from OpenGL import GL

import xr
from xr.utils import GraphicsAPI, Matrix4x4f

logger = logging.getLogger("xr_examples.pico_vr_client.image_layer")


_VERTEX_SHADER = inspect.cleandoc("""
    #version 410
    in vec2 a_pos;
    in vec2 a_uv;
    out vec2 v_uv;
    uniform mat4 u_mvp;
    void main() {
        v_uv = a_uv;
        gl_Position = u_mvp * vec4(a_pos, 0.0, 1.0);
    }
""")

_FRAGMENT_SHADER = inspect.cleandoc("""
    #version 410
    in vec2 v_uv;
    out vec4 frag_color;
    uniform sampler2D u_image;
    void main() {
        frag_color = vec4(texture(u_image, v_uv).rgb, 1.0);
    }
""")


# Unit quad in [-1, 1]^2. UV (0, 0) maps to the top-left of the image, so the
# V coordinate is flipped relative to OpenGL's bottom-left texture convention.
_QUAD_VERTICES = np.array([
    # x,   y,    u,   v
    -1.0, -1.0,  0.0, 1.0,
     1.0, -1.0,  1.0, 1.0,
    -1.0,  1.0,  0.0, 0.0,
     1.0,  1.0,  1.0, 0.0,
], dtype=np.float32)
_QUAD_BYTES = _QUAD_VERTICES.tobytes()

# Virtual screen distance in metres. The runtime reprojects around this point
# for late head motion, so smaller values give crisper stability but a more
# obviously head-locked screen.
_VIRTUAL_SCREEN_DISTANCE_M = 2.0


def _drain_gl_errors(prefix: str) -> None:
    while True:
        err = GL.glGetError()
        if err == GL.GL_NO_ERROR:
            return
        logger.error(f"{prefix}: GL error {err:#x}")


class ImageLayer:
    def __init__(self, distance_m: float = _VIRTUAL_SCREEN_DISTANCE_M) -> None:
        self._distance_m = distance_m
        self._program: Optional[int] = None
        self._vao: Optional[int] = None
        self._vbo: Optional[int] = None
        self._texture: Optional[int] = None
        self._tex_w = 0
        self._tex_h = 0
        self._last_frame_id = -1
        self._u_mvp: int = -1
        self._first_draw_logged = False
        self._first_upload_logged = False

    def initialize(self) -> None:
        self._program = _compile_program(_VERTEX_SHADER, _FRAGMENT_SHADER)
        a_pos = GL.glGetAttribLocation(self._program, "a_pos")
        a_uv = GL.glGetAttribLocation(self._program, "a_uv")
        u_image = GL.glGetUniformLocation(self._program, "u_image")
        self._u_mvp = GL.glGetUniformLocation(self._program, "u_mvp")
        if a_pos < 0 or a_uv < 0:
            raise RuntimeError(f"Vertex attribs not found: a_pos={a_pos}, a_uv={a_uv}")

        self._vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self._vao)
        self._vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, len(_QUAD_BYTES), _QUAD_BYTES, GL.GL_STATIC_DRAW)
        stride = 4 * _QUAD_VERTICES.itemsize
        GL.glEnableVertexAttribArray(a_pos)
        GL.glVertexAttribPointer(a_pos, 2, GL.GL_FLOAT, False, stride, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(a_uv)
        GL.glVertexAttribPointer(a_uv, 2, GL.GL_FLOAT, False, stride,
                                 ctypes.c_void_p(2 * _QUAD_VERTICES.itemsize))
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        _drain_gl_errors("ImageLayer.initialize: VBO setup")

        self._texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        GL.glUseProgram(self._program)
        GL.glUniform1i(u_image, 0)
        GL.glUseProgram(0)
        _drain_gl_errors("ImageLayer.initialize: texture/program setup")

    def destroy(self) -> None:
        if self._vbo is not None:
            GL.glDeleteBuffers(1, [self._vbo])
            self._vbo = None
        if self._vao is not None:
            GL.glDeleteVertexArrays(1, [self._vao])
            self._vao = None
        if self._texture is not None:
            GL.glDeleteTextures(1, [self._texture])
            self._texture = None
        if self._program is not None:
            GL.glDeleteProgram(self._program)
            self._program = None

    def update_texture(self, frame: np.ndarray, frame_id: int) -> None:
        """Upload ``frame`` (H, W, 3) uint8 RGB into the texture, if it changed."""
        if frame is None or frame_id == self._last_frame_id:
            return
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            logger.warning(f"Unsupported frame shape/dtype: {frame.shape} {frame.dtype}")
            return
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        h, w = frame.shape[:2]
        pixel_bytes = frame.tobytes()
        expected = w * h * 3
        if len(pixel_bytes) != expected:
            logger.warning(f"Pixel byte size mismatch: got {len(pixel_bytes)}, expected {expected}")
            return
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        if w != self._tex_w or h != self._tex_h:
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB8, w, h, 0,
                            GL.GL_RGB, GL.GL_UNSIGNED_BYTE, pixel_bytes)
            self._tex_w, self._tex_h = w, h
        else:
            GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, w, h,
                               GL.GL_RGB, GL.GL_UNSIGNED_BYTE, pixel_bytes)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        _drain_gl_errors("ImageLayer.update_texture")
        self._last_frame_id = frame_id
        if not self._first_upload_logged:
            self._first_upload_logged = True
            logger.info(f"First texture upload: {w}x{h} (frame_id={frame_id})")

    def draw(self, view: xr.View,
             background_rgba=(0.0, 0.0, 0.0, 1.0),
             near: float = 0.05, far: float = 100.0) -> None:
        GL.glClearColor(*background_rgba)
        GL.glClearDepth(1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if self._texture is None or self._tex_w == 0:
            return
        mvp = self._compute_mvp(view, near, far)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glDisable(GL.GL_BLEND)
        GL.glDisable(GL.GL_SCISSOR_TEST)
        GL.glDisable(GL.GL_STENCIL_TEST)
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glUseProgram(self._program)
        GL.glUniformMatrix4fv(self._u_mvp, 1, False, mvp.as_numpy())
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        GL.glBindVertexArray(0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)
        if not self._first_draw_logged:
            self._first_draw_logged = True
            _drain_gl_errors("ImageLayer.draw (first call)")
            logger.info(
                f"First textured draw issued (image {self._tex_w}x{self._tex_h})"
            )

    def _compute_mvp(self, view: xr.View, near: float, far: float) -> Matrix4x4f:
        """Build MVP that places the textured quad at z=-d in view space."""
        proj = Matrix4x4f.create_projection_fov(GraphicsAPI.OPENGL, view.fov, near, far)
        d = self._distance_m
        # FOV-filling rectangle (view space) at z=-d.
        left = d * math.tan(view.fov.angle_left)
        right = d * math.tan(view.fov.angle_right)
        up = d * math.tan(view.fov.angle_up)
        down = d * math.tan(view.fov.angle_down)
        view_w = right - left
        view_h = up - down
        # Aspect-correct: letterbox / pillarbox so the image isn't stretched.
        if self._tex_w > 0 and self._tex_h > 0 and view_w > 0 and view_h > 0:
            view_aspect = view_w / view_h
            img_aspect = self._tex_w / self._tex_h
            if img_aspect > view_aspect:
                # Image wider than view: keep width, shrink height.
                new_h = view_w / img_aspect
                cy = 0.5 * (up + down)
                up, down = cy + 0.5 * new_h, cy - 0.5 * new_h
            else:
                new_w = view_h * img_aspect
                cx = 0.5 * (left + right)
                left, right = cx - 0.5 * new_w, cx + 0.5 * new_w
        h_center = 0.5 * (left + right)
        v_center = 0.5 * (up + down)
        h_half = 0.5 * (right - left)
        v_half = 0.5 * (up - down)
        identity_q = xr.Quaternionf(0.0, 0.0, 0.0, 1.0)
        # Place the unit quad [-1, 1]^2 at z=-d with the right scale.
        quad_in_view = Matrix4x4f.create_translation_rotation_scale(
            xr.Vector3f(h_center, v_center, -d),
            identity_q,
            xr.Vector3f(h_half, v_half, 1.0),
        )
        # The view.pose component cancels (view_matrix @ view.pose = identity)
        # so the MVP simplifies to projection @ quad_in_view. This is what makes
        # the quad track the head perfectly across late reprojection.
        return proj @ quad_in_view


def _compile_program(vertex_src: str, fragment_src: str) -> int:
    vs = _compile_shader(GL.GL_VERTEX_SHADER, vertex_src)
    fs = _compile_shader(GL.GL_FRAGMENT_SHADER, fragment_src)
    program = GL.glCreateProgram()
    GL.glAttachShader(program, vs)
    GL.glAttachShader(program, fs)
    GL.glLinkProgram(program)
    if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
        log = GL.glGetProgramInfoLog(program).decode("utf-8", errors="replace")
        raise RuntimeError(f"Failed to link shader program: {log}")
    GL.glDeleteShader(vs)
    GL.glDeleteShader(fs)
    return program


def _compile_shader(kind: int, src: str) -> int:
    shader = GL.glCreateShader(kind)
    GL.glShaderSource(shader, src)
    GL.glCompileShader(shader)
    if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
        log = GL.glGetShaderInfoLog(shader).decode("utf-8", errors="replace")
        raise RuntimeError(f"Failed to compile shader: {log}")
    return shader
