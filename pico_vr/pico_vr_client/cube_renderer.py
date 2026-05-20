"""
Render solid cubes at given XR poses using hello_xr's per-face colour scheme.

Used in the Pico VR client to visualise the live left/right controller positions
inside the head-locked image overlay. Geometry and shading mirror
``xr_examples/hello_xr/geometry.py`` + ``graphics_plugin_opengl.py``: each cube
face is a bright/dark variant of one of R/G/B, so the cube's 3D shape and
orientation are immediately legible. Grab state is encoded by shrinking the
cube (size is supplied per-instance).

The renderer assumes the depth buffer was cleared before it runs and that
preceding passes did not write to it (the image layer disables depth test,
so this holds). It enables depth test / depth write itself so multiple cubes
sort correctly.
"""

import ctypes
import inspect
import logging
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from OpenGL import GL

import xr
from xr.utils import GraphicsAPI, Matrix4x4f

logger = logging.getLogger("xr_examples.pico_vr_client.cube_renderer")


_VERTEX_SHADER = inspect.cleandoc("""
    #version 410
    in vec3 a_pos;
    in vec3 a_color;
    out vec3 v_color;
    uniform mat4 u_mvp;
    void main() {
        v_color = a_color;
        gl_Position = u_mvp * vec4(a_pos, 1.0);
    }
""")

_FRAGMENT_SHADER = inspect.cleandoc("""
    #version 410
    in vec3 v_color;
    out vec4 frag_color;
    void main() {
        frag_color = vec4(v_color, 1.0);
    }
""")


# Unit-cube corners, named LRBT-FB (Left/Right, Bottom/Top, Back/Front).
_LBB = (-0.5, -0.5, -0.5)
_LBF = (-0.5, -0.5,  0.5)
_LTB = (-0.5,  0.5, -0.5)
_LTF = (-0.5,  0.5,  0.5)
_RBB = ( 0.5, -0.5, -0.5)
_RBF = ( 0.5, -0.5,  0.5)
_RTB = ( 0.5,  0.5, -0.5)
_RTF = ( 0.5,  0.5,  0.5)

# Bright + dark variants of R/G/B, one pair per axis. Matches hello_xr.
_RED       = (1.00, 0.00, 0.00)
_DARK_RED  = (0.25, 0.00, 0.00)
_GREEN     = (0.00, 1.00, 0.00)
_DARK_GREEN= (0.00, 0.25, 0.00)
_BLUE      = (0.00, 0.00, 1.00)
_DARK_BLUE = (0.00, 0.00, 0.25)


def _face(v1, v2, v3, v4, v5, v6, color):
    """6 (pos, color) vertices spanning two triangles for one face."""
    return [(*v, *color) for v in (v1, v2, v3, v4, v5, v6)]


# 6 faces × 6 vertices, interleaved (px,py,pz, cr,cg,cb). Winding matches
# hello_xr's clockwise convention; the renderer disables face culling so
# orientation is not critical.
_CUBE_VERTICES = np.array(
    _face(_LTB, _LBF, _LBB, _LTB, _LTF, _LBF, _DARK_RED)
    + _face(_RTB, _RBB, _RBF, _RTB, _RBF, _RTF, _RED)
    + _face(_LBB, _LBF, _RBF, _LBB, _RBF, _RBB, _DARK_GREEN)
    + _face(_LTB, _RTB, _RTF, _LTB, _RTF, _LTF, _GREEN)
    + _face(_LBB, _RBB, _RTB, _LBB, _RTB, _LTB, _DARK_BLUE)
    + _face(_LBF, _LTF, _RTF, _LBF, _RTF, _RBF, _BLUE),
    dtype=np.float32,
)
_CUBE_VERTEX_BYTES = _CUBE_VERTICES.tobytes()
_CUBE_VERTEX_COUNT = _CUBE_VERTICES.shape[0]  # 36
_CUBE_STRIDE = 6 * _CUBE_VERTICES.itemsize    # pos(3) + color(3)


@dataclass
class CubeInstance:
    pose: xr.Posef
    size: float


class CubeRenderer:
    def __init__(self) -> None:
        self._program = None
        self._vao = None
        self._vbo = None
        self._u_mvp = -1

    def initialize(self) -> None:
        self._program = _compile_program(_VERTEX_SHADER, _FRAGMENT_SHADER)
        a_pos = GL.glGetAttribLocation(self._program, "a_pos")
        a_color = GL.glGetAttribLocation(self._program, "a_color")
        self._u_mvp = GL.glGetUniformLocation(self._program, "u_mvp")
        if a_pos < 0 or a_color < 0:
            raise RuntimeError(
                f"CubeRenderer: attribs not found (a_pos={a_pos}, a_color={a_color})"
            )

        self._vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self._vao)
        self._vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, len(_CUBE_VERTEX_BYTES),
                        _CUBE_VERTEX_BYTES, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(a_pos)
        GL.glVertexAttribPointer(a_pos, 3, GL.GL_FLOAT, False,
                                 _CUBE_STRIDE, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(a_color)
        GL.glVertexAttribPointer(a_color, 3, GL.GL_FLOAT, False, _CUBE_STRIDE,
                                 ctypes.c_void_p(3 * _CUBE_VERTICES.itemsize))
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def destroy(self) -> None:
        if self._vbo is not None:
            GL.glDeleteBuffers(1, [self._vbo])
            self._vbo = None
        if self._vao is not None:
            GL.glDeleteVertexArrays(1, [self._vao])
            self._vao = None
        if self._program is not None:
            GL.glDeleteProgram(self._program)
            self._program = None

    def draw(self, view: xr.View, cubes: Sequence[CubeInstance],
             near: float = 0.05, far: float = 100.0) -> None:
        if not cubes or self._program is None:
            return
        proj = Matrix4x4f.create_projection_fov(
            GraphicsAPI.OPENGL, view.fov, near, far,
        )
        unit = xr.Vector3f(1, 1, 1)
        to_view = Matrix4x4f.create_translation_rotation_scale(
            view.pose.position, view.pose.orientation, unit,
        )
        view_matrix = Matrix4x4f.invert_rigid_body(to_view)
        vp = proj @ view_matrix

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDepthFunc(GL.GL_LEQUAL)
        GL.glDisable(GL.GL_BLEND)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glUseProgram(self._program)
        GL.glBindVertexArray(self._vao)
        for cube in cubes:
            scale = xr.Vector3f(cube.size, cube.size, cube.size)
            # Force a unit quaternion so any upstream drift cannot leak into
            # the rotation matrix as scale/shear and visibly distort the cube.
            rot = _unit_quat(cube.pose.orientation)
            model = Matrix4x4f.create_translation_rotation_scale(
                cube.pose.position, rot, scale,
            )
            mvp = vp @ model
            GL.glUniformMatrix4fv(self._u_mvp, 1, False, mvp.as_numpy())
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, _CUBE_VERTEX_COUNT)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)


def _unit_quat(q: xr.Quaternionf) -> xr.Quaternionf:
    """Return a unit-length copy of ``q``; identity if ``q`` is degenerate."""
    n = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if n < 1e-8:
        return xr.Quaternionf(0.0, 0.0, 0.0, 1.0)
    return xr.Quaternionf(q.x / n, q.y / n, q.z / n, q.w / n)


def _compile_program(vertex_src: str, fragment_src: str) -> int:
    vs = _compile_shader(GL.GL_VERTEX_SHADER, vertex_src)
    fs = _compile_shader(GL.GL_FRAGMENT_SHADER, fragment_src)
    program = GL.glCreateProgram()
    GL.glAttachShader(program, vs)
    GL.glAttachShader(program, fs)
    GL.glLinkProgram(program)
    if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
        log = GL.glGetProgramInfoLog(program).decode("utf-8", errors="replace")
        raise RuntimeError(f"Failed to link cube shader: {log}")
    GL.glDeleteShader(vs)
    GL.glDeleteShader(fs)
    return program


def _compile_shader(kind: int, src: str) -> int:
    shader = GL.glCreateShader(kind)
    GL.glShaderSource(shader, src)
    GL.glCompileShader(shader)
    if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
        log = GL.glGetShaderInfoLog(shader).decode("utf-8", errors="replace")
        raise RuntimeError(f"Failed to compile cube shader: {log}")
    return shader
