"""Client-side observation -> policy-input building for the inference pipeline.

One place for every transform that turns a raw observation into the policy-server request,
so the layout stays identical to MDM_data_collection/build_dataset.py (training) and the
server's decode. Matches task/dual_arm_ee_image.yaml shape_meta:

  request (obs) fields (all images standardized to 16:9 = IMG_W x IMG_H = 256 x 144):
    agentview_image            head camera         (TOP-cropped to 16:9 + resized base64 JPEG)
    robotl_eye_in_hand_image   left  wrist camera  (~16:9, centered crop + resized base64 JPEG)
    robotr_eye_in_hand_image   right wrist camera  (~16:9, centered crop + resized base64 JPEG)
    robotl_eef_pos   (To, 9)   left  EE [pos(3) + rot6d(6)]
    robotr_eef_pos   (To, 9)   right EE [pos(3) + rot6d(6)]
    robot0_grip      (To, 2)   [left, right] gripper (raw, as the recordings stored it)

Two kinds of building live here:
  * IMAGES: crop/resize/encode each camera frame to base64 JPEG (encode_image). Doing the
    crop+resize HERE, BEFORE the JPEG encode, shrinks the head frame from native (e.g.
    1280x800) to IMG_W x IMG_H first, so imencode runs on ~25x fewer pixels. NOTE: with
    AGENT_CROP_ZOOM == 1.0 the server's own crop+resize become no-ops on an already-target-
    size 16:9 frame, so NO server change is needed PROVIDED the server's configured
    image_shape is [3, 144, 256] (16:9) too. If AGENT_CROP_ZOOM is ever raised > 1, the
    server MUST disable its center-crop or the head frame gets zoomed twice.
  * PROPRIOCEPTION: pack each per-frame EE pose (build_ee_pose_row, with quat->rot6d done
    EXACTLY as training's quat_to_rot6d) and the 2-gripper row (build_grip_row) into the
    training zarr layout. The live env (humanoid_env) and the offline replay source
    (recorded_obs) both build their obs rows through these, so the format is defined once.
    The FK / SDK reads that PRODUCE the numbers stay with their sources; only the layout
    (and the quaternion->6D rotation conversion) lives here.
"""

import base64

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

IMG_H = 144             # target rows  (build_dataset.IMG_H) — 16:9
IMG_W = 256             # target cols  (build_dataset.IMG_W) — 16:9 (256/144 = 16/9)
AGENT_CROP_ZOOM = 1.0   # head crop zoom (build_dataset.AGENT_CROP_ZOOM); keep in sync


def crop_to_aspect(frame, target_w, target_h, zoom=1.0, keep="center"):
    """Crop `frame` to the target aspect ratio (target_w/target_h), optionally zooming in.

    All cameras are standardized to 16:9 (the target aspect) by cropping BEFORE the resize, so
    the resize never distorts. The horizontal crop is always centered. The vertical crop position
    (which rows to drop when the source is too tall) is set by `keep`:
      "center" -> drop equally from top & bottom (default; wrist cams are already ~16:9, ~0 rows)
      "bottom" -> KEEP the bottom, drop rows from the TOP   (head cam: 16:10 is too tall — preserve
                  the workspace at the bottom of the frame, crop the ceiling/background at the top)
      "top"    -> KEEP the top, drop rows from the bottom
    Returns a cropped view."""
    h, w = frame.shape[:2]
    target_ar = target_w / target_h
    if w / h > target_ar:        # source too wide -> limited by height
        crop_h, crop_w = h, int(round(h * target_ar))
    else:                        # source too tall -> limited by width
        crop_w, crop_h = w, int(round(w / target_ar))
    crop_w = min(w, int(round(crop_w / zoom)))
    crop_h = min(h, int(round(crop_h / zoom)))
    x0 = (w - crop_w) // 2
    if keep == "bottom":
        y0 = h - crop_h          # drop rows from the TOP, keep the bottom
    elif keep == "top":
        y0 = 0                   # drop rows from the bottom, keep the top
    else:
        y0 = (h - crop_h) // 2   # centered
    return frame[y0:y0 + crop_h, x0:x0 + crop_w]


def preprocess_frame(rgb_image, keep="center"):
    """RGB frame -> cropped-to-16:9 + resized RGB uint8 at (IMG_H, IMG_W).

    THE shared pixel transform, used by BOTH training (build_dataset.read_video, building the zarr)
    and inference (encode_image, building the request). EVERY camera is cropped to the target 16:9
    aspect, then INTER_AREA-resized to (IMG_W, IMG_H). The head passes keep="bottom" (it is 16:10 —
    too tall — so the crop drops rows from the TOP only, preserving the workspace at the bottom and
    applying AGENT_CROP_ZOOM); the wrist cams are already ~16:9 so their crop is a few rows, centered.
    Defining it once here is what keeps the train and deploy pixels identical."""
    zoom = AGENT_CROP_ZOOM if keep == "bottom" else 1.0
    rgb_image = crop_to_aspect(rgb_image, IMG_W, IMG_H, zoom=zoom, keep=keep)
    if rgb_image.shape[:2] != (IMG_H, IMG_W):
        rgb_image = cv2.resize(rgb_image, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return rgb_image


def encode_image(rgb_image, keep="center"):
    """RGB frame -> base64 JPEG string for the /predict request.

    Applies the shared preprocess_frame transform (so it matches training pixel-for-pixel up to the
    JPEG codec), then BGR-JPEG-encodes the already-small frame, which is what makes it fast.
    The head passes keep="bottom" (top-crop); wrist cams use the default centered crop."""
    rgb_image = preprocess_frame(rgb_image, keep)
    rgb_image_cv2 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', rgb_image_cv2, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode('utf-8')


def quat_to_rot6d(quat):
    """(...,4) [qx,qy,qz,qw] quaternion -> (...,6) continuous 6D rotation (the first two columns
    of the rotation matrix). IDENTICAL to build_dataset.quat_to_rot6d, so the obs rotation fed to
    the policy matches what training wrote into robot*_eef_pos."""
    mat = R.from_quat(quat).as_matrix()                          # (...,3,3)
    return np.concatenate([mat[..., 0], mat[..., 1]], axis=-1)   # (...,6)


def build_ee_pose(pos, quat):
    """Vectorized EE obs rows: (N,3) pos + (N,4) quat xyzw -> (N,9) [pos + rot6d] float64.

    THE shared EE-layout definition: build_dataset uses this for bulk zarr building and
    build_ee_pose_row (per-frame, below) delegates to it, so training and inference produce the
    same robot*_eef_pos rows from the same rotation math."""
    pos = np.asarray(pos, dtype=np.float64)
    quat = np.asarray(quat, dtype=np.float64)
    return np.concatenate([pos, quat_to_rot6d(quat)], axis=-1)


def build_ee_pose_row(pos, quat):
    """One EE obs row: [eef_pos(3) + rot6d(6)] = 9 dims (the training robot*_eef_pos layout).

    Per-frame wrapper over build_ee_pose for the live env / replay obs: `pos` is array-like (3,),
    `quat` xyzw (4,). Returns plain Python floats so the row is JSON-serialisable for the /predict
    request and homogeneous across the live env and replay."""
    row = build_ee_pose(np.asarray(pos)[None], np.asarray(quat)[None])[0]
    return [float(v) for v in row]


def build_grip_row(left_grip, right_grip):
    """One robot0_grip obs row: [left, right] gripper = 2 dims.

    Raw values, as the recordings stored them (no binarisation in obs — that matches training's
    raw robot0_grip). Returns plain Python floats (JSON-safe)."""
    return [float(left_grip), float(right_grip)]


def build_predict_request(obs):
    """Assemble the dual-arm policy-server /predict request from one observation snapshot.

    Pure transform (obs dict -> request dict): encodes the three camera frames and passes through
    the already-packed proprioception rows (built via build_ee_pose_row / build_grip_row). The
    field names and image preprocessing here are the contract with task/dual_arm_ee_image.yaml's
    shape_meta and the server decode — keep in sync."""
    return {
        # head (agent) is top-cropped to 16:9 like training (keep="bottom"); wrists are already
        # ~16:9 so they get the default centered few-row crop. Both go through the shared transform.
        'agentview_image':          [encode_image(img, keep="bottom") for img in obs['agent_imgs']],
        'robotl_eye_in_hand_image': [encode_image(img) for img in obs['handl_imgs']],
        'robotr_eye_in_hand_image': [encode_image(img) for img in obs['handr_imgs']],
        'robotl_eef_pos':           obs['robotl_eef_pos'],   # (To, 9) [pos3 + rot6d6]
        'robotr_eef_pos':           obs['robotr_eef_pos'],   # (To, 9)
        'robot0_grip':              obs['robot0_grip'],      # (To, 2) [left, right]
    }
