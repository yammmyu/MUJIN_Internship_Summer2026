"""Point cloud extraction from depth + masking out detected oriented boxes.

Pipeline (all in the Detector's output-frame conventions):

    depth + intrinsics ──► (H,W,3) camera-optical points
                                      │ CAMERA_TO_OUTPUT_ROTATION
                                      ▼
                                  (N,3) output-frame  (camera at origin,
                                                       +X depth, +Y left, +Z up)
                                      │ CameraMountingRotation(yaw,pitch,roll)?
                                      ▼
                                  (N,3) compensated frame (matches
                                                            CompensateCameraOrientation
                                                            applied to detections)
                                      │ VoxelDownsample
                                      │ MaskOutOrientedBoxes(detections, margin)
                                      ▼
                                  (M,3) scene cloud around detected objects

The caller (PlanningServer.ProcessVision) then applies the camera→world
transform to land these points in the OpenRAVE world frame and spawns them as
a single KinBody composed of small boxes.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from detector import (
    CAMERA_TO_OUTPUT_ROTATION,
    CameraIntrinsics,
    CameraMountingRotation,
    Detector,
    OrientedBox3D,
)


def DepthToPointCloud(
    depth: NDArray,
    intrinsics: CameraIntrinsics,
    cameraYawDeg: float = 0.0,
    cameraPitchDeg: float = 0.0,
    cameraRollDeg: float = 0.0,
    finiteOnly: bool = True,
) -> NDArray:
    """Back-projects a depth image into (N,3) points in the detector output frame.

    Non-zero yaw/pitch/roll additionally rotates the cloud by
    `CameraMountingRotation(...)` so it sits in the gravity-leveled,
    yaw-compensated frame the Detector uses for compensated `orientedBox3D`
    (matches `CompensateCameraOrientation`).
    """
    structured = Detector._DepthToPointCloud(depth, intrinsics)  # (H,W,3) optical
    pts = structured.reshape(-1, 3).astype(np.float64, copy=False)
    if finiteOnly:
        valid = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0)
        pts = pts[valid]
    # Optical -> output
    pts = pts @ CAMERA_TO_OUTPUT_ROTATION.T
    # Mounting compensation
    if cameraYawDeg or cameraPitchDeg or cameraRollDeg:
        rot = CameraMountingRotation(cameraYawDeg, cameraPitchDeg, cameraRollDeg)
        pts = pts @ rot.T
    return pts


def VoxelDownsample(points: NDArray, voxelSize: float) -> NDArray:
    """Returns one representative point per voxel of the given side length.

    Picks the first point that lands in each voxel (stable, fast).
    """
    if len(points) == 0 or voxelSize <= 0:
        return points
    keys = np.floor(np.asarray(points) / float(voxelSize)).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


def _QuaternionToRotation(quat: NDArray) -> NDArray:
    """(w,x,y,z) quaternion → 3x3 rotation matrix (box-local → output frame)."""
    q = np.asarray(quat, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def MaskOutOrientedBoxes(
    points: NDArray,
    boxes: Iterable[OrientedBox3D],
    margin: float = 0.02,
) -> NDArray:
    """Drops every point lying inside any of the given OBBs (inflated by margin).

    `margin` (meters) expands each face of each box outward — useful because
    the detector's `size` is a lower bound along the face normal (single view),
    so the true object usually extends a bit further than the OBB suggests.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return pts
    keep = np.ones(len(pts), dtype=bool)
    for box in boxes:
        if box is None:
            continue
        R = _QuaternionToRotation(np.asarray(box.quaternion, dtype=float))
        center = np.asarray(box.position, dtype=float)
        half = np.asarray(box.size, dtype=float) / 2.0 + float(margin)
        # box-local coords:  local_col = R.T @ (p - c)
        # For row vectors:   local_row = (p - c) @ R
        local = (pts - center) @ R
        inside = np.all(np.abs(local) <= half, axis=1)
        keep &= ~inside
    return pts[keep]


def ExtractSceneCloud(
    depth: NDArray,
    intrinsics: CameraIntrinsics,
    boxes: Sequence[OrientedBox3D],
    voxelSize: float = 0.02,
    maskMargin: float = 0.02,
    cameraYawDeg: float = 0.0,
    cameraPitchDeg: float = 0.0,
    cameraRollDeg: float = 0.0,
    maxRangeMeters: Optional[float] = 3.0,
    minRangeMeters: float = 0.10,
) -> NDArray:
    """End-to-end helper: depth + detections → downsampled scene cloud.

    Returned (N,3) points are in the SAME frame as the input `boxes`
    (compensated frame if yaw/pitch/roll were given to both the detector and
    here, raw detector output frame otherwise). The caller maps to world.

    The detector's output frame has +X = depth/forward, +Y = image-left, +Z =
    up; range filtering is applied along that +X axis (camera-relative depth).

    Args:
        voxelSize:        downsampling step in meters (0 disables)
        maskMargin:       inflate each detected box by this much before masking
        maxRangeMeters:   drop points farther than this in camera-depth (+X)
        minRangeMeters:   drop points closer than this in camera-depth (+X)
    """
    pts = DepthToPointCloud(
        depth, intrinsics,
        cameraYawDeg=cameraYawDeg,
        cameraPitchDeg=cameraPitchDeg,
        cameraRollDeg=cameraRollDeg,
    )
    if len(pts) and (maxRangeMeters is not None or minRangeMeters > 0):
        mask = np.ones(len(pts), dtype=bool)
        if minRangeMeters > 0:
            mask &= pts[:, 0] >= float(minRangeMeters)
        if maxRangeMeters is not None:
            mask &= pts[:, 0] <= float(maxRangeMeters)
        pts = pts[mask]
    if voxelSize > 0:
        pts = VoxelDownsample(pts, voxelSize)
    if boxes:
        pts = MaskOutOrientedBoxes(pts, boxes, margin=maskMargin)
    return pts
