"""Box face detector built on top of the instance segmentation wrapper.

Pipeline (single RGB + Depth pair):

    RGB  ──► Segmenter (segmenter.py) ──► instance masks
                                              │
    Depth + intrinsics ──► structured point cloud ──► normals
                                              │
    per instance: plane fit ──► tilt check ──► split into perpendicular faces
                                              │
    per face: 3D OBB ──► classify (top / side) ──► container & grasp filtering
                                              ▼
                              list of valid grasp faces

The face-splitting step mirrors the depalletizer detector's "Split faces of
tilted boxes" logic (see
`mujindetection.shared.instancesegmentation.depallet.detector._Postprocess` and
`mujindetection.shared.postprocessing.SplitSegmentationsOfTiltedItems`). When a
box is tilted enough that the camera sees two faces at once (e.g. a box facing
the camera at ~45deg shows both its short-edge and long-edge side faces), the
single instance mask is broken into the two perpendicular planar faces so each
can be evaluated as an independent grasp target. Horizontal (top) faces are kept
but flagged as low-value grasp targets.

Intended to be run inside the mujin docker container (same as segmenter.py),
where `mujindetection.shared.pointcloud` and `mujinvisioncommonutilitiesbindings`
are importable. Both are optional: pure-numpy fallbacks are used when they are
not available, so the module also runs in a plain numpy/opencv environment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy
from numpy.typing import NDArray

if TYPE_CHECKING:  # Imported lazily in Detector.__init__ (needs the docker env).
    from segmenter import InstanceResult

log = logging.getLogger(__name__)

# Default model type, kept local so this module can be imported (and its
# geometry post-processing used / tested) without the mujindetection package.
_DEFAULT_MODEL_TYPE = "rfdetr0m"

# Categories whose tilted instances are worth splitting into separate faces.
# Mirrors `categorySetToProcess={CategoryType.BOX}` in the reference detector.
_SPLITTABLE_NAMES = {"box", "irregularMultiPack"}


# --------------------------------------------------------------------------- #
# Input data structures
# --------------------------------------------------------------------------- #
@dataclass
class CameraIntrinsics:
    """Pinhole intrinsics for the depth image (and the aligned RGB image)."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    # Multiply raw depth by this to obtain meters (e.g. 0.001 for uint16 mm).
    depthScale: float = 1.0


@dataclass
class Container:
    """Container (bin / pallet region) pose and size in the depth-camera frame.

    Attributes:
        transform: 4x4 isometric matrix mapping container-local coordinates to
            the depth-camera frame. The container's up axis is column 2.
        extents: Full inner size (dx, dy, dz) of the container in meters, along
            the container-local x, y, z axes.
    """
    transform: NDArray[numpy.float64]
    extents: tuple[float, float, float]

    @classmethod
    def FromPositionSizeQuaternion(
        cls,
        position: tuple[float, float, float],
        size: tuple[float, float, float],
        quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ) -> "Container":
        """Builds a Container from a center position, size and (w,x,y,z) quaternion.

        All values are expressed in the depth-camera frame.
        """
        w, x, y, z = quaternion
        n = (w * w + x * x + y * y + z * z) ** 0.5
        w, x, y, z = w / n, x / n, y / n, z / n
        rot = numpy.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        transform = numpy.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = position
        return cls(transform=transform, extents=tuple(size))

    @property
    def upVector(self) -> NDArray[numpy.float64]:
        """Container up axis (z) in the depth-camera frame, normalized."""
        up = self.transform[:3, 2]
        return up / numpy.linalg.norm(up)

    def ContainsPoints(self, points3D: NDArray, margin: float = 0.0) -> NDArray[numpy.bool_]:
        """Returns a bool mask of which (N,3) camera-frame points lie inside.

        Args:
            points3D: (N, 3) points in the depth-camera frame.
            margin: Inflate the container bounds by this many meters on each side.
                Negative values shrink it.
        """
        inv = numpy.linalg.inv(self.transform)
        local = points3D @ inv[:3, :3].T + inv[:3, 3]
        half = numpy.asarray(self.extents) / 2.0 + margin
        return numpy.all(numpy.abs(local) <= half, axis=1)


# --------------------------------------------------------------------------- #
# Output data structures
# --------------------------------------------------------------------------- #
@dataclass
class GraspFace:
    """One planar face extracted from a detected instance.

    Geometry is expressed in the depth-camera frame. The face normal always
    points toward the camera (away from the object body).
    """
    instanceIndex: int               # index into the returned DetectionResult list
    mask: NDArray[numpy.bool_]       # (H, W) bool, depth-image resolution
    center3D: NDArray[numpy.float64]  # (3,) face center
    normal3D: NDArray[numpy.float64]  # (3,) unit normal toward the camera
    corners3D: NDArray[numpy.float64]  # (4, 3) ordered OBB corners
    corners2D: NDArray[numpy.float64]  # (4, 2) corners projected into the image
    dimensions: tuple[float, float]  # (long edge, short edge) in meters
    area: float                      # face area in m^2 (long * short)
    faceType: str                    # "side" (vertical) or "top" (horizontal)
    tiltWrtCameraDeg: float          # angle between normal and line of sight
    isValidGraspFace: bool           # passed all grasp-face filters


@dataclass
class DetectionResult:
    """An instance segmentation detection plus its post-processed grasp faces."""
    instance: "InstanceResult"
    faces: list[GraspFace] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Optional mujin backends (available inside the docker container)
# --------------------------------------------------------------------------- #
try:  # Battle-tested plane fitting and face splitting from the depalletizer.
    from mujindetection.shared import pointcloud as _mujinpointcloud
except Exception:  # pragma: no cover - depends on runtime environment
    _mujinpointcloud = None

try:
    import mujinvisioncommonutilitiesbindings as _visionbindings
except Exception:  # pragma: no cover - depends on runtime environment
    _visionbindings = None


class Detector:
    """RGB-D box detector producing instance masks and valid grasp faces."""

    def __init__(
        self,
        modelType: str = _DEFAULT_MODEL_TYPE,
        modelFilePath: Optional[str] = None,
        labelMap: Optional[dict[int, str]] = None,
    ):
        """Loads the instance segmentation model (see `Segmenter`)."""
        # Imported here so the geometry post-processing can be used without the
        # mujindetection package (segmenter.py imports it at module load time).
        from segmenter import DEFAULT_LABEL_MAP, Segmenter

        self._segmenter = Segmenter(
            modelType=modelType,
            modelFilePath=modelFilePath,
            labelMap=labelMap if labelMap is not None else DEFAULT_LABEL_MAP,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def Detect(
        self,
        rgb: NDArray,
        depth: NDArray,
        intrinsics: CameraIntrinsics,
        container: Optional[Container] = None,
        minConfidenceThreshold: float = 0.5,
        iouThreshold: float = 0.6,
        faceSplitTiltThresholdDeg: float = 30.0,
        graspFaceMaxTiltDeg: float = 75.0,
        horizontalFaceAngleThresholdDeg: float = 30.0,
        minFaceArea: float = 0.01,
        includeTopFacesAsGraspable: bool = False,
        containerMargin: float = 0.02,
    ) -> list[DetectionResult]:
        """Runs the full RGB-D detection + face post-processing pipeline.

        Args:
            rgb: RGB image (H, W, 3) uint8. Must be aligned to `depth`.
            depth: Depth image (H, W). Raw values are scaled by
                `intrinsics.depthScale` to obtain meters.
            intrinsics: Pinhole intrinsics for the depth/RGB image.
            container: Optional container pose/size used to reject faces whose
                center falls outside the bin and to classify top vs side faces.
            minConfidenceThreshold: Segmentation confidence cutoff.
            iouThreshold: Segmentation NMS IOU threshold.
            faceSplitTiltThresholdDeg: A face tilted more than this w.r.t. the
                camera z-axis is assumed to expose a second face, so the instance
                is split. Mirrors `faceSplitTiltThreshould` in the reference.
            graspFaceMaxTiltDeg: A face whose normal is tilted more than this
                w.r.t. the line of sight is too oblique to grasp reliably.
            horizontalFaceAngleThresholdDeg: A face whose normal is within this
                angle of the container up-axis is classified "top", else "side".
            minFaceArea: Minimum face area (m^2) for a valid grasp face.
            includeTopFacesAsGraspable: If False (default), horizontal "top"
                faces are reported but never marked as valid grasp faces
                (顶面意义不大). If True they are judged by the same geometric
                criteria as side faces.
            containerMargin: Margin (m) added to the container bounds when
                testing whether a face center is inside the container.

        Returns:
            One DetectionResult per detected instance, each carrying its
            instance mask and the list of extracted GraspFace objects.
        """
        # 1. Instance segmentation on the RGB image.
        instances = self._segmenter.Infer(
            rgb,
            minConfidenceThreshold=minConfidenceThreshold,
            iouThreshold=iouThreshold,
        )
        results = [DetectionResult(instance=inst) for inst in instances]
        if not instances:
            return results

        # 2. Structured point cloud + normals from depth.
        points = self._DepthToPointCloud(depth, intrinsics)
        normals = self._ComputeNormals(points)
        validMask = numpy.isfinite(points[:, :, 2])

        containerUp = container.upVector if container is not None else None

        # 3. Per-instance face extraction.
        for instanceIndex, result in enumerate(results):
            instanceMask = result.instance.mask & validMask
            if int(instanceMask.sum()) < 3:
                continue

            splittable = result.instance.name in _SPLITTABLE_NAMES
            faceMasks, facePlanes = self._SplitInstanceIntoFaces(
                instanceMask=instanceMask,
                points=points,
                normals=normals,
                validNormalsMask=numpy.isfinite(normals[:, :, 0]),
                intrinsics=intrinsics,
                minFaceArea=minFaceArea,
                faceSplitTiltThresholdDeg=faceSplitTiltThresholdDeg if splittable else 90.0,
            )

            for faceMask, planeModel in zip(faceMasks, facePlanes):
                face = self._BuildGraspFace(
                    instanceIndex=instanceIndex,
                    faceMask=faceMask,
                    planeModel=planeModel,
                    points=points,
                    intrinsics=intrinsics,
                    container=container,
                    containerUp=containerUp,
                    graspFaceMaxTiltDeg=graspFaceMaxTiltDeg,
                    horizontalFaceAngleThresholdDeg=horizontalFaceAngleThresholdDeg,
                    minFaceArea=minFaceArea,
                    includeTopFacesAsGraspable=includeTopFacesAsGraspable,
                    containerMargin=containerMargin,
                )
                if face is not None:
                    result.faces.append(face)

        return results

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _DepthToPointCloud(depth: NDArray, intr: CameraIntrinsics) -> NDArray:
        """Back-projects a depth image into a structured (H, W, 3) point cloud.

        Pixels with non-positive depth become NaN. A multi-channel depth image
        (e.g. a 3-channel PNG/JPG where depth is replicated across channels) is
        collapsed to a single channel.
        """
        z = numpy.asarray(depth)
        if z.ndim == 3:
            z = z[..., 0]
        z = z.astype(numpy.float64) * intr.depthScale
        h, w = z.shape[:2]
        us, vs = numpy.meshgrid(numpy.arange(w), numpy.arange(h))
        x = (us - intr.cx) * z / intr.fx
        y = (vs - intr.cy) * z / intr.fy
        points = numpy.stack((x, y, z), axis=-1)
        points[z <= 0] = numpy.nan
        return points

    @staticmethod
    def _ComputeNormals(points: NDArray) -> NDArray:
        """Estimates per-pixel surface normals from a structured point cloud.

        Uses the mujin C++ binding when available (same as the reference
        pipeline), otherwise a numpy cross-product of neighbour differences.
        """
        if _visionbindings is not None:
            try:
                return numpy.asarray(
                    _visionbindings.ComputeNormals(
                        points.astype(numpy.float32),
                        normalSmoothingSize=30.0,
                        nThreads=4,
                    )
                )
            except Exception as exc:  # pragma: no cover
                log.warning("ComputeNormals binding failed (%s); using numpy fallback.", exc)

        normals = numpy.full_like(points, numpy.nan)
        du = points[1:-1, 2:] - points[1:-1, :-2]   # gradient along columns (x)
        dv = points[2:, 1:-1] - points[:-2, 1:-1]   # gradient along rows (y)
        n = numpy.cross(du, dv)
        norm = numpy.linalg.norm(n, axis=2, keepdims=True)
        with numpy.errstate(invalid="ignore", divide="ignore"):
            n = n / norm
        normals[1:-1, 1:-1] = n
        return normals

    @staticmethod
    def _DetectPlane(maskPoints: NDArray, centroid: NDArray) -> Optional[NDArray]:
        """Fits a plane [a, b, c, d] (a*x+b*y+c*z+d=0, unit normal) to (N,3) points."""
        if _mujinpointcloud is not None:
            try:
                model = _mujinpointcloud.DetectPlane(maskPoints, centroid=centroid)[1]
                if model is not None:
                    return numpy.asarray(model, dtype=numpy.float64)
            except Exception as exc:  # pragma: no cover
                log.warning("DetectPlane failed (%s); using RANSAC fallback.", exc)
        return Detector._RansacPlane(maskPoints)

    @staticmethod
    def _RansacPlane(
        points: NDArray,
        distThreshold: float = 0.005,
        iterations: int = 200,
    ) -> Optional[NDArray]:
        """RANSAC plane fit (fallback for pcl). Returns the dominant planar surface.

        Unlike a plain SVD fit (which averages multiple surfaces), this locks
        onto the largest planar face, matching pcl's behaviour closely enough for
        the tilt gate and face splitting.
        """
        n = len(points)
        if n < 3:
            return None
        rng = numpy.random.default_rng(0)
        bestInliers = None
        for _ in range(iterations):
            idx = rng.choice(n, 3, replace=False)
            p = points[idx]
            normal = numpy.cross(p[1] - p[0], p[2] - p[0])
            norm = numpy.linalg.norm(normal)
            if norm < 1e-9:
                continue
            normal = normal / norm
            d = -float(normal @ p[0])
            inliers = numpy.abs(points @ normal + d) < distThreshold
            if bestInliers is None or inliers.sum() > bestInliers.sum():
                bestInliers = inliers
        if bestInliers is None or bestInliers.sum() < 3:
            return None
        # Refit on the inlier set for a stable model.
        inlierPts = points[bestInliers]
        centroid = inlierPts.mean(axis=0)
        _, _, vh = numpy.linalg.svd(inlierPts - centroid, full_matrices=False)
        normal = vh[-1] / numpy.linalg.norm(vh[-1])
        d = -float(normal @ centroid)
        return numpy.array([normal[0], normal[1], normal[2], d])

    def _SplitInstanceIntoFaces(
        self,
        instanceMask: NDArray,
        points: NDArray,
        normals: NDArray,
        validNormalsMask: NDArray,
        intrinsics: CameraIntrinsics,
        minFaceArea: float,
        faceSplitTiltThresholdDeg: float,
    ) -> tuple[list[NDArray], list[NDArray]]:
        """Returns (faceMasks, facePlanes) for one instance.

        If the instance plane is tilted enough w.r.t. the camera z-axis, the
        mask is split into its (up to two) perpendicular planar faces, exactly as
        `SplitSegmentationsOfTiltedItems` does. Otherwise the whole mask is
        returned as a single face.
        """
        instancePoints = points[instanceMask]
        instancePoints = instancePoints[numpy.isfinite(instancePoints[:, 0])]
        if len(instancePoints) < 3:
            return [], []
        centroid = instancePoints.mean(axis=0)

        planeModel = self._DetectPlane(instancePoints, centroid)
        if planeModel is None:
            return [], []

        # Quick tilt check: how far is the face normal from the camera z-axis?
        cosToCameraZ = abs(planeModel[2])
        minCosToSplit = numpy.cos(numpy.deg2rad(faceSplitTiltThresholdDeg))
        if cosToCameraZ >= minCosToSplit:
            # Not tilted enough to expose a second face; treat as one face.
            return [instanceMask], [planeModel]

        # Tilted: try to split into two perpendicular faces.
        pixelSize = float(centroid[2]) / ((intrinsics.fx + intrinsics.fy) / 2.0)
        faceMasks, facePlanes = self._SegmentFaces(
            component=instanceMask,
            componentPlaneModel=planeModel,
            points=points,
            normals=normals,
            validNormalsMask=validNormalsMask,
            pixelSize=pixelSize,
            centroid=centroid,
            minFaceArea=minFaceArea,
        )
        if not faceMasks:
            # Splitting did not find two valid faces; keep the original.
            return [instanceMask], [planeModel]
        return faceMasks, facePlanes

    def _SegmentFaces(
        self,
        component: NDArray,
        componentPlaneModel: NDArray,
        points: NDArray,
        normals: NDArray,
        validNormalsMask: NDArray,
        pixelSize: float,
        centroid: NDArray,
        minFaceArea: float,
    ) -> tuple[list[NDArray], list[NDArray]]:
        """Splits a multi-face component into perpendicular planar faces.

        Primary path delegates to the reference
        `pointcloud.SegmentBoxFacesIteratively`. The numpy fallback clusters
        pixels by how well their normal matches the dominant plane, then refits
        a plane to the leftover pixels.
        """
        if _mujinpointcloud is not None:
            try:
                hasValidFaces, faceMaskList, facePlaneList = (
                    _mujinpointcloud.SegmentBoxFacesIteratively(
                        component=component,
                        componentPlaneModel=list(componentPlaneModel),
                        points=points,
                        normals=normals,
                        nonNanNormalsMask2D=validNormalsMask,
                        pixelSizeOfComponent=pixelSize,
                        centroid3DOfComponent=centroid,
                        minAreaFace=minFaceArea,
                        maxAngleDifferenceDegInSameFace=30,
                        minAngleDifferenceDegBetweenDifferentFaces=80,
                        dilateKernelSize=7,
                        openKernelSize=3,
                        maxFaceCountToReturn=2,
                        deformationSearchDiameter=0,
                    )
                )
                if hasValidFaces and faceMaskList:
                    return (
                        [numpy.asarray(m, dtype=bool) for m in faceMaskList],
                        [numpy.asarray(p, dtype=numpy.float64) for p in facePlaneList],
                    )
                return [], []
            except Exception as exc:  # pragma: no cover
                log.warning("SegmentBoxFacesIteratively failed (%s); using numpy fallback.", exc)

        return self._SegmentFacesNumpy(
            component=component,
            componentPlaneModel=componentPlaneModel,
            points=points,
            normals=normals,
            validNormalsMask=validNormalsMask,
            pixelSize=pixelSize,
            centroid=centroid,
            minFaceArea=minFaceArea,
        )

    @staticmethod
    def _SegmentFacesNumpy(
        component: NDArray,
        componentPlaneModel: NDArray,
        points: NDArray,
        normals: NDArray,
        validNormalsMask: NDArray,
        pixelSize: float,
        centroid: NDArray,
        minFaceArea: float,
    ) -> tuple[list[NDArray], list[NDArray]]:
        """Pure-numpy two-face split (fallback for SegmentBoxFacesIteratively)."""
        import cv2

        minFacePixels = max(minFaceArea, 0.015 ** 2) / (pixelSize ** 2)
        minCosSameFace = numpy.cos(numpy.deg2rad(30.0))
        minCosDiffFace = numpy.cos(numpy.deg2rad(80.0))  # faces must differ > 80deg
        openKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        work = component & validNormalsMask

        def _largestBlob(mask: NDArray) -> NDArray:
            mask = cv2.morphologyEx(mask.view("uint8"), cv2.MORPH_OPEN, openKernel)
            num, labels = cv2.connectedComponents(mask)
            if num <= 1:
                return numpy.zeros_like(mask, dtype=bool)
            sizes = [(labels == i).sum() for i in range(1, num)]
            return labels == (1 + int(numpy.argmax(sizes)))

        # Face A: pixels whose normal matches the dominant component plane.
        normA = componentPlaneModel[:3]
        cosA = numpy.full(component.shape, -1.0)
        cosA[work] = numpy.abs(normals[work] @ normA)
        maskA = _largestBlob((cosA > minCosSameFace) & component)
        if maskA.sum() < minFacePixels:
            return [], []

        # Face B: fit the dominant plane of the remaining points.
        remaining = component & ~maskA & validNormalsMask
        remPts = points[remaining]
        remPts = remPts[numpy.isfinite(remPts[:, 0])]
        if len(remPts) < 3:
            return [], []
        planeB = Detector._RansacPlane(remPts)
        if planeB is None:
            return [], []
        normB = planeB[:3]
        if abs(float(normA @ normB)) > minCosDiffFace:
            return [], []  # second face not perpendicular enough; not a tilted box
        cosB = numpy.full(component.shape, -1.0)
        cosB[remaining] = numpy.abs(normals[remaining] @ normB)
        maskB = _largestBlob((cosB > minCosSameFace) & component & ~maskA)
        if maskB.sum() < minFacePixels:
            return [], []
        # Largest face first, matching the reference convention.
        if maskA.sum() >= maskB.sum():
            return [maskA, maskB], [componentPlaneModel, planeB]
        return [maskB, maskA], [planeB, componentPlaneModel]

    def _BuildGraspFace(
        self,
        instanceIndex: int,
        faceMask: NDArray,
        planeModel: NDArray,
        points: NDArray,
        intrinsics: CameraIntrinsics,
        container: Optional[Container],
        containerUp: Optional[NDArray],
        graspFaceMaxTiltDeg: float,
        horizontalFaceAngleThresholdDeg: float,
        minFaceArea: float,
        includeTopFacesAsGraspable: bool,
        containerMargin: float,
    ) -> Optional[GraspFace]:
        """Computes the 3D OBB, classification and grasp validity of one face."""
        import cv2

        facePoints3D = points[faceMask]
        facePoints3D = facePoints3D[numpy.isfinite(facePoints3D[:, 0])]
        if len(facePoints3D) < 3:
            return None

        normal = planeModel[:3] / numpy.linalg.norm(planeModel[:3])
        center3D = facePoints3D.mean(axis=0)

        # Orient the normal toward the camera (camera at origin looking +z).
        if normal @ center3D > 0:
            normal = -normal

        # Build an in-plane basis to compute an oriented bounding rectangle.
        ref = numpy.array([0.0, 0.0, 1.0])
        if abs(normal @ ref) > 0.9:
            ref = numpy.array([0.0, 1.0, 0.0])
        u = ref - (ref @ normal) * normal
        u = u / numpy.linalg.norm(u)
        v = numpy.cross(normal, u)

        local = numpy.stack((
            (facePoints3D - center3D) @ u,
            (facePoints3D - center3D) @ v,
        ), axis=1).astype(numpy.float32)
        rect = cv2.minAreaRect(local)
        (rw, rh) = rect[1]
        if rw <= 0 or rh <= 0:
            return None
        boxPts2D = cv2.boxPoints(rect)  # (4, 2) in local plane coords
        corners3D = center3D + numpy.outer(boxPts2D[:, 0], u) + numpy.outer(boxPts2D[:, 1], v)

        # Project corners back into the image for visualization / 2D consumers.
        corners2D = numpy.stack((
            corners3D[:, 0] / corners3D[:, 2] * intrinsics.fx + intrinsics.cx,
            corners3D[:, 1] / corners3D[:, 2] * intrinsics.fy + intrinsics.cy,
        ), axis=1)

        longEdge, shortEdge = max(rw, rh), min(rw, rh)
        area = longEdge * shortEdge

        # Tilt of the face w.r.t. the line of sight to its center.
        lineOfSight = center3D / numpy.linalg.norm(center3D)
        tiltWrtCamera = numpy.rad2deg(numpy.arccos(numpy.clip(abs(normal @ lineOfSight), 0.0, 1.0)))

        # Classify top (horizontal) vs side (vertical) using the container up-axis.
        faceType = "side"
        if containerUp is not None:
            cosUp = abs(float(normal @ containerUp))
            if cosUp > numpy.cos(numpy.deg2rad(horizontalFaceAngleThresholdDeg)):
                faceType = "top"

        # Grasp-face validity.
        isValid = area >= minFaceArea and tiltWrtCamera <= graspFaceMaxTiltDeg
        if faceType == "top" and not includeTopFacesAsGraspable:
            isValid = False
        if container is not None:
            inside = bool(container.ContainsPoints(center3D[None, :], margin=containerMargin)[0])
            isValid = isValid and inside

        return GraspFace(
            instanceIndex=instanceIndex,
            mask=faceMask.astype(numpy.bool_),
            center3D=center3D,
            normal3D=normal,
            corners3D=corners3D,
            corners2D=corners2D,
            dimensions=(float(longEdge), float(shortEdge)),
            area=float(area),
            faceType=faceType,
            tiltWrtCameraDeg=float(tiltWrtCamera),
            isValidGraspFace=bool(isValid),
        )


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def DrawDetections(
    rgb: NDArray,
    results: list[DetectionResult],
    onlyValidGraspFaces: bool = False,
) -> NDArray:
    """Draws instance masks and per-face OBBs / normals on a copy of `rgb`.

    Valid grasp faces are outlined in green, invalid ones in gray, and each
    face is annotated with its type and dimensions.
    """
    import cv2

    canvas = rgb.copy()
    overlay = rgb.copy()
    palette = [(231, 97, 37), (25, 20, 210), (60, 180, 75), (255, 130, 48),
               (240, 50, 230), (66, 212, 245), (180, 30, 145), (0, 165, 255)]
    for i, r in enumerate(results):
        overlay[r.instance.mask] = palette[i % len(palette)]
    canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0.0)

    for r in results:
        for face in r.faces:
            if onlyValidGraspFaces and not face.isValidGraspFace:
                continue
            color = (60, 200, 60) if face.isValidGraspFace else (150, 150, 150)
            pts = numpy.round(face.corners2D).astype(numpy.int32)
            cv2.polylines(canvas, [pts], True, color, 2)
            cx, cy = pts.mean(axis=0).astype(int)
            caption = f"{face.faceType} {face.dimensions[0]:.2f}x{face.dimensions[1]:.2f}"
            cv2.putText(canvas, caption, (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas


def _main() -> None:
    import argparse

    import cv2

    parser = argparse.ArgumentParser(
        description="Detect boxes and their valid grasp faces from an RGB-D pair."
    )
    parser.add_argument("rgbPath", help="Path to the aligned RGB image.")
    parser.add_argument("depthPath", help="Path to the aligned depth image (16-bit PNG or .npy).")
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--depth-scale", dest="depthScale", type=float, default=0.001,
                        help="Multiply raw depth by this to get meters (0.001 for uint16 mm).")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--model-type", dest="modelType", default=_DEFAULT_MODEL_TYPE)
    parser.add_argument("--model", default=None, help="Explicit .onnx weights path.")
    parser.add_argument("-o", "--output", help="If set, save an annotated image here.")
    args = parser.parse_args()

    bgr = cv2.imread(args.rgbPath, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"Failed to read RGB image: {args.rgbPath}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if args.depthPath.endswith(".npy"):
        depth = numpy.load(args.depthPath)
    else:
        depth = cv2.imread(args.depthPath, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise SystemExit(f"Failed to read depth image: {args.depthPath}")

    intr = CameraIntrinsics(
        fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
        width=rgb.shape[1], height=rgb.shape[0], depthScale=args.depthScale,
    )

    detector = Detector(modelType=args.modelType, modelFilePath=args.model)
    results = detector.Detect(rgb, depth, intr, minConfidenceThreshold=args.conf)

    nFaces = sum(len(r.faces) for r in results)
    nValid = sum(1 for r in results for f in r.faces if f.isValidGraspFace)
    print(f"Detected {len(results)} instance(s), {nFaces} face(s), {nValid} valid grasp face(s):")
    for i, r in enumerate(results):
        print(f"  [{i}] {r.instance.name} score={r.instance.score:.3f} faces={len(r.faces)}")
        for j, f in enumerate(r.faces):
            print(
                f"      face {j}: type={f.faceType} valid={f.isValidGraspFace} "
                f"dims={f.dimensions[0]:.3f}x{f.dimensions[1]:.3f}m "
                f"tilt={f.tiltWrtCameraDeg:.1f}deg "
                f"center=({f.center3D[0]:.3f},{f.center3D[1]:.3f},{f.center3D[2]:.3f})"
            )

    if args.output:
        annotated = DrawDetections(rgb, results)
        cv2.imwrite(args.output, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        print(f"Saved annotated image to {args.output}")


if __name__ == "__main__":
    _main()
