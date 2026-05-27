"""Instance segmentation wrapper.

Thin wrapper around the mujindetection instance segmentation models
(`yolo8s` / `rfdetr0m`) that exposes a simple inference API for a single
RGB image. The concrete model class is resolved from the weight file name
via `MODEL_REGISTRY`, so both architectures share the same code path.

Intended to be run inside the mujin docker container, e.g.:
    docker exec -it 734699238a41 bash -lc "python3 /path/to/segmenter.py <image>"

The login shell (`bash -lc`) is required so that PYTHONPATH picks up
`/opt/lib/python3.13/site-packages` where `mujindetection` is installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy
from numpy.typing import NDArray

from mujindetection.shared.instancesegmentation.models.modelregistry import MODEL_REGISTRY


# Default weights shipped with the mujindetectors install in the container,
# keyed by model type. Used when no explicit weight path is given.
DEFAULT_MODEL_PATHS = {
    "yolo8s": (
        "/opt/share/mujindetectors/dnnmodels/instancesegmentation/"
        "depallet/yolo8s/yolo8s_15.6.5.onnx"
    ),
    "rfdetr0m": (
        "/opt/share/mujindetectors/dnnmodels/instancesegmentation/"
        "depallet/rfdetr0m/rfdetr0m_0.0.10.onnx"
    ),
}

# Which model type to use when the caller does not specify one.
DEFAULT_MODEL_TYPE = "rfdetr0m"

# Label map shared by the depallet yolo8s and rfdetr0m weights.
DEFAULT_LABEL_MAP = {
    0: "box",
    1: "irregularMultiPack",
    2: "splitterSheet",
    3: "pallet",
}


@dataclass
class InstanceResult:
    """One detected instance."""
    bbox: tuple[float, float, float, float]  # xyxy in original image coordinates
    label: int
    name: str
    score: float
    mask: NDArray[numpy.bool_]  # (H, W) bool, original image shape


class Segmenter:
    """Instance segmentation inference wrapper for a single RGB image."""

    def __init__(
        self,
        modelType: str = DEFAULT_MODEL_TYPE,
        modelFilePath: Optional[str] = None,
        labelMap: Optional[dict[int, str]] = None,
    ):
        """Loads an instance segmentation ONNX model.

        Args:
            modelType: Which model to use, one of `DEFAULT_MODEL_PATHS` keys
                (e.g. "yolo8s" or "rfdetr0m"). Selects the default weight path.
            modelFilePath: Explicit path to the .onnx weights file. If given,
                overrides the default path for `modelType`. The model class is
                resolved from the file name prefix via `MODEL_REGISTRY`.
            labelMap: Optional {classId: className} mapping. Defaults to
                the depallet 4-class map.
        """
        if modelFilePath is None:
            if modelType not in DEFAULT_MODEL_PATHS:
                raise ValueError(
                    f"Unknown modelType '{modelType}'. "
                    f"Expected one of {list(DEFAULT_MODEL_PATHS)}."
                )
            modelFilePath = DEFAULT_MODEL_PATHS[modelType]

        # Resolve the concrete model class from the weight file name prefix,
        # e.g. "rfdetr0m_0.0.10.onnx" -> "rfdetr0m". Same convention as Segmentor.
        modelName = Path(modelFilePath).name.split("_")[0]
        if modelName not in MODEL_REGISTRY:
            raise ValueError(
                f"No model registered for weight '{modelName}'. "
                f"Registered: {list(MODEL_REGISTRY)}."
            )

        self._model = MODEL_REGISTRY[modelName](modelFilePath=modelFilePath)
        self._labelMap = labelMap if labelMap is not None else DEFAULT_LABEL_MAP

    def Infer(
        self,
        image: NDArray,
        minConfidenceThreshold: float = 0.5,
        iouThreshold: float = 0.6,
    ) -> list[InstanceResult]:
        """Runs instance segmentation on a single RGB image.

        Args:
            image: RGB image array of shape (H, W, 3), dtype uint8.
            minConfidenceThreshold: Detection confidence cutoff in [0, 1].
            iouThreshold: NMS IOU threshold in [0, 1].

        Returns:
            List of `InstanceResult`. Empty list if no detection passes the thresholds.
            Bboxes and masks are in the original image's resolution.
        """
        bboxes, labels, scores, masks = self._model.Predict(
            image=image,
            minConfidenceThreshold=minConfidenceThreshold,
            iouThreshold=iouThreshold,
        )

        if len(bboxes) == 0:
            return []

        results: list[InstanceResult] = []
        for bbox, label, score, mask in zip(bboxes, labels, scores, masks):
            labelInt = int(label)
            results.append(
                InstanceResult(
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    label=labelInt,
                    name=self._labelMap.get(labelInt, str(labelInt)),
                    score=float(score),
                    mask=mask.astype(numpy.bool_),
                )
            )
        return results


# Distinct BGR colors cycled per instance for visualization.
_PALETTE = [
    (231, 97, 37),   # mujin orange
    (25, 20, 210),   # mujin red
    (60, 180, 75),
    (255, 130, 48),
    (240, 50, 230),
    (66, 212, 245),
    (180, 30, 145),
    (0, 165, 255),
]


def DrawResults(
    image: NDArray,
    results: list[InstanceResult],
    maskAlpha: float = 0.45,
) -> NDArray:
    """Overlays instance segmentation results on an image.

    Args:
        image: RGB image array of shape (H, W, 3), dtype uint8 (the input image).
        results: Instances returned by `Segmenter.Infer`.
        maskAlpha: Blend weight for the mask overlay in [0, 1].

    Returns:
        A new RGB image with masks, bounding boxes and labels drawn.
    """
    import cv2

    canvas = image.copy()
    overlay = image.copy()
    for i, r in enumerate(results):
        color = _PALETTE[i % len(_PALETTE)]
        overlay[r.mask] = color

    canvas = cv2.addWeighted(overlay, maskAlpha, canvas, 1.0 - maskAlpha, 0.0)

    for i, r in enumerate(results):
        color = _PALETTE[i % len(_PALETTE)]
        x1, y1, x2, y2 = (int(round(v)) for v in r.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        caption = f"[{i}] {r.name} {r.score:.2f}"
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        textTop = max(0, y1 - th - baseline)
        cv2.rectangle(canvas, (x1, textTop), (x1 + tw, textTop + th + baseline), color, -1)
        cv2.putText(
            canvas, caption, (x1, textTop + th),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return canvas


def _main() -> None:
    import argparse

    import cv2

    parser = argparse.ArgumentParser(description="Run instance segmentation on one image.")
    parser.add_argument("imagePath", help="Path to an RGB or BGR image file.")
    parser.add_argument(
        "--model-type", dest="modelType",
        choices=sorted(DEFAULT_MODEL_PATHS), default=DEFAULT_MODEL_TYPE,
        help="Which model to use (selects the default weight path).",
    )
    parser.add_argument(
        "--model", default=None,
        help="Explicit path to .onnx weights (overrides --model-type's default path).",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="Min confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IOU threshold.")
    parser.add_argument(
        "-o", "--output",
        help="If set, save the input image with segmentation results drawn to this path.",
    )
    args = parser.parse_args()

    bgr = cv2.imread(args.imagePath, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"Failed to read image: {args.imagePath}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    segmenter = Segmenter(modelType=args.modelType, modelFilePath=args.model)
    results = segmenter.Infer(rgb, minConfidenceThreshold=args.conf, iouThreshold=args.iou)

    print(f"Detected {len(results)} instance(s) on {args.imagePath} (shape={rgb.shape}):")
    for i, r in enumerate(results):
        x1, y1, x2, y2 = r.bbox
        print(
            f"  [{i}] {r.name} (id={r.label}) score={r.score:.3f} "
            f"bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) mask={r.mask.shape} pixels={int(r.mask.sum())}"
        )

    if args.output:
        annotated = DrawResults(rgb, results)
        cv2.imwrite(args.output, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        print(f"Saved annotated image to {args.output}")


if __name__ == "__main__":
    _main()
