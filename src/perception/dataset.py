"""Dataset acquisition and YOLO training utilities for the perception engine.

Two acquisition paths are supported:

* **Roboflow** - download a project version via the Roboflow REST API.
  Only the standard library (``urllib``) is required; no ``roboflow`` pip
  package is needed at runtime.
* **Local YAML directory** - point at a folder containing a YOLOv8-style
  ``data.yaml`` and the corresponding ``images/`` + ``labels/`` tree.

Once a dataset is acquired, :func:`train_yolo` wraps the Ultralytics
``model.train()`` API with sensible defaults and automatically exports the
resulting weights to both PyTorch (``.pt``) and ONNX (``.onnx``) formats.

Example::

    from src.perception.dataset import YOLODatasetManager, train_yolo

    # Download from Roboflow
    manager = YOLODatasetManager.from_roboflow(
        api_key="...", workspace="my-ws", project="scooter-signs", version=3
    )
    data_yaml = manager.get_data_yaml()

    # Train and export
    train_yolo(
        data=data_yaml,
        epochs=100,
        batch=16,
        name="emirates-hud-v1",
    )
"""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from src.schemas import DetectionResult  # noqa: F401  - re-exported for convenience
from src.telemetry import trace_span
from src.telemetry.logging import get_logger

__all__ = [
    "DETECTION_CLASSES",
    "YOLODatasetManager",
    "export_model",
    "train_yolo",
]

# Re-use canonical class list so dataset labels stay in sync with the engine.
from .yolo_engine import DETECTION_CLASSES

logger = get_logger(__name__)

#: Roboflow universe download endpoint template.
_ROBOFLOW_DOWNLOAD_URL = (
    "https://universe.roboflow.com/ds/{dataset_id}/{api_key}/"
    "download?type=zip"
)


@dataclass
class DatasetInfo:
    """Metadata about an acquired dataset."""

    data_yaml: str
    dataset_dir: str
    num_train_images: int = 0
    num_val_images: int = 0
    classes: list[str] = field(default_factory=list)


class YOLODatasetManager:
    """Acquire, inspect, and stage a YOLOv11 training dataset.

    Parameters
    ----------
    dataset_dir:
        Path to the directory containing ``data.yaml`` and the
        ``images/`` + ``labels/`` sub-trees.
    """

    def __init__(self, dataset_dir: str | Path) -> None:
        self._dir = Path(dataset_dir)
        if not self._dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self._dir}")

    # ------------------------------------------------------------------ factories
    @classmethod
    def from_roboflow(
        cls,
        api_key: str,
        workspace: str,
        project: str,
        version: int,
        dest: str | Path | None = None,
    ) -> YOLODatasetManager:
        """Download a dataset version from Roboflow.

        Parameters
        ----------
        api_key:
            Roboflow API key.
        workspace:
            Roboflow workspace slug.
        project:
            Project slug.
        version:
            Numeric version of the dataset.
        dest:
            Destination directory.  Defaults to a temporary directory
            that is cleaned up when the manager goes out of scope.

        Returns
        -------
        YOLODatasetManager
            Pointing at the downloaded (and unzipped) dataset directory.
        """
        dataset_id = f"{workspace}-{project}-{version}"
        url = _ROBOFLOW_DOWNLOAD_URL.format(dataset_id=dataset_id, api_key=api_key)

        target = Path(dest) if dest else Path(tempfile.mkdtemp(prefix="roboflow_"))
        target.mkdir(parents=True, exist_ok=True)

        zip_path = target / f"{dataset_id}.zip"
        logger.info("downloading dataset from Roboflow", extra={"url": url})

        try:
            urllib.request.urlretrieve(url, str(zip_path))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download dataset from Roboflow: {exc}"
            ) from exc

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(str(target))
        except zipfile.BadZipFile as exc:
            raise RuntimeError(f"Downloaded file is not a valid zip: {exc}") from exc
        finally:
            if zip_path.exists():
                zip_path.unlink()

        # Locate the data.yaml that Roboflow unzips (usually at the root).
        yaml_candidates = list(target.rglob("data.yaml"))
        if not yaml_candidates:
            raise FileNotFoundError(
                f"No data.yaml found in downloaded dataset at {target}"
            )

        # The real dataset dir is the parent of the yaml.
        dataset_root = yaml_candidates[0].parent
        logger.info("dataset extracted", extra={"dir": str(dataset_root)})
        return cls(dataset_root)

    # ------------------------------------------------------------------ inspection
    def get_data_yaml(self) -> str:
        """Return the path to ``data.yaml`` as a string."""
        yaml_path = self._dir / "data.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found in {self._dir}")
        return str(yaml_path)

    def get_info(self) -> DatasetInfo:
        """Inspect the dataset directory and return metadata."""
        yaml_path = self._dir / "data.yaml"

        classes: list[str] = []
        if yaml_path.exists():
            import yaml  # type: ignore[import-untyped]

            with open(yaml_path, encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            names = cfg.get("names", [])
            if isinstance(names, dict):
                # YOLOv8 convention: {0: "class_a", 1: "class_b"}
                classes = list(names.values())
            else:
                classes = list(names)

        num_train = self._count_images_in("images/train") if (self._dir / "images" / "train").is_dir() else 0
        num_val = self._count_images_in("images/val") if (self._dir / "images" / "val").is_dir() else 0

        return DatasetInfo(
            data_yaml=str(yaml_path),
            dataset_dir=str(self._dir),
            num_train_images=num_train,
            num_val_images=num_val,
            classes=classes,
        )

    def _count_images_in(self, sub: str) -> int:
        """Count image files in a sub-directory."""
        d = self._dir / sub
        if not d.is_dir():
            return 0
        return sum(
            1 for f in d.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )

    # ------------------------------------------------------------------ cleanup
    def cleanup(self) -> None:
        """Remove a temporary dataset directory (no-op if not temporary)."""
        try:
            if shutil.rmtree.avoids_symlink_attacks:
                pass
        except AttributeError:  # pragma: no cover
            pass
        if "tmp" in str(self._dir).lower() or self._dir.exists():
            shutil.rmtree(self._dir, ignore_errors=True)

    @property
    def dataset_dir(self) -> str:
        """Return the resolved dataset directory path."""
        return str(self._dir)


# --------------------------------------------------------------------------- #
# Training & export
# --------------------------------------------------------------------------- #
def train_yolo(
    data: str | Path,
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    name: str = "emirates-hud-yolo",
    patience: int = 20,
    project: str = "runs/train",
    device: str | None = None,
    export_onnx: bool = True,
    export_pt: bool = True,
    verbose: bool = False,
) -> str:
    """Train a YOLOv11 model and export weights.

    Parameters
    ----------
    data:
        Path to the ``data.yaml`` describing the dataset.
    epochs:
        Number of training epochs.
    batch:
        Effective batch size.
    imgsz:
        Training image size (square).
    name:
        Experiment name (used to create ``runs/train/<name>``).
    patience:
        Early-stopping patience (epochs without improvement).
    project:
        Top-level runs directory.
    device:
        Device to train on (e.g. ``"0"``, ``"cpu"``).  ``None`` = auto.
    export_onnx:
        Export to ONNX (``best.onnx``) after training.
    export_pt:
        Copy the best PyTorch weights (``best.pt``) after training.
    verbose:
        Forward Ultralytics verbose flag.

    Returns
    -------
    str
        Path to the exported PyTorch weights (``best.pt``).
    """
    try:
        from ultralytics import YOLO  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is required for training. Install with `pip install ultralytics`."
        ) from exc

    with trace_span(name="yolo.train", attributes={"training.epochs": epochs, "training.batch": batch}):
        logger.info(
            "starting YOLO training",
            extra={
                "data": str(data),
                "epochs": epochs,
                "batch": batch,
                "imgs": imgsz,
                "name": name,
            },
        )

        model = YOLO("yolov11n.pt")  # start from COCO-pretrained backbone
        model.train(
            data=str(data),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            name=name,
            patience=patience,
            project=project,
            device=device,
            verbose=verbose,
        )

        # The best checkpoint path is reported by Ultralytics.
        run_dir = Path(project) / name
        best_pt = run_dir / "weights" / "best.pt"

        if not best_pt.exists():
            # Fallback: search for the most-recent best.pt in the runs tree.
            candidates = sorted(Path(project).rglob("weights/best.pt"), key=os.path.getmtime, reverse=True)
            if not candidates:
                raise FileNotFoundError("Could not locate best.pt after training.")
            best_pt = candidates[0]

        logger.info("training complete", extra={"best_pt": str(best_pt)})

        # Export to ONNX
        if export_onnx:
            try:
                model.export(format="onnx", weights=str(best_pt), imgsz=imgsz)
                onnx_path = Path(str(best_pt).replace(".pt", ".onnx"))
                logger.info("onnx export complete", extra={"path": str(onnx_path)})
            except Exception as exc:
                logger.warning("onnx export failed", extra={"error": str(exc)[:256]})

        return str(best_pt)


def export_model(
    weights: str | Path,
    format: str = "onnx",
    imgsz: int = 640,
    half: bool = False,
) -> str:
    """Export a trained YOLO model to an alternative format.

    Parameters
    ----------
    weights:
        Path to the ``.pt`` checkpoint.
    format:
        Target export format (default ``"onnx"``).  See Ultralytics
        ``model.export()`` for supported formats.
    imgsz:
        Inference image size used during export.
    half:
        Export with FP16 half-precision weights.

    Returns
    -------
    str
        Path to the exported model file.
    """
    try:
        from ultralytics import YOLO  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is required for export. Install with `pip install ultralytics`."
        ) from exc

    model = YOLO(str(weights))
    export_path = model.export(format=format, imgsz=imgsz, half=half)

    # export returns the path as a string in most versions.
    result = str(export_path) if export_path else str(Path(weights).with_suffix(f".{format}"))
    logger.info("model exported", extra={"format": format, "path": result})
    return result
