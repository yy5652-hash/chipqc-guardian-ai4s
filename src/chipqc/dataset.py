"""Join spreadsheet IDs to image bytes and build an auditable feature table."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path, PurePosixPath
import zipfile

import pandas as pd

from .features import FEATURE_COLUMNS, extract_quality_features
from .metadata import require_trainable_metadata


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _directory_label(parts: tuple[str, ...]) -> int | None:
    tags = {part.lower() for part in parts}
    if "good" in tags and "bad" not in tags:
        return 1
    if "bad" in tags and "good" not in tags:
        return 0
    return None


class ImageIndex:
    """Read images directly from a complete ZIP or an unpacked directory."""

    def __init__(self, source: str | Path):
        self.source = Path(source)
        if not self.source.exists():
            raise FileNotFoundError(f"Image source does not exist: {self.source}")
        self.is_zip = self.source.is_file()
        if self.is_zip:
            if self.source.suffix.lower() != ".zip" or not zipfile.is_zipfile(self.source):
                raise ValueError(f"Image ZIP is incomplete or invalid: {self.source}")
            with zipfile.ZipFile(self.source) as archive:
                names = [name for name in archive.namelist() if not name.endswith("/")]
        elif self.source.is_dir():
            names = [path.relative_to(self.source).as_posix() for path in self.source.rglob("*") if path.is_file()]
        else:
            raise ValueError(f"Unsupported image source: {self.source}")
        self.by_id: dict[str, list[str]] = defaultdict(list)
        for name in names:
            item = PurePosixPath(name)
            if item.suffix.lower() in IMAGE_SUFFIXES and "__MACOSX" not in item.parts:
                self.by_id[item.stem.casefold()].append(name)

    def matches(self, image_id: str) -> list[str]:
        return self.by_id.get(image_id.casefold(), [])

    def read(self, member: str) -> bytes:
        if self.is_zip:
            with zipfile.ZipFile(self.source) as archive:
                return archive.read(member)
        return (self.source / member).read_bytes()

    @property
    def image_count(self) -> int:
        return sum(map(len, self.by_id.values()))


def prepare_feature_table(metadata: pd.DataFrame, index: ImageIndex) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (usable features, all-row manifest, matching summary).

    The ZIP's good/bad directories are used only to detect label conflicts, not
    as training features. The ZIP's train/test directories are never used for
    splitting; groups from image IDs govern the split instead.
    """
    require_trainable_metadata(metadata)
    feature_rows: list[dict] = []
    manifest_rows: list[dict] = []
    for row in metadata.itertuples(index=False):
        image_id = str(row.image_id)
        matches = index.matches(image_id)
        member = matches[0] if len(matches) == 1 else None
        status = "matched"
        error = None
        if not matches:
            status = "missing_image"
        elif len(matches) > 1:
            status = "ambiguous_image_id"
        elif (directory_label := _directory_label(PurePosixPath(member).parts)) is not None and directory_label != int(row.label_good):
            status = "label_conflict"
        if status == "matched":
            try:
                features = extract_quality_features(index.read(member))
            except Exception as exc:  # Keep one bad image from hiding the rest of the audit.
                status = "decode_or_feature_error"
                error = f"{type(exc).__name__}: {exc}"
            else:
                feature_rows.append({
                    "image_id": image_id,
                    "group_id": str(row.group_id),
                    "cell_type": str(row.cell_type),
                    "label_good": int(row.label_good),
                    **features,
                })
        manifest_rows.append({
            "image_id": image_id,
            "group_id": str(row.group_id),
            "cell_type": str(row.cell_type),
            "label_good": int(row.label_good),
            "member": member or "",
            "status": status,
            "error": error or "",
        })
    feature_columns = ["image_id", "group_id", "cell_type", "label_good", *FEATURE_COLUMNS]
    feature_frame = pd.DataFrame(feature_rows, columns=feature_columns)
    manifest = pd.DataFrame(manifest_rows)
    status_counts = {str(key): int(value) for key, value in manifest["status"].value_counts().items()}
    summary = {
        "source_image_count": index.image_count,
        "metadata_record_count": len(metadata),
        "matched_feature_count": len(feature_frame),
        "status_counts": status_counts,
        "image_source": str(index.source),
        "image_source_bytes": index.source.stat().st_size if index.is_zip else None,
    }
    return feature_frame, manifest, summary
