"""ChipQC Guardian: acquisition gate -> culture-quality score -> risk-controlled decision, with evidence.

Three outcomes. PASS: the frame may be used without a person looking (the share of bad frames among passed frames
is held to a chosen target). REACQUIRE: the frame is not a usable observation. REVIEW: a person decides; the score
only orders the queue. Automatic rejection is deliberately absent: on unseen dates low scores were not reliable
enough (see results/system_outcomes.json).

A released model is a folder with `model.json` (head weights, calibrator, thresholds, acquisition limits and the
evaluation it was released with) and `reference.npz` (embeddings of the labelled reference frames, used to show
the most similar good and bad examples). No pickles: everything is JSON or plain arrays.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .descriptors import acquisition_descriptors, acquisition_flags
from .frames import Source, open_frame, to_grey, to_model_input

PASS, REVIEW, REACQUIRE = "PASS", "REVIEW", "REACQUIRE"


@dataclass
class Assessment:
    decision: str
    p_good: float
    reasons: list[str]
    acquisition: dict[str, float]
    evidence: np.ndarray                      # h x w; each cell's share of the calibrated log-odds (positive = looks good)
    neighbours: list[dict] = field(default_factory=list)

    def to_record(self) -> dict:
        return {"decision": self.decision, "p_good": round(self.p_good, 4), "reasons": self.reasons,
                "acquisition": {k: round(v, 4) for k, v in self.acquisition.items()}, "neighbours": self.neighbours}


class Guardian:
    def __init__(self, model_dir: str | Path, device: str | None = None):
        self.dir = Path(model_dir)
        self.spec = json.loads((self.dir / "model.json").read_text())
        self.v = np.asarray(self.spec["cell_weights"], dtype=np.float32)
        self.b = float(self.spec["cell_bias"])
        self.platt = tuple(self.spec["platt"])
        self.limits = self.spec["acquisition_limits"]
        self.default_target = str(self.spec["default_error_target"])
        self._device, self._encoder, self._ref = device, None, None

    @property
    def encoder(self):
        if self._encoder is None:
            from .backbone import FrameEncoder
            self._encoder = FrameEncoder(self.spec["backbone"], self._device)
        return self._encoder

    @property
    def reference(self):
        if self._ref is None and (self.dir / "reference.npz").exists():
            r = np.load(self.dir / "reference.npz", allow_pickle=False)
            e = r["embeddings"].astype(np.float32)
            self._ref = {"unit": e / np.linalg.norm(e, axis=1, keepdims=True), "image_id": r["image_id"], "label_good": r["label_good"], "cell_line": r["cell_line"]}
        return self._ref

    @property
    def atlas_dir(self) -> Path:
        return (self.dir / self.spec.get("atlas", "atlas")).resolve()

    def pass_threshold(self, error_target: float | str | None = None) -> float:
        return float(self.spec["thresholds"][str(error_target) if error_target is not None else self.default_target]["t_pass"])

    def score_cells(self, cells: np.ndarray) -> tuple[float, np.ndarray]:
        """cells: h x w x C descriptors of one frame -> (P(good), evidence map). The map averages to the frame's log-odds."""
        a, c = self.platt
        evidence = a * (cells @ self.v + self.b) + c
        return float(1.0 / (1.0 + np.exp(-evidence.mean()))), evidence

    def neighbours(self, embedding: np.ndarray, k: int = 3) -> list[dict]:
        ref = self.reference
        if ref is None:
            return []
        mu, sd = np.asarray(self.spec["embedding_mean"], dtype=np.float32), np.asarray(self.spec["embedding_std"], dtype=np.float32)
        q = (embedding - mu) / sd
        sim = ref["unit"] @ (q / np.linalg.norm(q))
        out = []
        for label in (1, 0):
            idx = np.flatnonzero(ref["label_good"] == label)
            for i in idx[np.argsort(-sim[idx])[:k]]:
                out.append({"image_id": str(ref["image_id"][i]), "label": "good" if label else "bad", "cell_line": str(ref["cell_line"][i]), "similarity": round(float(sim[i]), 3)})
        return out

    def assess(self, source: Source, error_target: float | str | None = None, with_neighbours: bool = True) -> Assessment:
        image = open_frame(source)
        acq = acquisition_descriptors(to_grey(image))
        flags = acquisition_flags(acq, self.limits)
        cells = self.encoder.cell_descriptors(to_model_input(image, self.encoder.frame_size)[None])[0]
        p, evidence = self.score_cells(cells)
        t_pass = self.pass_threshold(error_target)
        target = float(error_target if error_target is not None else self.default_target)
        if flags:
            decision, reasons = REACQUIRE, flags
        elif p >= t_pass:
            decision, reasons = PASS, [f"P(good) {p:.2f} is at or above the pass threshold {t_pass:.2f} (target: at most {target:.0%} of passed frames bad)"]
        else:
            hint = "likely bad: review first" if p < 0.5 else "uncertain"
            decision, reasons = REVIEW, [f"P(good) {p:.2f} is below the pass threshold {t_pass:.2f} ({hint})"]
        near = self.neighbours(cells.mean(axis=(0, 1))) if with_neighbours else []
        return Assessment(decision, p, reasons, acq, evidence, near)
