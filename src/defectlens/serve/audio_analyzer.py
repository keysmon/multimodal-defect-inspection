"""Equipment-audio serving component: anomaly score -> band + guidance cards (Phase 5.3).

Mirrors Describer's optional-component shape:
- env gate DEFECTLENS_NO_AUDIO=1 disables it (like vlm_disabled());
- a missing/unbuilt bank artifact ALSO disables it (enabled=False), so the API
  still boots and reports "audio": null until models/audio_bank/ is built and
  synced (the controller builds it; see scripts/build_audio_bank.py).

Kept cheap to import — torch/CLAP/numpy load inside load()/analyze() only, so
serve.api stays importable without the ML stack.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from defectlens.rag import audio_db
from defectlens.rag.retrieve import Hit, card_lookup, hits_from_rows

# Calibrated score band -> (band label, report severity). Bands are the ones
# scripts/build_audio_bank.py calibrates percentiles for.
BAND_SEVERITY = {
    "normal_operation": "cosmetic",
    "atypical": "monitor",
    "anomalous": "urgent",
}

SEVERITY_RANK = {"cosmetic": 0, "monitor": 1, "urgent": 2, "structural": 3}
_RANK_SEVERITY = {rank: sev for sev, rank in SEVERITY_RANK.items()}


def audio_disabled() -> bool:
    return os.environ.get("DEFECTLENS_NO_AUDIO", "") == "1"


def band_for_score(score: float, p90: float, p99: float) -> tuple[str, str]:
    """Map an anomaly score to (band, severity) by calibrated percentiles.

    score < p90 -> normal_operation/cosmetic; p90 <= score <= p99 ->
    atypical/monitor; score > p99 -> anomalous/urgent.
    """
    if score < p90:
        band = "normal_operation"
    elif score > p99:
        band = "anomalous"
    else:
        band = "atypical"
    return band, BAND_SEVERITY[band]


def combine_severity(visual: str, audio: str) -> str:
    """Late-fusion severity (spec decision 6): worst-of by rank, then escalate.

    Escalation: when the visual finding is at least 'monitor' AND the audio band
    is urgent (anomalous), bump one rank, capped at 'structural'.
    """
    combined = max(SEVERITY_RANK[visual], SEVERITY_RANK[audio])
    if SEVERITY_RANK[visual] >= SEVERITY_RANK["monitor"] and audio == "urgent":
        combined = min(combined + 1, SEVERITY_RANK["structural"])
    return _RANK_SEVERITY[combined]


@dataclass
class AudioFinding:
    score: float
    band: str
    severity: str
    hits: list[Hit] = field(default_factory=list)


class AudioAnalyzer:
    """Loads CLAP + the normal-sound bank + calibration once; scores clips."""

    def __init__(
        self,
        bank_dir: Path = Path("models/audio_bank"),
        corpus_dir: Path = Path("corpus"),
    ) -> None:
        self.bank_dir = Path(bank_dir)
        self.corpus_dir = Path(corpus_dir)
        self.model = None
        self.processor = None
        self.device: str | None = None
        self.scorer = None
        self.p50: float | None = None
        self.p90: float | None = None
        self.p99: float | None = None
        self.conn = None
        self.lookup: dict = {}
        self.enabled = False

    def load(self) -> None:
        if audio_disabled():
            return

        bank_path = self.bank_dir / "bank.npz"
        calib_path = self.bank_dir / "calibration.json"
        if not (bank_path.is_file() and calib_path.is_file()):
            # Artifact not built/synced yet — stay disabled so the API still
            # boots (reports "audio": null), mirroring Describer's missing-adapter path.
            return

        import numpy as np

        from defectlens.audio.anomaly import KNNAnomalyScorer
        from defectlens.audio.embed import load_clap
        from defectlens.eval.clip_zeroshot import pick_device

        calib = json.loads(calib_path.read_text(encoding="utf-8"))
        pct = calib["normal_score_percentiles"]
        self.p50, self.p90, self.p99 = pct["p50"], pct["p90"], pct["p99"]

        bank = np.load(bank_path)["embeddings"]
        self.scorer = KNNAnomalyScorer(k=int(calib.get("k", 5))).fit(bank)

        self.device = pick_device()
        self.model, self.processor = load_clap(self.device)

        # Card metadata for retrieval joins. DB or corpus trouble degrades
        # retrieval to empty hits (band still drives severity); it must not
        # block audio scoring from being enabled.
        try:
            self.conn = audio_db.connect()
            audio_db.ensure_schema(self.conn)
        except Exception:
            self.conn = None
        try:
            from defectlens.corpus import load_corpus_dir

            self.lookup = card_lookup(load_corpus_dir(self.corpus_dir))
        except Exception:
            self.lookup = {}

        self.enabled = True

    def analyze(self, wav_bytes: bytes) -> AudioFinding:
        if not self.enabled:
            raise RuntimeError("AudioAnalyzer.analyze() called while disabled")

        from defectlens.audio.embed import embed_audio_files

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(wav_bytes)
            tmp.flush()
            emb = embed_audio_files(
                self.model, self.processor, [Path(tmp.name)], self.device
            )

        score = float(self.scorer.score(emb)[0])
        band, severity = band_for_score(score, self.p90, self.p99)

        hits: list[Hit] = []
        if self.conn is not None and self.lookup:
            # top_k([]) on an empty table returns [] -> hits_from_rows -> [].
            rows = audio_db.top_k(self.conn, emb[0], 5)
            try:
                hits = hits_from_rows(rows, self.lookup)
            except KeyError:
                # Audio index references a card no longer in corpus/ (drift):
                # degrade to score-only; re-run defectlens.rag.audio_embed_cards.
                hits = []
        return AudioFinding(score=score, band=band, severity=severity, hits=hits)
