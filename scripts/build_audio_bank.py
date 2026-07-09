"""Build the normal-sound bank + score-calibration artifact for serving.

Fit CLAP embeddings of ALL fan+pump train normals into one bank (bank.npz),
score every test clip with KNNAnomalyScorer, and calibrate severity-band
thresholds from the anomaly-score distribution over test NORMALS only. At serve
time an uploaded clip is embedded, scored against the bank, and mapped to a band
by these percentiles: score < p90 -> normal/cosmetic; p90..p99 -> monitor;
> p99 -> urgent.

The full run embeds ~7k train clips (~30 min) — the controller runs it and syncs
the artifacts to S3. Use --limit N for a fast smoke test: it caps files at N per
machine/split, taking N normals AND N anomalies for test so calibration still has
test-normal scores (test files sort anomaly-before-normal, so a naive head() would
select zero normals).

Usage:
  python scripts/build_audio_bank.py                          # full build
  python scripts/build_audio_bank.py --limit 20 --out-dir /tmp/smoke   # smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def calibration_percentiles(normal_scores) -> dict[str, float]:
    """p50/p90/p99 of anomaly scores over test NORMALS. Pure; empty input errors."""
    scores = np.asarray(normal_scores, dtype=float)
    if scores.size == 0:
        raise ValueError("no test-normal scores to calibrate from (empty input)")
    return {
        "p50": float(np.percentile(scores, 50)),
        "p90": float(np.percentile(scores, 90)),
        "p99": float(np.percentile(scores, 99)),
    }


def select_train_normals(scan, audio_root: Path, machines: list[str], limit: int | None) -> list[Path]:
    """Train-split NORMAL clips across machines; --limit caps per machine."""
    paths: list[Path] = []
    for machine in machines:
        rows = scan(audio_root / machine, machine=machine)
        normals = [Path(r.path) for r in rows if r.split == "train" and r.label == "normal"]
        if limit is not None:
            normals = normals[:limit]
        paths.extend(normals)
    return paths


def select_test_clips(
    scan, audio_root: Path, machines: list[str], limit: int | None
) -> tuple[list[Path], list[str]]:
    """Test-split clips + labels; --limit caps N normals AND N anomalies per machine."""
    paths: list[Path] = []
    labels: list[str] = []
    for machine in machines:
        rows = [r for r in scan(audio_root / machine, machine=machine) if r.split == "test"]
        if limit is not None:
            capped = {"normal": [], "anomaly": []}
            for r in rows:
                if r.label in capped and len(capped[r.label]) < limit:
                    capped[r.label].append(r)
            rows = capped["normal"] + capped["anomaly"]
        for r in rows:
            paths.append(Path(r.path))
            labels.append(r.label)
    return paths, labels


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--audio-root", type=Path, default=Path("data/raw/audio"))
    parser.add_argument("--machines", nargs="+", default=["fan", "pump"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="smoke test: cap files per machine/split (N normals AND N anomalies)",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("models/audio_bank"))
    args = parser.parse_args(argv)

    from defectlens.audio.anomaly import KNNAnomalyScorer
    from defectlens.audio.dataset import scan_machine_dir
    from defectlens.audio.embed import embed_audio_files, load_clap
    from defectlens.eval.clip_zeroshot import pick_device

    train_paths = select_train_normals(scan_machine_dir, args.audio_root, args.machines, args.limit)
    test_paths, test_labels = select_test_clips(
        scan_machine_dir, args.audio_root, args.machines, args.limit
    )
    if not train_paths:
        raise SystemExit(f"no train normals under {args.audio_root} — check --audio-root/--machines")
    if not test_paths:
        raise SystemExit(f"no test clips under {args.audio_root} — check --audio-root/--machines")
    print(f"train normals: {len(train_paths)}; test clips: {len(test_paths)}")

    device = pick_device()
    model, processor = load_clap(device)

    bank = embed_audio_files(model, processor, train_paths, device, args.batch_size)
    test_embs = embed_audio_files(model, processor, test_paths, device, args.batch_size)

    scorer = KNNAnomalyScorer(k=args.k).fit(bank)
    scores = scorer.score(test_embs)
    labels = np.array(test_labels)
    calib = calibration_percentiles(scores[labels == "normal"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bank_path = args.out_dir / "bank.npz"
    calib_path = args.out_dir / "calibration.json"
    # Array key "embeddings" is the loader contract for the serving AudioAnalyzer.
    np.savez_compressed(bank_path, embeddings=bank)
    payload = {
        "normal_score_percentiles": calib,
        "k": args.k,
        "machines": args.machines,
        "n_train_normals": int(len(train_paths)),
        "n_test_normals": int((labels == "normal").sum()),
        "limit": args.limit,
    }
    calib_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {bank_path} ({bank.shape[0]} x {bank.shape[1]})")
    print(f"Wrote {calib_path}")
    print("Severity band thresholds (anomaly score over test normals):")
    print(f"  normal_operation/cosmetic: score < p90 = {calib['p90']:.4f}")
    print(f"  monitor:                   p90..p99 = [{calib['p90']:.4f}, {calib['p99']:.4f}]")
    print(f"  urgent:                    score > p99 = {calib['p99']:.4f}")
    print(f"  (p50 reference = {calib['p50']:.4f})")


if __name__ == "__main__":
    main()
