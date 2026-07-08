"""Per-machine-ID audio anomaly AUC on the DCASE 2020 T2 dev set.

Protocol (matches the challenge): per machine type + ID, fit the scorer on
that ID's TRAIN normals only, score its TEST clips, AUC against the
normal/anomaly labels. Compares against the published AE baseline
(Koizumi et al. 2020, arXiv:2006.05822).

Usage: python -m defectlens.eval.audio_auc            # both machines
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# Official baseline dev-set AUCs, from the DCASE2020 Task 2 baseline repo
# results table (github.com/y-kawagu/dcase2020_task2_baseline; the paper
# arXiv:2006.05822 reports the same system). Fetched 2026-07-08.
BASELINE_AUC: dict[str, dict[str, float]] = {
    "fan": {"00": 0.5396, "02": 0.7219, "04": 0.6221, "06": 0.7228},
    "pump": {"00": 0.6708, "02": 0.6094, "04": 0.8886, "06": 0.7349},
}


def results_table(ours: dict, baseline: dict) -> dict:
    table: dict = {}
    for machine, ids in ours.items():
        table[machine] = {}
        for mid, auc in ids.items():
            base = baseline.get(machine, {}).get(mid)
            table[machine][mid] = {
                "auc": auc,
                "baseline": base,
                "beats_baseline": (base is not None and auc > base),
            }
    return table


def main(argv: list[str] | None = None) -> None:
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from tqdm import tqdm

    from defectlens.audio.anomaly import KNNAnomalyScorer
    from defectlens.audio.dataset import scan_machine_dir
    from defectlens.audio.embed import embed_audio_files, load_clap
    from defectlens.eval.clip_zeroshot import pick_device

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, default=Path("data/raw/audio"))
    parser.add_argument("--machines", nargs="+", default=["fan", "pump"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("results/audio_auc.json"))
    args = parser.parse_args(argv)

    device = pick_device()
    model, processor = load_clap(device)

    ours: dict = {}
    for machine in args.machines:
        rows = scan_machine_dir(args.audio_root / machine, machine=machine)
        by_id = defaultdict(lambda: {"train": [], "test": []})
        for r in rows:
            by_id[r.machine_id][r.split].append(r)
        ours[machine] = {}
        for mid, splits in tqdm(sorted(by_id.items()), desc=machine):
            train_paths = [Path(r.path) for r in splits["train"] if r.label == "normal"]
            test_rows = splits["test"]
            train_embs = embed_audio_files(model, processor, train_paths, device, args.batch_size)
            test_embs = embed_audio_files(
                model, processor, [Path(r.path) for r in test_rows], device, args.batch_size
            )
            scorer = KNNAnomalyScorer(k=args.k).fit(train_embs)
            scores = scorer.score(test_embs)
            y_true = np.array([r.label == "anomaly" for r in test_rows], dtype=int)
            ours[machine][mid] = float(roc_auc_score(y_true, scores))

    table = results_table(ours, BASELINE_AUC)
    payload = {
        "method": "CLAP (laion/clap-htsat-unfused) embeddings + kNN(k=%d) on train normals" % args.k,
        "protocol": "DCASE 2020 Task 2 dev set, per-machine-ID, unsupervised",
        "baseline": "AE baseline, Koizumi et al. 2020 (arXiv:2006.05822)",
        "results": table,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    for machine, ids in table.items():
        aucs = [v["auc"] for v in ids.values()]
        print(f"{machine}: avg AUC {sum(aucs)/len(aucs):.4f}")
        for mid, v in sorted(ids.items()):
            beat = "BEATS" if v["beats_baseline"] else "below"
            print(f"  id_{mid}: {v['auc']:.4f} vs baseline {v['baseline']} ({beat})")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
