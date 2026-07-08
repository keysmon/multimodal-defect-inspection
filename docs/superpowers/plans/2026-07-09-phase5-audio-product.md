# Phase 5.3: Audio Product Integration - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Equipment audio becomes a full third mode: upload a clip -> anomaly score + severity band + CLAP-retrieved HVAC guidance cards, fused into the combined inspection report alongside photo findings.

**Architecture:** A prebuilt normal-sound bank (CLAP embeddings of all DCASE train normals + calibration percentiles) ships as a model artifact; serving scores an uploaded clip against it (kNN, Phase 5.2 scorer) and maps the score to a severity band by calibrated percentile. Guidance retrieval runs in CLAP space against a NEW `audio_card_vectors` table (512-dim - CLIP's table is 768-dim, so a separate table, not a new `kind`). ~40 new HVAC-maintenance cards carry audio fault-family tags. Late fusion: `/analyze` gains an optional audio upload; combined severity = worst-of with the escalation rule; the description model narrates both signals.

**Tech Stack:** existing stack + CLAP (already in). No new deps.

**Branch:** `feat/phase5-audio-product` off `main`. Spec: decisions 5-6. Money: $0.

**Established-pattern shorthand:** where a step says "per the note-field pattern", mirror the exact structure used in Phase 5.1 commits (form field -> canonical normalize -> stub-spy tests -> UI controlled input); where it says "per corpus conventions", match `corpus/engineering_guides.yaml` (own-words passages, real citations, severity from `("structural","urgent","monitor","cosmetic")`).

**Metric adaptation (spec deviation, agreed):** MIMII labels normal/abnormal only - fault families are unlabeled, so the spec's "correct fault-family retrieval" is unmeasurable as written. Measured instead: (a) machine-type retrieval accuracy - a fan clip's top-5 cards must be fan-family, not pump-family; (b) anomaly-score AUC already measured in 5.2. Fault-family retrieval quality is qualitative (demo evidence), stated as such in the README.

---

### Task 1: HVAC-maintenance corpus cards (author + validate)

**Files:**
- Create: `corpus/hvac_maintenance.yaml` (~40 cards)
- Test: existing corpus tests pick the file up automatically (`load_corpus_dir`); add one count/coverage test to `tests/test_corpus.py`

Card taxonomy - `class_tags` use NEW audio fault-family tags (they do not collide with the 9 visual classes): `fan_imbalance`, `bearing_wear`, `belt_drive`, `airflow_restriction`, `motor_electrical`, `pump_cavitation`, `pump_seal_leak`, `compressor_knock`, `mounting_vibration`, `normal_operation`. Coverage: >=4 cards per tag, 40 total. Sources to cite (own-words passages, cite by reference): ASHRAE Handbook fundamentals chapters on sound/vibration, manufacturer service literature summarized generically ("industry service practice"), DOE/EERE pump & fan system sourcebooks (public), InterNACHI HVAC inspection articles. Severity mapping guidance: bearing_wear/compressor_knock -> urgent; motor_electrical -> urgent; fan_imbalance/mounting_vibration/airflow_restriction -> monitor; pump_cavitation -> urgent; pump_seal_leak -> urgent; belt_drive -> monitor; normal_operation -> cosmetic (the "sounds normal" card family explains benign sounds).

- [ ] Step 1: author the YAML per corpus conventions (index_sentence = one audible-symptom sentence, e.g. "rhythmic metallic knocking from a pump that worsens under load"). The index_sentence doubles as the CLAP text-embedding source - write it as a SOUND description, not a visual one.
- [ ] Step 2: add to `tests/test_corpus.py`:

```python
def test_hvac_maintenance_cards_coverage():
    from collections import Counter

    from defectlens.corpus import load_corpus_dir

    cards = [c for c in load_corpus_dir(Path("corpus")) if c.id.startswith("hvac-")]
    assert len(cards) >= 40
    tags = Counter(t for c in cards for t in c.class_tags)
    for tag in ("fan_imbalance", "bearing_wear", "belt_drive", "airflow_restriction",
                "motor_electrical", "pump_cavitation", "pump_seal_leak",
                "compressor_knock", "mounting_vibration", "normal_operation"):
        assert tags[tag] >= 4, tag
```

- [ ] Step 3: `pytest -q` green (corpus loader validates severity enum + required keys itself); commit `feat: 40 HVAC-maintenance guidance cards (audio fault families)`.

USER GATE: after commit, the controller shows the user 5 sample cards for domain sanity-check (spec decision 5); revisions land as a follow-up commit.

---

### Task 2: audio card vectors table + indexer

**Files:**
- Create: `src/defectlens/rag/audio_db.py` (512-dim table, mirrors `rag/db.py` shapes)
- Create: `src/defectlens/rag/audio_embed_cards.py` (`python -m` entry: CLAP-text-embed all hvac-* index_sentences -> upsert)
- Test: `tests/test_rag_audio_db.py` (mirror `tests/test_rag_db.py`'s dedicated-test-database pattern INCLUDING the skip-if-db-down guard)

`audio_card_vectors`: `card_id TEXT, class_tags TEXT[], embedding vector(512), PRIMARY KEY (card_id)` - single vector per card (CLAP text), so no `kind` column. `top_k(conn, embedding, k)` returns `(card_id, class_tags, dist)` ordered by cosine distance, mirroring `db.top_k`'s return shape so `hits_from_rows` reuses cleanly. CLAP text embedding via `model.get_text_features` (unwrap `.pooler_output` if not a tensor - same v5 pattern as embed.py), L2-normalized.

---

### Task 3: normal-bank artifact + calibration

**Files:**
- Create: `scripts/build_audio_bank.py`
- Output artifacts (gitignored, S3 canonical like the adapter): `models/audio_bank/bank.npz` (all fan+pump train-normal CLAP embeddings, ~7k x 512 ≈ 14MB), `models/audio_bank/calibration.json`

Calibration: score every TEST clip (both machines) against the bank with `KNNAnomalyScorer(k=5)`; store `{"normal_score_percentiles": {"p50": ..., "p90": ..., "p99": ...}}` computed over test NORMALS only. Severity banding at serve time: score < p90 -> "normal_operation" band/cosmetic; p90-p99 -> monitor ("atypical - re-listen / re-record"); > p99 -> urgent ("clearly outside normal envelope"). Print the chosen thresholds. Sync artifacts to BOTH S3 buckets under `phase5/audio_bank/` (aws s3 cp, --profile defectlens).

---

### Task 4: serving AudioAnalyzer + /analyze audio upload

**Files:**
- Create: `src/defectlens/serve/audio_analyzer.py`
- Modify: `src/defectlens/serve/api.py` (optional `audio: UploadFile | None = File(None)` param; response gains `"audio": {...} | None`; combined severity)
- Test: `tests/test_audio_analyzer.py` (banding logic pure-TDD), `tests/test_api.py` (stub-spy per the note-field pattern)

`AudioAnalyzer.load()`: env gate `DEFECTLENS_NO_AUDIO=1` -> disabled (mirror `vlm_disabled()`); loads CLAP + bank.npz + calibration.json once. `analyze(wav_bytes) -> AudioFinding(score, band, severity, hits)`: tmp-file or BytesIO -> `load_wav_48k` -> embed -> kNN score -> percentile band -> `audio_db.top_k` retrieval (5 cards) with label fallback: if retrieval returns nothing (empty table), look up cards by the band's fault-family default tag via `card_lookup`.

Combined severity in `/analyze` (late fusion, spec decision 6): `SEVERITY_RANK = {"cosmetic": 0, "monitor": 1, "urgent": 2, "structural": 3}`; final = max(visual, audio) by rank, PLUS escalation: visual >= monitor AND audio band == urgent -> bump one rank (cap at structural). The response includes per-modality findings and the combined value; description prompt gains one sentence naming the audio band when audio present (adapter OFF as established).

---

### Task 5: UI third panel

**Files:** `frontend/src/DefectLens.js`, `DefectLens.css`, `DefectLens.test.js`

Per the note-field pattern: audio file input (accept=".wav,audio/wav", optional), appended to the same FormData under `audio`; results section gains an "Equipment audio" panel (score, band chip color-matched to severity styles, retrieved cards reuse `CardList`); combined severity banner labeled "Combined severity" when both modalities present; report export gains the audio section. One js test: FormData contains the audio file when selected.

---

### Task 6: machine-type retrieval eval

**Files:**
- Create: `src/defectlens/eval/audio_retrieval.py`
- Output: `results/audio_retrieval.json`

For each MIMII TEST clip (sample 50 per machine, seed 42): embed, retrieve top-5 hvac-* cards, machine-type accuracy = fraction where >=3 of 5 cards carry a tag from that machine's family set (fan family: fan_imbalance, belt_drive, airflow_restriction, bearing_wear, motor_electrical, mounting_vibration, normal_operation; pump family: pump_cavitation, pump_seal_leak, bearing_wear, motor_electrical, mounting_vibration, normal_operation - overlap tags count for both). Report per machine + overall; measure-and-report (no gate).

---

### Task 7: E2E + README + merge (controller)

Playwright drive: photo + note + a MIMII anomaly wav through the UI; verify combined severity + audio panel + cards; screenshot. README: "Audio in the product (Phase 5.3)" section - banding thresholds, retrieval eval numbers, MIMII attribution reminder, honest statement that fault-family retrieval is qualitative. Full suites; merge to main.

## Self-review

Spec decision 5 (Level 2: score+severity+cards in UI, ~40 cards, CLAP retrieval with label fallback) - Tasks 1,2,4,5. Decision 6 (late fusion, worst-of + escalation, visible combined assessment, narrative names both signals) - Task 4,5. Metric row adapted (stated at top). Placeholders: none - Task 1 authoring has acceptance rules + user gate; artifact paths exact. Type consistency: AudioFinding consumed by api; audio_db.top_k mirrors db.top_k row shape for hits_from_rows reuse.
