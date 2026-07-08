# Phase 5.1: Photo + Inspector-Note Joint Input - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An optional free-text inspector note conditions BOTH the fine-tuned VLM classification and the RAG card retrieval, with a three-condition sensitivity study proving the empty-note hard gate and measuring note influence.

**Architecture:** The note threads through the ONE prompt source (`qlora.build_messages`) so train/eval/serve stay identical; absent note = byte-identical training prompt (hard gate). RAG conditioning adds the note's CLIP text-embedding card ranking as a third list into the existing `rrf_fuse`. Eval is a sensitivity study (empty / informative / misleading notes) on the crack-vs-no_defect ambiguous subset, with hand-authored notes and an explicit authorship caveat.

**Tech Stack:** existing repo stack - PyTorch/transformers/peft (MPS), FastAPI, React (CRA), pytest.

**Branch:** `feat/phase5-notes` off `main`. Spec: `docs/superpowers/specs/2026-07-08-phase5-multimodal-aws-design.md` (decisions 6, and the Photo+note metric row).

**Money:** $0 (all local MPS/CPU).

---

### Task 1: Note-aware prompt in the single prompt source

**Files:**
- Modify: `src/defectlens/train/qlora.py:78-95` (build_messages)
- Test: `tests/test_train.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_train.py`:

```python
def test_build_messages_without_note_is_unchanged():
    """HARD GATE anchor: no note -> byte-identical prompt to training format."""
    msgs = build_messages("img.jpg", "crack")
    assert msgs[0]["content"][1]["text"] == QUESTION
    assert len(msgs[0]["content"]) == 2


def test_build_messages_with_note_prefixes_inspector_note():
    msgs = build_messages("img.jpg", "crack", note="damp smell below bathroom")
    text = msgs[0]["content"][1]["text"]
    assert text.startswith("Inspector note: damp smell below bathroom\n")
    assert text.endswith(QUESTION)
    assert msgs[1]["content"] == "crack"


def test_build_messages_blank_note_treated_as_absent():
    for blank in (None, "", "   "):
        msgs = build_messages("img.jpg", "crack", note=blank)
        assert msgs[0]["content"][1]["text"] == QUESTION
```

Also extend the existing import line in `tests/test_train.py` to include `QUESTION` if not already imported.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_train.py -q -k note`
Expected: FAIL with `TypeError: build_messages() got an unexpected keyword argument 'note'`

- [ ] **Step 3: Implement**

Replace `build_messages` in `src/defectlens/train/qlora.py`:

```python
def build_messages(image_path: str, label: str, note: str | None = None) -> list[dict]:
    """Qwen chat-format messages for one (image, label) training example.

    `image_path` is embedded as-is into the "image" content field: pass a
    path string (as these tests do) or an already-opened PIL.Image (as the
    training Dataset does) — the processor accepts either.

    `note` (optional inspector free-text) is prefixed before the question.
    A None/blank note produces the EXACT training-time prompt — the serve
    layer's empty-note hard gate (spec Phase 5) rests on this equality, so
    never restructure the no-note branch without retraining.
    """
    question = QUESTION
    if note and note.strip():
        question = f"Inspector note: {note.strip()}\n{QUESTION}"
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        },
        {"role": "assistant", "content": HUMANIZED[label]},
    ]
```

- [ ] **Step 4: Run the full suite**

Run: `pytest -q`
Expected: all pass (137 + 3 new)

- [ ] **Step 5: Commit**

```bash
git add src/defectlens/train/qlora.py tests/test_train.py
git commit -m "feat: optional inspector note in build_messages (empty note = training prompt)"
```

---

### Task 2: Thread note through scoring and the serving classifier

**Files:**
- Modify: `src/defectlens/eval/vlm_topk.py:99` (score_answers signature + build_messages call)
- Modify: `src/defectlens/serve/describer.py` (rank_classes signature)
- Test: `tests/test_vlm_topk.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_vlm_topk.py` (this file already stubs model/processor for score_answers-adjacent logic; if it does not, test via signature inspection which is what we need locked):

```python
import inspect

from defectlens.eval import vlm_topk
from defectlens.serve.describer import Describer


def test_score_answers_accepts_note_kwarg():
    assert "note" in inspect.signature(vlm_topk.score_answers).parameters


def test_rank_classes_accepts_note_kwarg():
    assert "note" in inspect.signature(Describer.rank_classes).parameters
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vlm_topk.py -q -k note`
Expected: FAIL (AssertionError: 'note' not in parameters)

- [ ] **Step 3: Implement**

In `src/defectlens/eval/vlm_topk.py`, change the `score_answers` signature and the one `build_messages` call:

```python
def score_answers(model, processor, image, device: str, note: str | None = None) -> dict[str, float]:
```

and inside the loop:

```python
        messages = build_messages(image, label, note=note)
```

In `src/defectlens/serve/describer.py`, change `rank_classes`:

```python
    def rank_classes(self, image, note: str | None = None) -> list[tuple[str, float]]:
```

and its `score_answers` call:

```python
        loglik = score_answers(self.model, self.processor, image, self.device, note=note)
```

- [ ] **Step 4: Run the full suite**

Run: `pytest -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/defectlens/eval/vlm_topk.py src/defectlens/serve/describer.py tests/test_vlm_topk.py
git commit -m "feat: thread inspector note through score_answers and rank_classes"
```

---

### Task 3: Note conditions RAG retrieval (third RRF ranking)

**Files:**
- Modify: `src/defectlens/serve/recognizer.py:142-166` (analyze_image_bytes)
- Test: `tests/test_recognizer.py` (append; follow the file's existing stub pattern for model/conn)

`rrf_fuse(rankings: list[list[str]])` already accepts N rankings. When a note is present, embed it with the recognizer's already-loaded CLIP text encoder (same path `TextSearcher`/`query_by_text` uses: `embed_texts`) and rank all cards by text-vector distance; append that ranking to the fusion inputs inside `fused_card_ranking`'s caller.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_recognizer.py`, following that file's existing fixtures (it stubs `db.top_k` and the CLIP model; read the top of the file first and reuse its helpers — the assertion below is the contract):

```python
def test_note_ranking_added_to_fusion(monkeypatch, stub_recognizer_env):
    """With a note, analyze_image_bytes fuses THREE rankings (centroid,
    zero-shot prompt, note-text); without, the original two."""
    captured = []

    def spy_rrf(rankings, k=60):
        captured.append(len(rankings))
        return [item for ranking in rankings for item in ranking]

    monkeypatch.setattr("defectlens.serve.recognizer.rrf_fuse", spy_rrf)

    stub_recognizer_env.analyze_image_bytes(PNG_BYTES, k=2)
    stub_recognizer_env.analyze_image_bytes(PNG_BYTES, k=2, note="musty smell")
    assert captured == [2, 3]
```

(If `tests/test_recognizer.py` has no reusable stub env fixture, build the minimal one in this test module: stub model/processor returning fixed embeddings, `FakeConn` for `db.top_k` — copy the pattern already used in that file's existing tests.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_recognizer.py -q -k note`
Expected: FAIL (`analyze_image_bytes() got an unexpected keyword argument 'note'` or captured == [2, 2])

- [ ] **Step 3: Implement**

In `src/defectlens/serve/recognizer.py`:

1. `fused_card_ranking` currently builds the two-ranking fusion. Give it an optional third input:

```python
def fused_card_ranking(
    centroid_ranked_ids: list[str],
    prompt_class_sims: dict[str, float],
    lookup: dict[str, "Card"],
    note_ranked_ids: list[str] | None = None,
) -> list[str]:
```

and where it calls `rrf_fuse([centroid_ranking, prompt_ranking])`, build the input list dynamically:

```python
    rankings = [centroid_ranked_ids, prompt_ranking]
    if note_ranked_ids:
        rankings.append(note_ranked_ids)
    return rrf_fuse(rankings)
```

(Adapt names to the function's actual internals — the second ranking is derived from `prompt_class_sims` inside the function; keep that derivation untouched.)

2. `analyze_image_bytes(self, data: bytes, k: int = 5, note: str | None = None)`: when `note` is non-blank, compute the note ranking before the fusion call:

```python
        note_ranked_ids = None
        if note and note.strip():
            note_emb = normalize(
                embed_texts(self.model, self.processor, [note.strip()], self.device)
            )[0]
            note_rows = db.top_k(self.conn, note_emb, len(self.cards), ("text",))
            note_ranked_ids = [cid for cid, _tags, _dist in note_rows]
```

and pass `note_ranked_ids=note_ranked_ids` into `fused_card_ranking`.

- [ ] **Step 4: Run the full suite**

Run: `pytest -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/defectlens/serve/recognizer.py tests/test_recognizer.py
git commit -m "feat: inspector note as third RRF ranking in card retrieval"
```

---

### Task 4: API accepts the note

**Files:**
- Modify: `src/defectlens/serve/api.py` (/analyze endpoint)
- Test: `tests/test_api.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_api.py` (reuse `StubRecognizer`, `StubDescriber`, `make_png_bytes`, `_analyze_result` already defined there):

```python
def test_analyze_forwards_note_to_recognizer_describer_and_response():
    result = _analyze_result()

    class NoteSpyRecognizer(StubRecognizer):
        def analyze_image_bytes(self, data, k, note=None):
            self.note = note
            return self.result

    class NoteSpyDescriber(StubDescriber):
        def rank_classes(self, img, note=None):
            self.note = note
            return [("water_damage", 0.9)]

    recognizer = NoteSpyRecognizer(result)
    describer = NoteSpyDescriber()
    app = create_app(recognizer=recognizer, describer=describer)
    client = TestClient(app)

    resp = client.post(
        "/analyze",
        files={"file": ("t.png", make_png_bytes(), "image/png")},
        data={"note": "musty smell near shower"},
    )
    assert resp.status_code == 200
    assert recognizer.note == "musty smell near shower"
    assert describer.note == "musty smell near shower"
    assert resp.json()["note"] == "musty smell near shower"


def test_analyze_without_note_passes_none():
    result = _analyze_result()

    class NoteSpyRecognizer(StubRecognizer):
        def analyze_image_bytes(self, data, k, note=None):
            self.note = note
            return self.result

    recognizer = NoteSpyRecognizer(result)
    app = create_app(recognizer=recognizer, describer=StubDescriber())
    client = TestClient(app)
    resp = client.post("/analyze", files={"file": ("t.png", make_png_bytes(), "image/png")})
    assert resp.status_code == 200
    assert recognizer.note is None
```

Note: `StubRecognizer.analyze_image_bytes` asserts `k == self.expected_k`; the spy subclasses above bypass that assert by overriding, which is fine.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api.py -q -k note`
Expected: FAIL (unexpected keyword / missing "note" key)

- [ ] **Step 3: Implement**

In `src/defectlens/serve/api.py`:

1. Import `Form`: extend the existing `from fastapi import ...` line with `Form`.
2. `/analyze` signature:

```python
    async def analyze(
        request: Request,
        file: UploadFile = File(...),
        note: str = Form(""),
    ) -> dict:
```

3. Normalize once near the top of the handler:

```python
        note_text = note.strip() or None
```

4. Forward it (existing lines change):

```python
        result = recognizer.analyze_image_bytes(data, k=5, note=note_text)
        ...
        vlm_classes = getattr(describer, "rank_classes", lambda _img, note=None: [])(
            img, note=note_text
        )
```

5. Also pass the note into the description prompt context by appending it to `top_labels` handling — change the describe call to:

```python
        description = describer.describe(img, top_labels)
```

(unchanged - the note already influenced the classes; keep describe's contract stable this task)

6. Add `"note": note_text` to the response dict.

Backward-compat: `StubRecognizer.analyze_image_bytes(self, data, k)` in OTHER existing tests has no `note` param — update that stub's signature in `tests/test_api.py` to `def analyze_image_bytes(self, data, k, note=None):` so all existing tests keep passing.

- [ ] **Step 4: Run the full suite**

Run: `pytest -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/defectlens/serve/api.py tests/test_api.py
git commit -m "feat: /analyze accepts optional inspector note form field"
```

---

### Task 5: Frontend note field

**Files:**
- Modify: `frontend/src/DefectLens.js` (upload section + FormData + report export)
- Modify: `frontend/src/DefectLens.css`
- Test: `frontend/src/DefectLens.test.js` (append)

- [ ] **Step 1: Write the failing test**

Append to `frontend/src/DefectLens.test.js` (reuse its existing render/axios-mock pattern — read the file's existing analyze test first and mirror its setup):

```javascript
test("note textarea renders and its value is sent with analyze", async () => {
  // mirror the existing successful-analyze test setup (axios mock), then:
  const noteBox = screen.getByPlaceholderText(/optional inspector note/i);
  fireEvent.change(noteBox, { target: { value: "musty smell near shower" } });
  // ...trigger analyze as the existing test does, then assert on the mock:
  const formData = axios.post.mock.calls[0][1];
  expect(formData.get("note")).toBe("musty smell near shower");
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd frontend && CI=true npm test -- --watchAll=false`
Expected: FAIL (no element with that placeholder)

- [ ] **Step 3: Implement**

In `frontend/src/DefectLens.js`:

1. State: `const [note, setNote] = useState("");`
2. UI, directly under the file input inside the upload section:

```jsx
<textarea
  className="note-input"
  placeholder="Optional inspector note (e.g., 'musty smell, below upstairs bathroom')"
  value={note}
  onChange={(e) => setNote(e.target.value)}
  rows={2}
  maxLength={500}
/>
```

3. Where the analyze FormData is built, add:

```javascript
if (note.trim()) formData.append("note", note.trim());
```

4. In the markdown report builder, after the severity line:

```javascript
if (analyzeResult.note) lines.push(`- Inspector note: ${analyzeResult.note}`);
```

In `frontend/src/DefectLens.css` append:

```css
.note-input {
  display: block;
  width: 100%;
  max-width: 440px;
  margin: 10px auto 0;
  padding: 8px 10px;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  font: inherit;
  font-size: 14px;
  resize: vertical;
}
```

- [ ] **Step 4: Run tests**

Run: `cd frontend && CI=true npm test -- --watchAll=false`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/DefectLens.js frontend/src/DefectLens.css frontend/src/DefectLens.test.js
git commit -m "feat(ui): optional inspector note input"
```

---

### Task 6: Note-sensitivity study (the eval)

**Files:**
- Create: `src/defectlens/eval/note_sensitivity.py`
- Create: `data/notes/ambiguous_subset_notes.json` (hand-authored during this task)
- Test: `tests/test_note_sensitivity.py`

**Design (from spec + grilling):** three conditions on a hand-picked ambiguous subset (crack + no_defect test images, n=40: pick the first 20 of each class from `data/manifests/test.csv` whose filenames sort deterministically). Conditions: `empty` (note=None), `informative` (hand-authored context note per image), `misleading` (a fixed off-topic note: "kitchen area, recently repainted, no issues reported"). Outputs per condition: macro top-1 on the subset + per-image predictions. HARD GATE asserted in code: `empty` condition prompt equals the training prompt (already locked by Task 1's unit test) and empty-vs-baseline accuracy identical (same seed, same images — deterministic scoring makes this exact).

- [ ] **Step 1: Write the failing tests (pure parts)**

Create `tests/test_note_sensitivity.py`:

```python
from defectlens.eval.note_sensitivity import build_conditions, select_ambiguous_rows


def _row(path, label):
    return {"image_path": path, "unified_label": label}


def test_select_ambiguous_rows_balanced_and_deterministic():
    rows = [_row(f"a/crack_{i}.jpg", "crack") for i in range(30)]
    rows += [_row(f"a/plain_{i}.jpg", "no_defect") for i in range(30)]
    rows += [_row("a/mold.jpg", "mold_algae")]
    picked = select_ambiguous_rows(rows, per_class=20)
    assert len(picked) == 40
    labels = [r["unified_label"] for r in picked]
    assert labels.count("crack") == 20 and labels.count("no_defect") == 20
    assert picked == select_ambiguous_rows(rows, per_class=20)  # deterministic


def test_build_conditions_shapes():
    notes = {"a/crack_0.jpg": "hairline line on garage slab"}
    conds = build_conditions(notes, misleading="kitchen, repainted")
    assert conds["empty"]("a/crack_0.jpg") is None
    assert conds["informative"]("a/crack_0.jpg") == "hairline line on garage slab"
    assert conds["informative"]("a/unknown.jpg") is None  # missing note -> skip as empty
    assert conds["misleading"]("a/crack_0.jpg") == "kitchen, repainted"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_note_sensitivity.py -q`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement the module**

Create `src/defectlens/eval/note_sensitivity.py`:

```python
"""Note-sensitivity study (spec Phase 5, photo+note workstream).

Three conditions over an ambiguous crack/no_defect subset:
  empty        - note=None; MUST reproduce the no-note baseline exactly
                 (build_messages equality is unit-locked; scoring is
                 deterministic, so accuracy must match to the last image)
  informative  - hand-authored context notes (data/notes/*.json). CAVEAT
                 (reported in output): notes were authored by the project
                 authors while viewing the images; this measures prompt
                 sensitivity, not field accuracy gain.
  misleading   - one fixed off-topic note for every image (robustness:
                 accuracy should not collapse under irrelevant text)

Usage:
  python -m defectlens.eval.note_sensitivity --notes data/notes/ambiguous_subset_notes.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

MISLEADING_NOTE = "kitchen area, recently repainted, no issues reported"


def select_ambiguous_rows(rows: list[dict], per_class: int = 20) -> list[dict]:
    """First `per_class` crack + no_defect rows by sorted image_path (deterministic)."""
    picked: list[dict] = []
    for cls in ("crack", "no_defect"):
        cls_rows = sorted(
            (r for r in rows if r["unified_label"] == cls),
            key=lambda r: r["image_path"],
        )
        picked.extend(cls_rows[:per_class])
    return picked


def build_conditions(notes: dict[str, str], misleading: str = MISLEADING_NOTE):
    """Map condition name -> (image_path -> note-or-None)."""
    return {
        "empty": lambda path: None,
        "informative": lambda path: notes.get(path),
        "misleading": lambda path: misleading,
    }


def main(argv: list[str] | None = None) -> None:
    from PIL import Image
    from tqdm import tqdm

    from defectlens.eval.vlm_topk import (
        _load_model_and_processor,
        pick_device,
        rank_answers,
        score_answers,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", type=Path, default=Path("data/manifests/test.csv"))
    parser.add_argument("--notes", type=Path, default=Path("data/notes/ambiguous_subset_notes.json"))
    parser.add_argument("--adapter", type=Path, default=Path("models/qwen25vl-lora-v1"))
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--max-pixels", type=int, default=589824)
    parser.add_argument("--out", type=Path, default=Path("results/note_sensitivity.json"))
    args = parser.parse_args(argv)

    rows = list(csv.DictReader(open(args.test_manifest)))
    subset = select_ambiguous_rows(rows, per_class=args.per_class)
    notes = json.loads(args.notes.read_text(encoding="utf-8"))
    conditions = build_conditions(notes)

    device = pick_device()
    model, processor = _load_model_and_processor(args.adapter, device, args.max_pixels)

    results: dict = {"n": len(subset), "authorship_caveat": (
        "informative notes were hand-authored by the project authors while "
        "viewing the images; this measures prompt sensitivity, not field accuracy gain"
    ), "conditions": {}}
    for name, note_fn in conditions.items():
        correct = 0
        preds = []
        for row in tqdm(subset, desc=name):
            img = Image.open(row["image_path"]).convert("RGB")
            loglik = score_answers(model, processor, img, device, note=note_fn(row["image_path"]))
            top = rank_answers(loglik)[0]
            preds.append({"path": row["image_path"], "true": row["unified_label"], "pred": top})
            correct += top == row["unified_label"]
        results["conditions"][name] = {"accuracy": correct / len(subset), "preds": preds}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    for name, c in results["conditions"].items():
        print(f"{name:>12}: accuracy {c['accuracy']:.3f}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit tests**

Run: `pytest tests/test_note_sensitivity.py -q`
Expected: PASS

- [ ] **Step 5: Author the notes file**

Create `data/notes/ambiguous_subset_notes.json`: run `python -c "..."` to print the 40 selected paths (`select_ambiguous_rows` on the test manifest), then for each image (controller views each with the Read tool) write one realistic 5-15 word context note that describes VISIBLE CONTEXT (surface, location-guess, moisture) WITHOUT naming any class label word (crack/spalling/mold/...). Keys are the manifest `image_path` strings, values the notes. Commit the JSON.

- [ ] **Step 6: Run the study (MPS, ~30-45 min for 120 scorings)**

Run: `python -m defectlens.eval.note_sensitivity`
Expected output shape:

```
       empty: accuracy 0.xxx
 informative: accuracy 0.xxx
  misleading: accuracy 0.xxx
Wrote results/note_sensitivity.json
```

HARD GATE check (manual): `empty` accuracy must equal the same 40 images' accuracy in the no-note pipeline. Verify with: rerun any 3 of the subset through `score_answers` without the note kwarg and diff the logliks (must be identical to float precision).

- [ ] **Step 7: Commit study + results**

```bash
git add src/defectlens/eval/note_sensitivity.py tests/test_note_sensitivity.py data/notes/ results/note_sensitivity.json
git commit -m "feat: note-sensitivity study (empty gate, informative delta, misleading robustness)"
```

---

### Task 7: E2E verification + README + merge

**Files:**
- Modify: `README.md` (results section: one paragraph + numbers from results/note_sensitivity.json)

- [ ] **Step 1: E2E drive (real flow, per workflow rules)**

Start db (`docker compose up -d db`), API (`uvicorn defectlens.serve.api:app --port 8000`), frontend (`cd frontend && BROWSER=none npm start`). Via Playwright MCP: upload a water-damage-adjacent test image TWICE - once with empty note, once with note "musty smell, ceiling below upstairs bathroom". Verify: (a) both return 200 with `classifier: vlm-qlora`; (b) the note case shows the note reflected in the response and (typically) shifts class probabilities; (c) screenshot the note-present result.

- [ ] **Step 2: README update**

Add under the Phase 3 results section a short "Inspector notes (Phase 5)" paragraph: the three-condition table from `results/note_sensitivity.json`, the empty-note hard-gate statement, and the authorship caveat sentence verbatim.

- [ ] **Step 3: Full suites one last time**

Run: `pytest -q && cd frontend && CI=true npm test -- --watchAll=false`
Expected: all pass

- [ ] **Step 4: Commit, merge to main, push**

```bash
git add README.md docs/images/
git commit -m "docs: note-sensitivity results"
git checkout main && git merge --no-ff feat/phase5-notes -m "Merge Phase 5.1: photo + inspector-note joint input"
git push origin main && git checkout feat/phase5-notes
```

---

## Self-review

**Spec coverage:** decision 6's image+note joint path (Tasks 1-2), note->RAG conditioning (Task 3), API+UI (Tasks 4-5), the metric row incl. hard gate + measure-and-report (Task 6), seam discipline (Task 7 merge). Audio/OOD/deploy are later plans by design.
**Placeholders:** none - every step has code or an exact command; Task 6 Step 5 is a content-authoring step with explicit acceptance rules (no label words, 5-15 words, visible context only).
**Type consistency:** `note: str | None = None` kwarg end-to-end; API normalizes `Form("")` -> `note_text: str | None`; `build_conditions` returns path->note-or-None callables consumed by `score_answers(note=...)`.
