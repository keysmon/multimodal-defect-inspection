# Multimodal Building-Defect Inspection

An ML-powered building-defect inspection assistant, built end to end: data
pipeline, fine-tuning, retrieval, serving, web UI, and AWS infrastructure.
Upload a defect photo for ranked defect classes, a severity band, and cited
remediation guidance from an inspection-standards corpus - or run the
flagship **walkthrough diagnostic report**: up to 10 site photos plus a
free-text concern note become a grounded, cited draft diagnostic in which
every claim either cites a retrieved standards card or is replaced by an
explicit "not observed - verify on-site".

**Live demo:** https://d2wxjiu5re5mow.cloudfront.net

| Fine-tuned classifier | Guidance retrieval | Audio anomaly (pump) | Walkthrough groundedness |
| :---: | :---: | :---: | :---: |
| **0.903** macro top-1 (12 classes) | **0.863** recall@5 | **0.801** AUC (0.726 baseline) | **1.0** measured pre-gate |

## Walkthrough diagnostic report


One multimodal call sees every photo at once, so the report reasons across
them ("the staining at the crack in photo 4 means the photo-1 crack is an
active moisture pathway"). The trust story is deterministic, not
prompt-deep: a citation gate drops every uncited claim into an audit log,
per-photo claims may cite only that photo's own retrieval, every concern
gets a cited answer or an explicit "not observed", and an optional
GPU-backed fine-tuned label merges only when consistent with the
observation. Visual accuracy is hand-rated rather than faked with an LLM
judge - and the split it found is the honest headline:

| Spot-check | accurate observations |
|---|---|
| 256px dataset crops (hard) | 17/30 strict |
| realistic field photos | 8/8 primary |

Full methodology, mechanism, and results: **[docs/case-study.md](docs/case-study.md)**

## Taxonomy v2 - multi-material coverage (second fine-tune cycle)

The classifier grew from a 9-class concrete-centric taxonomy to 12 classes
spanning masonry, brick, timber, steel, and grid electrical insulators
(`finish_detachment`, `bulge_deformation`, `insulator_damage`), trained on
three added licensed datasets (MBDD2025 CC BY, VT Corrosion Condition State
CC0, insulator defect detection CC BY). The v1 frozen test split is archived
byte-identical and an invariant test proves every v1 test row survives in
the v2 split, so backward compatibility is a measured number, not a hope:

| Metric | Result |
|---|---|
| v2 frozen split (4,824 imgs, 12 classes) | **0.903** macro top-1 / 0.992 top-3 |
| v1 frozen split backward-compat (2,648 imgs) | **0.841** vs the 0.831 floor (v1 adapter: 0.851) |
| New classes | insulator 0.978 / bulge 0.974 / finish_detachment 0.941 top-1 |
| VT corrosion severity states | corrosion recognized on fair 0.93 / poor 1.00 / severe 1.00 |
| Cross-dataset OOD (METU crack, 400 imgs) | **0.900** macro top-1 (v1 adapter: 0.877) |

The walkthrough enrichment gate's confidence floor is now evidence-derived
(0.375, max kept-correct at <= 5% merged-incorrect over 4,824 per-image
confidences; the full curve ships in `results/gate_floor_v2.json`) - it
previously dropped correct labels observed live at 0.436 under the old 0.5
floor.

**Verified negative findings, stated plainly:** no license-clean
HVAC-equipment or residential electrical-panel visual-defect dataset exists
(corrosion severity states stand in as the equipment-corrosion proxy), and
`insulator_damage` covers grid transmission insulators, not panels.

## Documented-case exemplars (image-grounded retrieval)

Guidance cards and analyze results now show licensed exemplar photographs:
70 images (public domain / CC0 / CC BY only - ShareAlike and NC excluded by
contract, enforced by tests) curated from the training datasets, FEMA/NPS
photography, and Wikimedia Commons with a recorded per-image license check.
`/analyze` returns "similar documented cases" by CLIP image similarity;
retrieval class-consistency is reported honestly as a pool-limited proxy
(0.436 top-1 overall - strong where the pool is deep, near-zero for
one-exemplar classes; `results/exemplar_retrieval.json`). Attribution:
[docs/exemplar-attribution.md](docs/exemplar-attribution.md).

## Stack

- **ML** - PyTorch + Hugging Face: QLoRA fine-tune of Qwen2.5-VL-3B (trained
  on EC2 spot; two cycles: 0.472 -> 0.851 on 9 classes, then 0.903 on the
  12-class multi-material taxonomy with backward-compat measured on the
  archived v1 split and 0.900 on a cross-dataset OOD split); CLIP/CLAP embeddings for cross-modal + exemplar-image RAG and
  unsupervised audio anomaly scoring; a controlled thermal-fusion study
  reported as an honest negative (init-confound resolved to parity).
- **Serving** - FastAPI on a Lambda container behind CloudFront + API
  Gateway; async submit/poll job path (S3 + Lambda self-invoke); SageMaker
  async endpoint autoscaling 0-1 for the fine-tuned GPU model; Bedrock for
  vision reasoning; React SPA frontend.
- **Evals** - frozen golden sets with committed, regression-gated baselines
  for the classifier, retrieval, audio, the inspection agent (citation
  validity 0.741 -> 1.000 via an on-class filter), and the walkthrough
  (two golden sets: dataset crops + licensed realistic field photos).
- **Infra** - AWS CDK (Python), GitHub Actions CI/CD with keyless OIDC,
  CloudWatch ops dashboard, scale-to-zero cost posture.

## Run locally

    docker compose up -d db                        # pgvector (indexed corpus)
    uvicorn defectlens.serve.api:app --port 8000   # DEFECTLENS_NO_VLM=1 skips the 7GB VLM
    cd frontend && npm install && npm start        # http://localhost:3000

For the walkthrough locally: set `DEFECTLENS_LOCAL_JOBS=1` (in-process async
worker) and `DEFECTLENS_DESCRIBER=bedrock` on the API process. Tests:
`python -m pytest -q` and `cd frontend && npm test`. Realistic eval photos:
`bash scripts/fetch_realistic_walkthrough.sh` (licensed; see
`data/manifests/walkthrough_realistic_attribution.md`).

## Use it from an agent (MCP)

The system's three capabilities - photo analysis, standards search, and
walkthrough reports - are exposed as [Model Context Protocol](https://modelcontextprotocol.io)
tools, so any MCP client (Claude Desktop, Claude Code, custom agents) can
drive the live API:

    pip install -e ".[mcp]"
    claude mcp add sitecheck -- sitecheck-mcp      # Claude Code

Claude Desktop config:

    { "mcpServers": { "sitecheck": { "command": "sitecheck-mcp" } } }

Tools: `analyze_photo(path, note)` - ranked defect classes, severity, and
cited guidance; `search_standards(query)` - the cited corpus;
`run_walkthrough(photo_paths, visit_note, photo_notes)` - a grounded, cited
initial-diagnostic report. Set `SITECHECK_API_URL` to target a self-hosted
API (defaults to the public demo, which may cold-start on the first call).
The server is a thin client of the public HTTP API - it holds no models or
credentials. Status: new; unit-tested end to end with the live search path
verified, full agent-session round-trip on the roadmap below.

## Roadmap

Near-term (in flight when development paused):

- Agent-session end-to-end harness for the MCP server (scripted MCP client
  driving all three tools against the live API).
- Defect localization overlays (bounding boxes on findings): a controlled
  grounding experiment on the base VLM has run at the production pixel
  budget; the feature ships only if the hand-rated verdict clears the
  pre-registered bar (>=70% useful boxes), otherwise it gets reported as
  an honest negative alongside the thermal study. Early signal: the model
  boxes full scenes far more readily than close-up crops.

Designed, not built:

- Video walkthrough input: client-side frame extraction (sharpness-ranked,
  capped to the vision API's 20-image budget, deselectable thumbnails)
  feeding the existing photo pipeline unchanged.

Future directions:

- Cloud audio analysis: the CLAP anomaly path runs locally today and is
  pre-staged for the cloud deployment, pending a platform memory-limit
  increase.
- Taxonomy v3 candidates: roofing damage (shingle/hail) and pipe-corrosion
  datasets have been license-verified (CC BY / CC0) and are ready to
  ingest in a future fine-tune cycle.
- Exemplar-pool growth: several classes retrieve from thin exemplar pools
  (mold has a single documented case); a curated pass over public-domain
  agency photo libraries would deepen them.
- Faster CPU cold starts: an ONNX CLIP spike verified accuracy-perfect
  fp32 export with a much lighter import footprint; wiring it in is
  deferred until the audio question settles the Lambda image contents.
- Repeat-visit comparison (exploration): a second walkthrough of the same
  site producing a cited delta report against the first visit.

## Data and licenses

Trained/evaluated on CODEBRIM, BD3, SDNET2018, MBDD2025, VT Corrosion
Condition State, the figshare insulator-defect set, DCASE2020,
METU/Ozgenel, and BFDD (see `docs/datasets.md`); guidance corpus cites
EPA/HUD/InterNACHI/FHWA/NPS sources; UI gallery, realistic eval photos,
and the served exemplar images are licensed (PD/CC0/CC BY) with
attribution files alongside (`docs/exemplar-attribution.md`). Code is MIT.
