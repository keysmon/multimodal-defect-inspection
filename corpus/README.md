# Guidance Card Corpus

This directory contains YAML-formatted guidance cards for the DefectLens RAG system. Cards encode defect detection, remediation, and inspection guidance indexed by building defect class.

## YAML Schema

Each file must contain a `source` block and a `cards` array. Here is the canonical structure:

```yaml
source:
  name: "EPA — Mold Remediation in Schools and Commercial Buildings"
  url: "https://www.epa.gov/mold/..."
  license: "US Government public domain"
cards:
  - id: epa-mold-001
    title: "Mold growth on interior walls"
    class_tags: [mold_algae, water_damage]
    severity: monitor
    index_sentence: "mold or algae growth on damp interior building surfaces"
    passage: >
      2-4 sentences of remediation/inspection guidance in your own words.
    citation: "EPA 402-K-01-001, Chapter 2"
```

### Field Definitions

- **source.name**: Human-readable source title (e.g., agency, standard, publication)
- **source.url**: URL to the authoritative source document
- **source.license**: License or copyright status of the source (e.g., "US Government public domain", "CC BY-4.0", "proprietary")
- **id**: Unique kebab-case identifier scoped to this file; must be unique across all files in the corpus
- **title**: Short, descriptive title (1-2 phrases)
- **class_tags**: List of zero or more defect classes from the unified taxonomy (see valid values below)
- **severity**: One of: `structural`, `urgent`, `monitor`, `cosmetic`
- **index_sentence**: Short CLIP-text-friendly sentence (≤120 chars) suitable for embedding; describes the defect or guidance in neutral language
- **passage**: 2-4 sentences of actionable inspection or remediation guidance **in your own words** (see rules below)
- **citation**: Full reference to the source section (e.g., section number, page range, standard code)

### Valid class_tags (Unified Taxonomy)

All nine defect classes are:

```
crack
corrosion
mold_algae
water_damage
structural_settlement
asbestos
lead_paint
pest_infestation
efflorescence
```

## Severity Scale

- **structural**: Safety risk or deferred maintenance requiring professional referral (e.g., foundation cracks >¼ inch, beam rot, load-bearing wall damage)
- **urgent**: Active deterioration or conditions that will rapidly worsen without prompt action (e.g., active water intrusion, mold blooming, severe corrosion)
- **monitor**: Ongoing maintenance watch items that require periodic inspection and may eventually require repair (e.g., minor hairline cracks, early-stage efflorescence, small moisture stains)
- **cosmetic**: Appearance-only defects with no structural or safety implication (e.g., minor surface crazing, paint fading, superficial stains)

## Authoring Rules

### 1. Passage Content

- Write guidance **in your own words** based on the source material
- **Do NOT verbatim copy** text from ICC codes, standards, or regulations
- **Cite sections by reference only** in the `citation` field
- Aim for 2-4 sentences; be specific and actionable
- Assume the reader is a building inspector, property manager, or maintenance professional

### 2. Unique IDs

- Format: `kebab-case`, e.g., `epa-mold-001`, `icc-2024-cracks-002`
- Must be unique **across all files** in the corpus
- Recommend a source abbreviation + descriptor + serial number (e.g., `[source]-[class/topic]-[counter]`)

### 3. Class Tags

- Must be a subset of the nine unified classes listed above
- At least one tag per card (zero tags will be rejected)
- Use all applicable classes if guidance covers multiple defect types

### 4. Index Sentence

- Optimized for CLIP embeddings (semantic similarity search)
- Keep ≤120 characters
- Use clear, neutral language describing the defect or guidance
- Avoid jargon; aim for general comprehension
- Examples:
  - "visible crack wider than 1/8 inch in structural walls"
  - "white powdery deposits on masonry surfaces after moisture exposure"
  - "active mold growth with musty odor in basement or crawlspace"

### 5. Citation

- Full, traceable reference to the source document
- Include section, chapter, page, or standard code (e.g., "EPA 402-K-01-001, Chapter 2", "ICC IBC 2024 §R703.2", "NFPA 101 §10.2.2")
- Enables verification and legal defensibility

## Coverage Requirements

The corpus must meet the following thresholds to be release-ready:

- **Minimum per class**: Each of the 9 unified classes must appear in at least 15 cards
- **Total size**: Minimum 200 cards
- **All sources sourced**: Every card must cite a legitimate, traceable authority (standards, codes, government guidance, peer-reviewed research)
- **No duplicates**: All card IDs must be unique

## File Organization

- One file per source (e.g., `epa-mold.yaml`, `icc-2024-residential.yaml`)
- Use source abbreviation in filename for clarity
- Alphabetically sorted filenames recommended (loader processes lexicographically)

## Validation

Use the Python loader to validate your YAML before committing:

```python
from defectlens.corpus import load_corpus_dir
from pathlib import Path

cards = load_corpus_dir(Path("corpus"))
print(f"Loaded {len(cards)} cards")
```

Validation checks:
- YAML syntax correctness
- Required fields present and non-empty
- class_tags are valid (subset of unified taxonomy)
- severity is one of: structural, urgent, monitor, cosmetic
- All card IDs are unique within the corpus
- ID format is kebab-case

## Examples

See generated example files (authored in Phase 2 Task 4) for reference implementations.
