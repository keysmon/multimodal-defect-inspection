from pathlib import Path

import pytest

from defectlens.corpus import load_corpus_dir, load_corpus_file

VALID = """
source:
  name: "Test Source"
  url: "https://example.gov/x"
  license: "public domain"
cards:
  - id: test-001
    title: "Crack guidance"
    class_tags: [crack]
    severity: monitor
    index_sentence: "a crack in a wall"
    passage: "Watch it."
    citation: "Test §1"
"""


def write(tmp_path, text, name="s.yaml"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_load_valid_card(tmp_path):
    cards = load_corpus_file(write(tmp_path, VALID))
    assert len(cards) == 1
    c = cards[0]
    assert c.id == "test-001" and c.class_tags == ["crack"]
    assert c.source_name == "Test Source"


def test_unknown_class_tag_rejected(tmp_path):
    bad = VALID.replace("[crack]", "[cracks]")
    with pytest.raises(ValueError, match="cracks"):
        load_corpus_file(write(tmp_path, bad))


def test_bad_severity_rejected(tmp_path):
    bad = VALID.replace("severity: monitor", "severity: meh")
    with pytest.raises(ValueError, match="meh"):
        load_corpus_file(write(tmp_path, bad))


def test_missing_field_rejected(tmp_path):
    bad = VALID.replace("    citation: \"Test §1\"\n", "")
    with pytest.raises(ValueError, match="citation"):
        load_corpus_file(write(tmp_path, bad))


def test_duplicate_ids_across_dir_rejected(tmp_path):
    write(tmp_path, VALID, "a.yaml")
    write(tmp_path, VALID, "b.yaml")
    with pytest.raises(ValueError, match="test-001"):
        load_corpus_dir(tmp_path)


def test_all_classes_covered_check(tmp_path):
    write(tmp_path, VALID)
    cards = load_corpus_dir(tmp_path)
    covered = {t for c in cards for t in c.class_tags}
    assert covered == {"crack"}


def test_non_string_passage_rejected(tmp_path):
    bad = VALID.replace('passage: "Watch it."', "passage: 12345")
    with pytest.raises(ValueError, match="passage"):
        load_corpus_file(write(tmp_path, bad))


def test_string_class_tags_rejected(tmp_path):
    bad = VALID.replace("class_tags: [crack]", "class_tags: crack")
    with pytest.raises(ValueError, match="must be a list"):
        load_corpus_file(write(tmp_path, bad))


def test_hvac_maintenance_cards_coverage():
    from collections import Counter

    from defectlens.corpus import load_corpus_dir
    from defectlens.taxonomy import AUDIO_FAULT_TAGS

    cards = [c for c in load_corpus_dir(Path("corpus")) if c.id.startswith("hvac-")]
    assert len(cards) >= 40
    tags = Counter(t for c in cards for t in c.class_tags)
    for tag in AUDIO_FAULT_TAGS:
        assert tags[tag] >= 4, tag


def test_visual_card_set_excludes_audio_and_maps_to_unified_classes():
    # Regression lock for the visual/audio split guards in rag/embed.py and
    # eval/rag_recall.py: the visual card set (non-hvac-*) must contain zero audio
    # fault tags, so it never pollutes the 768-dim visual index and every tag is a
    # UNIFIED_CLASS that rag_recall's UNIFIED_CLASSES.index() can resolve.
    from defectlens.corpus import is_audio_card, load_corpus_dir
    from defectlens.taxonomy import AUDIO_FAULT_TAGS, UNIFIED_CLASSES

    all_cards = load_corpus_dir(Path("corpus"))
    audio = [c for c in all_cards if is_audio_card(c)]
    visual = [c for c in all_cards if not is_audio_card(c)]
    assert audio, "expected hvac-* audio cards present in corpus"
    assert visual, "expected visual (non-hvac-*) cards present in corpus"
    assert all(not c.id.startswith("hvac-") for c in visual)
    assert not any(set(c.class_tags) & set(AUDIO_FAULT_TAGS) for c in visual)
    # mirrors rag_recall.py's UNIFIED_CLASSES.index(t); must not raise on the corpus
    assert all(t in UNIFIED_CLASSES for c in visual for t in c.class_tags)
