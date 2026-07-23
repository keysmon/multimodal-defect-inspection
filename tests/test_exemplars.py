"""Exemplar-manifest contract tests (plan Task C1).

The serving posture is license-first: nothing outside {public_domain, cc0,
cc_by} is ever served, every entry carries credit + a recorded license check,
card joins must resolve, and class tags must be valid taxonomy tags. These
run against the COMMITTED manifest, so a bad curation batch fails CI before
anything ships.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from defectlens.corpus import VALID_CLASS_TAGS, load_corpus_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "data" / "exemplars" / "manifest.yaml"
ALLOWED_LICENSES = {"public_domain", "cc0", "cc_by"}


@pytest.fixture(scope="module")
def entries() -> list[dict]:
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(data.get("exemplars"), list) and data["exemplars"]
    return data["exemplars"]


@pytest.fixture(scope="module")
def corpus_ids() -> set[str]:
    return {c.id for c in load_corpus_dir(REPO_ROOT / "corpus")}


def test_ids_unique_and_well_formed(entries):
    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids))
    assert all(re.fullmatch(r"[a-z0-9][a-z0-9-]+", i) for i in ids)


def test_every_entry_license_allowed(entries):
    bad = [(e["id"], e.get("license")) for e in entries if e.get("license") not in ALLOWED_LICENSES]
    assert not bad, f"disallowed licenses (only PD/CC0/CC-BY serve): {bad}"


def test_every_entry_credited_and_checked(entries):
    for e in entries:
        assert e.get("credit", "").strip(), f"{e['id']}: empty credit"
        assert e.get("license_check", "").strip(), f"{e['id']}: license check not recorded"
        assert e.get("caption", "").strip(), f"{e['id']}: empty caption"
        assert re.fullmatch(r"[0-9a-f]{64}", e.get("sha256", "")), f"{e['id']}: bad sha256"
        assert e.get("source_url", "").startswith("https://"), f"{e['id']}: bad source_url"


def test_card_ids_resolve(entries, corpus_ids):
    for e in entries:
        unknown = [c for c in e.get("card_ids", []) if c not in corpus_ids]
        assert not unknown, f"{e['id']}: unknown card_ids {unknown}"


def test_class_tags_valid_and_nonempty(entries):
    for e in entries:
        tags = e.get("class_tags", [])
        assert tags, f"{e['id']}: class_tags empty"
        bad = [t for t in tags if t not in VALID_CLASS_TAGS]
        assert not bad, f"{e['id']}: invalid class_tags {bad}"


def test_new_classes_have_exemplar_coverage(entries):
    """The three v2 classes were the point of the expansion — each must have
    at least 3 exemplars so similar-cases retrieval has something to show."""
    from collections import Counter

    counts = Counter(t for e in entries for t in e["class_tags"])
    for cls in ("finish_detachment", "bulge_deformation", "insulator_damage"):
        assert counts[cls] >= 3, f"{cls}: only {counts[cls]} exemplars"
