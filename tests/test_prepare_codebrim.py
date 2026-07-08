import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_codebrim import parse_metadata, prepare  # noqa: E402

DEFECTS_XML = """\
<Annotation>
    <Defect name="crack1.png">
        <Background>0</Background>
        <Crack>1</Crack>
        <Spallation>0</Spallation>
        <Efflorescence>0</Efflorescence>
        <ExposedBars>0</ExposedBars>
        <CorrosionStain>0</CorrosionStain>
    </Defect>
    <Defect name="multi1.png">
        <Background>0</Background>
        <Crack>0</Crack>
        <Spallation>0</Spallation>
        <Efflorescence>1</Efflorescence>
        <ExposedBars>0</ExposedBars>
        <CorrosionStain>1</CorrosionStain>
    </Defect>
    <Defect name="ghost.png">
        <Background>0</Background>
        <Crack>1</Crack>
        <Spallation>0</Spallation>
        <Efflorescence>0</Efflorescence>
        <ExposedBars>0</ExposedBars>
        <CorrosionStain>0</CorrosionStain>
    </Defect>
</Annotation>
"""

BACKGROUND_XML = """\
<Annotation>
    <Defect name="bg1.png">
        <Background>1</Background>
        <Crack>0</Crack>
        <Spallation>0</Spallation>
        <Efflorescence>0</Efflorescence>
        <ExposedBars>0</ExposedBars>
        <CorrosionStain>0</CorrosionStain>
    </Defect>
</Annotation>
"""


def touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"fake")


def make_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Mini CODEBRIM classification_dataset: real-layout dirs + metadata XMLs."""
    source = tmp_path / "classification_dataset"
    (source / "metadata").mkdir(parents=True)
    (source / "metadata" / "defects.xml").write_text(DEFECTS_XML)
    (source / "metadata" / "background.xml").write_text(BACKGROUND_XML)
    touch(source / "train" / "defects" / "crack1.png")
    touch(source / "train" / "defects" / "multi1.png")
    touch(source / "train" / "background" / "bg1.png")
    touch(source / "train" / "defects" / "nometa.png")  # on disk, not in metadata
    # ghost.png is in metadata but has no file on disk
    return source, tmp_path / "codebrim_by_class"


def test_parse_metadata(tmp_path):
    xml = tmp_path / "defects.xml"
    xml.write_text(DEFECTS_XML)
    parsed = parse_metadata(xml)
    assert parsed["crack1.png"] == ["crack"]
    assert set(parsed["multi1.png"]) == {"efflorescence", "corrosion_stain"}
    assert len(parsed["multi1.png"]) == 2
    assert parsed["ghost.png"] == ["crack"]


def test_prepare_keeps_single_label(tmp_path):
    source, out = make_fixture(tmp_path)
    stats = prepare(source, out)
    kept = out / "crack" / "crack1.png"
    assert kept.is_symlink()
    assert kept.resolve() == (source / "train" / "defects" / "crack1.png").resolve()
    assert (out / "background" / "bg1.png").is_symlink()
    # multi-label crop must not appear anywhere in the staging tree
    assert not list(out.rglob("multi1.png"))
    assert stats["kept_crack"] == 1
    assert stats["kept_background"] == 1
    assert stats["multi_label_skipped"] == 1


def test_prepare_reports_missing_both_ways(tmp_path):
    source, out = make_fixture(tmp_path)
    stats = prepare(source, out)
    assert stats["missing_metadata"] == 1  # nometa.png on disk, no XML entry
    assert stats["metadata_without_file"] == 1  # ghost.png in XML, no file


def test_prepare_idempotent(tmp_path):
    source, out = make_fixture(tmp_path)
    first = prepare(source, out)
    second = prepare(source, out)  # must not raise on existing symlinks
    assert first == second
    assert len(list((out / "crack").iterdir())) == 1
