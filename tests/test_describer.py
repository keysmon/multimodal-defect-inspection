import subprocess
import sys

from defectlens.serve.describer import Describer, _decode_tail, build_prompt, vlm_disabled

# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


def test_build_prompt_humanizes_underscored_class_names():
    prompt = build_prompt(["exposed_rebar", "mold_algae"])
    assert "exposed rebar" in prompt
    assert "mold algae" in prompt
    assert "_" not in prompt


def test_build_prompt_caps_at_three_classes():
    prompt = build_prompt(["crack", "spalling", "mold_algae", "no_defect"])
    assert "crack" in prompt
    assert "spalling" in prompt
    assert "mold algae" in prompt
    assert "no defect" not in prompt


def test_build_prompt_is_deterministic():
    assert build_prompt(["crack"]) == build_prompt(["crack"])


# ---------------------------------------------------------------------------
# vlm_disabled
# ---------------------------------------------------------------------------


def test_vlm_disabled_true_when_env_set_to_1(monkeypatch):
    monkeypatch.setenv("DEFECTLENS_NO_VLM", "1")
    assert vlm_disabled() is True


def test_vlm_disabled_false_when_env_unset(monkeypatch):
    monkeypatch.delenv("DEFECTLENS_NO_VLM", raising=False)
    assert vlm_disabled() is False


def test_vlm_disabled_false_for_non_1_values(monkeypatch):
    monkeypatch.setenv("DEFECTLENS_NO_VLM", "true")
    assert vlm_disabled() is False


# ---------------------------------------------------------------------------
# Describer.describe — degrade path (no model download/load involved)
# ---------------------------------------------------------------------------


def test_describe_returns_empty_string_when_disabled(monkeypatch):
    monkeypatch.setenv("DEFECTLENS_NO_VLM", "1")
    d = Describer()
    assert d.describe(object(), ["crack"]) == ""


def test_describe_returns_empty_string_when_not_loaded(monkeypatch):
    monkeypatch.delenv("DEFECTLENS_NO_VLM", raising=False)
    d = Describer()
    assert d.model is None
    assert d.processor is None
    assert d.describe(object(), ["crack"]) == ""


# ---------------------------------------------------------------------------
# _decode_tail — pure trimming/decoding helper, tested with simple fakes
# ---------------------------------------------------------------------------


def test_decode_tail_slices_off_the_prompt_tokens_before_decoding():
    class FakeProcessor:
        def batch_decode(self, sequences, skip_special_tokens=True):
            assert skip_special_tokens is True
            assert sequences == [[5, 6, 7]]
            return ["  Visible cracking along the wall.  "]

    result = _decode_tail(FakeProcessor(), [[1, 2, 3, 4, 5, 6, 7]], input_len=4)
    assert result == "Visible cracking along the wall."


def test_decode_tail_strips_whitespace_from_decoded_text():
    class FakeProcessor:
        def batch_decode(self, sequences, skip_special_tokens=True):
            return ["\n  padded description  \n"]

    result = _decode_tail(FakeProcessor(), [[9, 9, 1, 2]], input_len=2)
    assert result == "padded description"


# ---------------------------------------------------------------------------
# Describer.describe — full generation wiring, model/processor mocked
# ---------------------------------------------------------------------------


def test_describe_wires_prompt_through_to_trimmed_decoded_output(monkeypatch):
    monkeypatch.delenv("DEFECTLENS_NO_VLM", raising=False)

    class FakeTensor(list):
        @property
        def shape(self):
            return (len(self), len(self[0]) if self else 0)

    class FakeBatchEncoding(dict):
        def __init__(self, input_ids):
            super().__init__(input_ids=input_ids)
            self.input_ids = input_ids

        def to(self, device):
            return self

    class FakeProcessor:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            assert add_generation_prompt is True
            assert tokenize is False
            content = messages[0]["content"]
            assert content[0] == {"type": "image", "image": "fake-image"}
            assert content[1]["type"] == "text"
            assert "crack" in content[1]["text"]
            return "PROMPT_TEXT"

        def __call__(self, text, images, return_tensors):
            assert text == ["PROMPT_TEXT"]
            assert images == ["fake-image"]
            assert return_tensors == "pt"
            return FakeBatchEncoding(FakeTensor([[1, 2, 3, 4]]))

        def batch_decode(self, sequences, skip_special_tokens=True):
            assert list(sequences) == [[5, 6, 7]]
            return ["  Visible cracking along the wall.  "]

    class FakeModel:
        def generate(self, **kwargs):
            assert kwargs.get("max_new_tokens") == 120
            assert kwargs.get("do_sample") is False
            return FakeTensor([[1, 2, 3, 4, 5, 6, 7]])

    d = Describer()
    d.model = FakeModel()
    d.processor = FakeProcessor()
    d.device = "cpu"

    result = d.describe("fake-image", ["crack"])
    assert result == "Visible cracking along the wall."


# ---------------------------------------------------------------------------
# Import sanity — describer module must stay cheap to import (spec §7)
# ---------------------------------------------------------------------------


def test_module_import_does_not_pull_in_torch_or_transformers():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "import defectlens.serve.describer\n"
            "assert 'torch' not in sys.modules, 'torch imported at module level'\n"
            "assert 'transformers' not in sys.modules, 'transformers imported at module level'\n"
            "print('OK')\n",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"
