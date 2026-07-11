from defectlens.agent.providers import MockProvider, Usage


def test_mock_provider_returns_scripted_responses_in_order():
    p = MockProvider(responses=["first", "second"])
    assert p.complete("prompt A") == "first"
    assert p.complete("prompt B", image=None) == "second"


def test_mock_provider_records_prompts():
    p = MockProvider(responses=["x"])
    p.complete("hello")
    assert p.calls[0].prompt == "hello"


def test_mock_provider_raises_when_exhausted():
    p = MockProvider(responses=[])
    import pytest

    with pytest.raises(IndexError):
        p.complete("anything")


def test_usage_accumulates():
    p = MockProvider(responses=["a", "b"])
    p.complete("one")
    p.complete("two")
    u = p.usage()
    assert isinstance(u, Usage)
    assert u.calls == 2 and u.input_tokens > 0


def test_local_qwen_provider_wraps_describer_chat():
    class FakeDescriber:
        def chat(self, prompt, image=None, max_new_tokens=400):
            return f"echo:{prompt}:{max_new_tokens}"

    from defectlens.agent.providers import LocalQwenProvider

    p = LocalQwenProvider(describer=FakeDescriber())
    assert p.complete("hi") == "echo:hi:1024"  # default token budget reaches chat
    assert p.usage().calls == 1


def test_usage_returns_a_copy_not_internal_state():
    p = MockProvider(responses=["a"])
    before = p.usage()
    p.complete("one")
    assert before.calls == 0 and p.usage().calls == 1


def test_image_to_png_bytes_roundtrips_rgba():
    import io

    from PIL import Image

    from defectlens.agent.providers import _image_to_png_bytes

    img = Image.new("RGBA", (4, 4), (10, 20, 30, 128))
    data = _image_to_png_bytes(img)
    back = Image.open(io.BytesIO(data))
    back.load()
    assert back.size == (4, 4) and back.mode == "RGBA"
    assert back.getpixel((0, 0)) == (10, 20, 30, 128)
