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
        def chat(self, prompt, image=None, max_new_tokens=400, images=None):
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


def test_mock_provider_records_image_count():
    p = MockProvider(responses=["ok"])
    p.complete("prompt", images=["img1", "img2"])
    assert p.calls[0].had_image is True
    assert p.calls[0].n_images == 2


def test_bedrock_provider_sends_one_block_per_image():
    from PIL import Image

    from defectlens.agent.providers import BedrockHaikuProvider

    captured = {}

    class FakeClient:
        def converse(self, **kwargs):
            captured.update(kwargs)
            return {
                "output": {"message": {"content": [{"text": "report"}]}},
                "usage": {"inputTokens": 10, "outputTokens": 5},
            }

    p = BedrockHaikuProvider()
    p._client = FakeClient()
    imgs = [Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))]
    out = p.complete("prompt", images=imgs, max_tokens=64)
    assert out == "report"
    content = captured["messages"][0]["content"]
    image_blocks = [b for b in content if "image" in b]
    assert len(image_blocks) == 2
    assert content[-1] == {"text": "prompt"}


def test_local_provider_forwards_images_to_chat():
    from defectlens.agent.providers import LocalQwenProvider

    class FakeDescriber:
        def __init__(self):
            self.kwargs = None

        def chat(self, prompt, image=None, max_new_tokens=400, images=None):
            self.kwargs = {"image": image, "images": images, "max_new_tokens": max_new_tokens}
            return "text"

    d = FakeDescriber()
    p = LocalQwenProvider(describer=d)
    assert p.complete("prompt", images=["a", "b"], max_tokens=128) == "text"
    assert d.kwargs == {"image": None, "images": ["a", "b"], "max_new_tokens": 128}


def test_providers_reject_both_image_and_images():
    import pytest

    p = MockProvider(responses=["ok"])
    with pytest.raises(ValueError):
        p.complete("prompt", image="a", images=["b"])
