"""Check API client routing without real credentials or network requests."""

from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pytest

from adversariallm.lm_utils.text_generation import APITextGenerator


@pytest.fixture
def api_clients():
    """Provide fake API clients and prevent loading real credentials."""
    with (
        patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}, clear=True),
        patch("adversariallm.lm_utils.text_generation.load_dotenv") as dotenv,
        patch("adversariallm.lm_utils.text_generation.OpenAI") as client,
    ):
        yield client, dotenv


@pytest.mark.parametrize("model", ["mistralai/test-model", "qwen/test-model", "openai/test-model"])
def test_openrouter_routes_and_preserves_api_model_id(api_clients, model):
    """Check that OpenRouter receives the model ID without the routing prefix."""

    # Isolate client selection with fake credentials and SDK constructors.
    client, dotenv = api_clients
    generator = APITextGenerator(f"openrouter/{model}")

    # Keep the API model ID intact after removing only the repository routing prefix.
    client.assert_called_once_with(base_url="https://openrouter.ai/api/v1", api_key="test-key")
    assert dotenv.call_args.kwargs == {"override": False}
    assert dotenv.call_args.args[0].name == ".env"

    # Serialize one completion through the selected client and inspect its final model ID.
    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.content = "response"
    client.return_value.chat.completions.create.return_value = completion
    messages = [{"role": "user", "content": "hello"}]
    assert generator._api_call(messages) == ["response"]
    client.return_value.chat.completions.create.assert_called_once_with(model=model, messages=messages)


def test_openrouter_base_url_override(api_clients, monkeypatch):
    """Check that an environment variable can override the OpenRouter URL."""

    # Isolate client selection with fake credentials and SDK constructors.
    client, _ = api_clients
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://example.test/v1")
    APITextGenerator("openrouter/qwen/test-model")

    # Use the explicitly configured endpoint without changing model routing.
    client.assert_called_once_with(base_url="https://example.test/v1", api_key="test-key")


def test_legacy_routing_is_preserved(api_clients, monkeypatch):
    """Check that existing GPT model names still use the original client."""

    # Isolate client selection with fake credentials and SDK constructors.
    client, _ = api_clients
    monkeypatch.setenv("GPT_API_KEY", "legacy-key")
    monkeypatch.setenv("BASE_URL_GPT", "https://legacy.test/v1")
    generator = APITextGenerator("gpt-4o")

    # Keep existing GPT model names on their legacy client.
    assert generator.model_name == "gpt-4o"
    assert generator.client is generator.clients["gpt"]
    assert client.call_count == 2


def test_openrouter_does_not_silently_use_legacy_client(api_clients, monkeypatch):
    """Check that missing OpenRouter credentials cause an error, not a fallback."""

    # Isolate client selection with fake credentials and SDK constructors.
    monkeypatch.delenv("OPENROUTER_API_KEY")
    monkeypatch.setenv("GPT_API_KEY", "legacy-key")
    monkeypatch.setenv("BASE_URL_GPT", "https://legacy.test/v1")

    # Missing OpenRouter credentials must fail instead of selecting another client.
    with pytest.raises(ValueError, match="openrouter client is not available"):
        APITextGenerator("openrouter/openai/test-model")


def test_openrouter_requires_model_id(api_clients):
    """Check that the OpenRouter prefix must be followed by a model ID."""

    # Reject an empty model ID before making a request.
    with pytest.raises(ValueError, match="API model ID is required"):
        APITextGenerator("openrouter/")


def test_legacy_api_generation_keeps_native_sample_and_token_parameters(api_clients, monkeypatch):
    # Isolate client selection with fake credentials and SDK constructors.
    client, _ = api_clients
    monkeypatch.setenv("GPT_API_KEY", "legacy-key")
    monkeypatch.setenv("BASE_URL_GPT", "https://legacy.test/v1")
    openrouter, legacy = MagicMock(), MagicMock()
    client.side_effect = [openrouter, legacy]
    legacy.chat.completions.create.return_value = SimpleNamespace(
        id="legacy-id", model="gpt-4o", usage=None,
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")
                 for text in ["one", "two"]],
    )
    generator = APITextGenerator("gpt-4o")
    messages = [{"role": "user", "content": "Hello"}]

    # Keep native sample/token parameters and avoid calling the OpenRouter client.
    assert generator.generate([messages], num_return_sequences=2, max_new_tokens=20).gen == [["one", "two"]]
    legacy.chat.completions.create.assert_called_once_with(
        model="gpt-4o", messages=messages, n=2, max_completion_tokens=20,
    )
    openrouter.chat.completions.create.assert_not_called()


def test_legacy_api_exhausted_retries_still_return_empty_response(api_clients, monkeypatch):
    # Isolate client selection with fake credentials and SDK constructors.
    client, _ = api_clients
    monkeypatch.setenv("GPT_API_KEY", "legacy-key")
    monkeypatch.setenv("BASE_URL_GPT", "https://legacy.test/v1")
    client.return_value.chat.completions.create.side_effect = RuntimeError("request failed")
    monkeypatch.setattr("adversariallm.lm_utils.text_generation.time.sleep", lambda _: None)
    generator = APITextGenerator("gpt-4o")

    # Retain the legacy retry limit and empty-result behavior.
    assert generator.generate([[{"role": "user", "content": "Hello"}]]).gen == [[]]
    assert client.return_value.chat.completions.create.call_count == 3
