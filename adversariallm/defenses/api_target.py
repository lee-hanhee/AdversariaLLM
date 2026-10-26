"""Generation-only hosted targets, without pretending to expose local internals."""

from .base import TargetSystem
from ..lm_utils.text_generation import build_text_generator


class APITarget(TargetSystem):
    NAME = "openrouter"

    def __init__(self, config):
        super().__init__(None, None)
        self.config = config
        self.text_generator = build_text_generator(config)

    @classmethod
    def from_config(cls, cfg, *, model=None, tokenizer=None, default_generate_kwargs=None):
        return cls(cfg)

    def generate(self, convs, **kwargs):
        """Remove empty assistant placeholders and enforce declared prefix support."""
        messages = []
        for conv in convs:
            conv = [dict(message) for message in conv]
            if conv and conv[-1]["role"] == "assistant":
                if conv[-1]["content"] == "":
                    conv.pop()
                elif not self.config.get("supports_assistant_prefill", False):
                    raise ValueError("This hosted target has not declared assistant-prefill support.")
            messages.append(conv)
        for local_option in ("initial_batch_size", "verbose"):
            kwargs.pop(local_option, None)
        top_k = kwargs.pop("top_k", None)
        if top_k not in (None, 0):
            kwargs["top_k"] = top_k
        return self.text_generator.generate(messages, **kwargs)
