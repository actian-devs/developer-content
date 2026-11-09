"""The models and quantization levels under test, as GGUF files on Hugging Face.

The model variants follow the Phi-4 Mini vs Gemma 4 vs Qwen3 comparison on the same board:
Phi-4 Mini (3.8B), Gemma 4 E2B and E4B, and Qwen3-4B. All files come from Unsloth's
GGUF repos, so every model is quantized by the same tool chain.
"""
from dataclasses import dataclass

# Q4_0 is the plain 4-bit format; the K-quants (Q3_K_M to Q6_K) mix precisions across layers.
LEVELS = ["Q3_K_M", "Q4_0", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]


@dataclass(frozen=True)
class Model:
    key: str            # short name used in result paths
    name: str           # name used in the article
    repo: str           # Hugging Face repo
    pattern: str        # file name, with {level} for the quantization level
    template_kwargs: dict | None = None   # extra chat-template arguments sent with every request

    def file(self, level: str) -> str:
        return self.pattern.format(level=level)


MODELS = [
    Model("phi4-mini", "Phi-4 Mini", "unsloth/Phi-4-mini-instruct-GGUF", "Phi-4-mini-instruct-{level}.gguf"),
    Model("gemma4-e2b", "Gemma 4 E2B", "unsloth/gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-{level}.gguf"),
    Model("gemma4-e4b", "Gemma 4 E4B", "unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-{level}.gguf"),
    # Qwen3-4B thinks before it answers unless told not to; every model here answers directly
    Model("qwen3-4b", "Qwen3-4B", "unsloth/Qwen3-4B-GGUF", "Qwen3-4B-{level}.gguf",
          template_kwargs={"enable_thinking": False}),
]


def model_file(model: Model, level: str) -> str:
    """Unsloth names Phi-4 Mini's Q8_0 file with a dot instead of a dash."""
    if model.key == "phi4-mini" and level == "Q8_0":
        return "Phi-4-mini-instruct.Q8_0.gguf"
    return model.file(level)


def by_key(key: str) -> Model:
    for m in MODELS:
        if m.key == key:
            return m
    raise KeyError(f"unknown model {key!r}; choose from {[m.key for m in MODELS]}")
