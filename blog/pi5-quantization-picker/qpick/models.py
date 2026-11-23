"""The models and quantization levels under test, as GGUF files on Hugging Face.

The model variants match the Phi-4 Mini vs Gemma 4 vs Qwen3 comparison article:
Phi-4 Mini (3.8B), Gemma 4 E2B and E4B, and Qwen3-4B. The files come from Unsloth's GGUF
repos, except Phi-4 Mini at Q4_0, which Unsloth doesn't publish (see SOURCE_OVERRIDES below).
"""
from dataclasses import dataclass

# The six levels tested, from smallest file to largest.
# Q4_0 is the plain 4-bit format; the K-quants (Q3_K_M to Q6_K) mix precisions across layers.
LEVELS = ["Q3_K_M", "Q4_0", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]


@dataclass(frozen=True)
class Model:
    """One model under test: its names, where its GGUF files live, and any special chat settings."""
    key: str            # short name used in result paths
    name: str           # name used in the article
    repo: str           # Hugging Face repo
    pattern: str        # file name, with {level} for the quantization level
    template_kwargs: dict | None = None   # extra chat-template arguments sent with every request

    def file(self, level: str) -> str:
        """The file name for one level, e.g. Qwen3-4B-Q4_K_M.gguf."""
        return self.pattern.format(level=level)


# The four models under test
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


# Unsloth publishes no Q4_0 file for Phi-4 Mini, so that one level comes from bartowski's repo instead
SOURCE_OVERRIDES = {
    ("phi4-mini", "Q4_0"): ("bartowski/microsoft_Phi-4-mini-instruct-GGUF", "microsoft_Phi-4-mini-instruct-Q4_0.gguf"),
}


def model_source(model: Model, level: str) -> tuple[str, str]:
    """The Hugging Face repo and file name to download for one model at one level."""
    return SOURCE_OVERRIDES.get((model.key, level), (model.repo, model_file(model, level)))


def by_key(key: str) -> Model:
    """Find a model by its short name, as typed after --models."""
    for m in MODELS:
        if m.key == key:
            return m
    raise KeyError(f"unknown model {key!r}; choose from {[m.key for m in MODELS]}")
