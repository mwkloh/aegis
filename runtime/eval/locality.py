"""Where a benchmarked model actually ran.

The benchmark asks how well small models run *on this device*, so every result
records whether the model was local. `provider` cannot answer that: Ollama's
cloud models (`gemma4:cloud`) are addressed through the local daemon, so
`provider` reads "ollama" for a model that runs on someone else's hardware. The
tell is the daemon's own model list: a pulled cloud model *is* in `/api/tags`, but
with a `remote_host` and no weights, so callers pass only the entries without one
(`OllamaClient.list_local_models`).
"""
from __future__ import annotations

from collections.abc import Collection
from typing import Literal

Locality = Literal["local", "cloud", "hosted"]


def classify_locality(provider: str, model: str, local_models: Collection[str]) -> Locality:
    """`local` only if the model is pulled onto this machine's Ollama.

    A cloud tag (`:cloud`, or the older `:120b-cloud` style) is never local even if
    some list carries it -- the label must not be fooled into counting remote
    inference as on-device.
    """
    if provider != "ollama":
        return "hosted"
    name, tag = _split_tag(model)
    if tag == "cloud" or tag.endswith("-cloud"):
        return "cloud"
    if (name, tag) not in {_split_tag(m) for m in local_models}:
        return "cloud"
    return "local"


def _split_tag(model: str) -> tuple[str, str]:
    """(`name`, `tag`) the way Ollama reads a model reference: the tag is a `:` after the
    last `/` (so a registry port is not one), and an untagged name means `latest`."""
    name, sep, tag = model.rpartition(":")
    if not sep or "/" in tag:
        return model, "latest"
    return name, tag


__all__ = ["Locality", "classify_locality"]
