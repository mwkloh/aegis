"""Where a benchmarked model actually ran.

The benchmark asks how well small models run *on this device*, so every result
records whether the model was local. `provider` cannot answer that: Ollama's
cloud models (`gemma4:cloud`) are addressed through the local daemon, so
`provider` reads "ollama" for a model that runs on someone else's hardware. The
tell is the daemon's own model list -- a pulled model is in `/api/tags`, a cloud
model is proxied and never is.
"""
from __future__ import annotations

from collections.abc import Collection
from typing import Literal

Locality = Literal["local", "cloud", "hosted"]


def classify_locality(provider: str, model: str, local_models: Collection[str]) -> Locality:
    """`local` only if the model is pulled onto this machine's Ollama.

    A `:cloud` tag is never local even if some daemon lists it -- the label must
    not be fooled into counting remote inference as on-device.
    """
    if provider != "ollama":
        return "hosted"
    if model.endswith(":cloud") or model not in local_models:
        return "cloud"
    return "local"


__all__ = ["Locality", "classify_locality"]
