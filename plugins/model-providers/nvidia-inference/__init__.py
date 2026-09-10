"""NVIDIA Inference Hub provider profile.

Inference Hub (https://inference.nvidia.com) issues ``sk-*`` keys that
authenticate against ``https://inference-api.nvidia.com/v1``. This is a
separate product from NVIDIA NIM / build.nvidia.com, which uses ``nvapi-*``
keys on ``https://integrate.api.nvidia.com/v1`` (the ``nvidia`` provider).
"""

from providers import register_provider
from providers.base import ProviderProfile

nvidia_inference = ProviderProfile(
    name="nvidia-inference",
    aliases=("nvidia-inference-hub", "inference-nvidia", "inference-hub"),
    env_vars=(
        "NVIDIA_INFERENCE_API_KEY",
        "NVIDIA_API_KEY",
        "NVIDIA_INFERENCE_BASE_URL",
    ),
    display_name="NVIDIA Inference Hub",
    description=(
        "NVIDIA Inference Hub — sk-* keys via inference-api.nvidia.com "
        "(distinct from NIM / build.nvidia.com nvapi-* keys)"
    ),
    signup_url="https://inference.nvidia.com/key-management",
    fallback_models=(
        "nvidia/nemotron-3-super-120b-a12b",
        "nvidia/nemotron-3-nano-30b-a3b",
    ),
    base_url="https://inference-api.nvidia.com/v1",
    default_max_tokens=16384,
)

register_provider(nvidia_inference)
