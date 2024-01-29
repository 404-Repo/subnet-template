import typing

import bittensor as bt
import torch

from shap_e.diffusion.sample import sample_latents
from shap_e.models.download import load_model, load_config
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config, GaussianDiffusion
from shap_e.models.transmitter.base import Transmitter
from shap_e.util.notebooks import decode_latent_mesh

from neurons import protocol


async def forward(
    synapse: protocol.Task404, device: torch.device, cache_dir: str
) -> protocol.Task404:
    text_to_3d(synapse.prompt, device, cache_dir)
    return synapse


def create_model(
    device: torch.device, cache_dir: str
) -> tuple[Transmitter, torch.nn.Module, GaussianDiffusion]:
    model_cache_dir = f"{cache_dir}/shap_e_model_cache"
    xm = load_model("transmitter", device=device, cache_dir=model_cache_dir)
    model = load_model("text300M", device=device, cache_dir=model_cache_dir)
    config = load_config("diffusion")
    diffusion = diffusion_from_config(config)
    return (
        typing.cast(Transmitter, xm),
        typing.cast(torch.nn.Module, model),
        diffusion,
    )


def text_to_3d(prompt: str, device: torch.device, cache_dir: str):
    xm, model, diffusion = create_model(device, cache_dir)

    batch_size = 1

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=15,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    output = f"{cache_dir}/output.ply"

    for i, latent in enumerate(latents):
        t = decode_latent_mesh(xm, latent).tri_mesh()
        t.verts = t.verts[:, [0, 2, 1]]
        with open(output, "wb") as f:
            t.write_ply(f)
