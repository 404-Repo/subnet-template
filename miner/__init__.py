import typing

import torch
from time import perf_counter

from shap_e.diffusion.sample import sample_latents
from shap_e.models.download import load_model, load_config
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config, GaussianDiffusion
from shap_e.models.transmitter.base import Transmitter
from shap_e.util.notebooks import decode_latent_mesh

from neurons import protocol




async def forward(synapse: protocol.Task404, device: torch.device) -> protocol.Task404:
    return synapse


def create_model(
        device: torch.device,
) -> tuple[Transmitter, torch.nn.Module, GaussianDiffusion]:
    cache_dir = f"shap_e_model_cache"  # TODO: set the cache dir
    xm = load_model("transmitter", device=device, cache_dir=cache_dir)
    model = load_model("text300M", device=device, cache_dir=cache_dir)
    config = load_config("diffusion")
    diffusion = diffusion_from_config(config)
    return (
        typing.cast(Transmitter, xm),
        typing.cast(torch.nn.Module, model),
        diffusion,
    )


def text_to_3d(device: torch.device, prompt: str, ply_output="test.ply"):
    xm, model, diffusion = create_model(device)

    batch_size = 1

    start = perf_counter()
    print("starting")

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

    inter = perf_counter()
    print(f"latents: {inter-start:0.4f}")

    for i, latent in enumerate(latents):
        t = decode_latent_mesh(xm, latent).tri_mesh()
        t.verts = t.verts[:, [0, 2, 1]]
        with open(ply_output, 'wb') as f:
            t.write_ply(f)

    end = perf_counter()
    print(f"saved: {end - inter:0.4f}")
