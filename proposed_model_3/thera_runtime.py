from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from contextlib import nullcontext
from dataclasses import dataclass

from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from torchvision import transforms
from torchvision.utils import save_image


@dataclass
class AdapterOutput(BaseOutput):
    sample: torch.FloatTensor


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        value, gate = self.proj(x).chunk(2, dim=-1)
        return value * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        inner = int(dim_in * mult)
        self.net = nn.Sequential(
            GEGLU(dim_in, inner),
            nn.Dropout(dropout),
            nn.Linear(inner, dim_out),
        )

    def forward(self, x):
        return self.net(x)


class TextAdapter(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, in_dim: int, int_dim: int, out_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.ff1 = FeedForward(in_dim, int_dim)
        self.norm2 = nn.LayerNorm(int_dim)
        self.ff2 = FeedForward(int_dim, out_dim)

    def forward(self, x):
        x = self.ff1(self.norm1(x))
        x = self.ff2(self.norm2(x))
        return AdapterOutput(sample=x)


class TherAAdapter(nn.Module):
    def __init__(self, adapter_dir: Path):
        super().__init__()
        self.adapter = TextAdapter.from_pretrained(str(adapter_dir))

        # TherA checkpoints include a scalar "scale" entry in llava_adapter.
        # Register it so the checkpoint state_dict loads exactly.
        self.register_buffer("scale", torch.tensor(1.0))

    def forward(self, hidden):
        return self.adapter(hidden).sample * self.scale


def convert_unet_to_8ch(unet: UNet2DConditionModel) -> UNet2DConditionModel:
    if int(unet.config.in_channels) == 8:
        return unet
    if int(unet.config.in_channels) != 4:
        raise ValueError(f"Expected a 4-channel UNet before conversion, got {unet.config.in_channels}")

    old = unet.conv_in
    new = nn.Conv2d(
        8,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=old.bias is not None,
    )

    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :4].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)

    unet.conv_in = new
    unet.register_to_config(in_channels=8)
    return unet


def load_cache(path: Path, device: torch.device):
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")

    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if value.ndim == 2:
        value = value.unsqueeze(0)
    return value.to(device)


def load_models(weights_dir: Path, device: str = "cuda"):
    device_obj = torch.device(device)

    checkpoint_path = weights_dir / "checkpoint" / "model.pt"
    if not checkpoint_path.exists():
        checkpoint_path = weights_dir / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"TherA checkpoint not found under {weights_dir}")

    merged = weights_dir / "merged_models"
    stable = weights_dir / "stable-diffusion"

    vae = AutoencoderKL.from_pretrained(str(stable), subfolder="vae")
    scheduler = DDIMScheduler.from_pretrained(str(stable), subfolder="scheduler")

    unet = UNet2DConditionModel.from_pretrained(str(merged / "unet"))
    unet = convert_unet_to_8ch(unet)

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "unet" not in state:
        raise KeyError("TherA checkpoint does not contain 'unet' weights")
    unet.load_state_dict(state["unet"], strict=True)

    adapter = TherAAdapter(merged / "adapter")
    if "llava_adapter" in state:
        adapter.load_state_dict(state["llava_adapter"], strict=True)

    for model in (vae, unet, adapter):
        model.requires_grad_(False)
        model.eval()
        model.to(device_obj)

    return vae, unet, adapter, scheduler, device_obj


def prepare_image(path: Path, device: torch.device, target_size: int | None):
    image = Image.open(path).convert("RGB")
    w, h = image.size

    if target_size:
        tw = th = int(target_size)
    else:
        tw = ((w + 31) // 32) * 32
        th = ((h + 31) // 32) * 32

    if (tw, th) != (w, h):
        image = image.resize((tw, th), Image.Resampling.LANCZOS)

    tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])(image).unsqueeze(0).to(device)
    return tensor


@torch.no_grad()
def translate(
    vae,
    unet,
    adapter,
    scheduler,
    rgb_tensor,
    reference_hidden,
    device,
    num_steps: int,
    cfg_text: float,
    cfg_image: float,
):
    rgb_latent = vae.encode(rgb_tensor).latent_dist.mode() * vae.config.scaling_factor

    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        autocast = torch.autocast("cuda", dtype=amp_dtype)
    else:
        autocast = nullcontext()

    with autocast:
        cond = adapter(reference_hidden.to(device))

    cond = cond.to(device=device, dtype=next(unet.parameters()).dtype)
    null_cond = torch.zeros_like(cond)

    latents = torch.randn(
        (1, 4, rgb_latent.shape[-2], rgb_latent.shape[-1]),
        device=device,
        dtype=rgb_latent.dtype,
    )

    scheduler.set_timesteps(int(num_steps), device=device)
    zeros_rgb = torch.zeros_like(rgb_latent)

    for step in scheduler.timesteps:
        t = torch.full((1,), int(step), device=device, dtype=torch.long)
        with_img = torch.cat([latents, rgb_latent], dim=1)
        no_img = torch.cat([latents, zeros_rgb], dim=1)

        eps_full = unet(with_img, t, encoder_hidden_states=cond).sample
        eps_no_text = unet(with_img, t, encoder_hidden_states=null_cond).sample
        eps_no_img = unet(no_img, t, encoder_hidden_states=cond).sample

        eps = (
            eps_full
            + float(cfg_text) * (eps_full - eps_no_text)
            + float(cfg_image) * (eps_full - eps_no_img)
        ).clamp(-5.0, 5.0)

        latents = scheduler.step(eps, step, latents, eta=0.0).prev_sample

    pred = vae.decode(latents / vae.config.scaling_factor).sample
    return (pred / 2 + 0.5).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", required=True)
    ap.add_argument("--rgb-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--reference-cache", required=True)
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--cfg-text", type=float, default=3.5)
    ap.add_argument("--cfg-image", type=float, default=1.5)
    ap.add_argument("--target-size", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    weights_dir = Path(args.weights_dir)
    rgb_dir = Path(args.rgb_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vae, unet, adapter, scheduler, device = load_models(weights_dir, args.device)
    reference_hidden = load_cache(Path(args.reference_cache), device)

    images = sorted(
        p for p in rgb_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )
    if not images:
        raise RuntimeError(f"No RGB frames found in {rgb_dir}")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    for index, image_path in enumerate(images, 1):
        rgb = prepare_image(image_path, device, args.target_size)
        pred = translate(
            vae, unet, adapter, scheduler, rgb, reference_hidden, device,
            num_steps=args.num_steps,
            cfg_text=args.cfg_text,
            cfg_image=args.cfg_image,
        )
        save_image(pred.cpu(), str(output_dir / image_path.name))
        print(f"[TherA] {index}/{len(images)} {image_path.name}", flush=True)


if __name__ == "__main__":
    main()
