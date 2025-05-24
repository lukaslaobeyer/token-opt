from pathlib import Path

from jaxtyping import Float

import torch
from torch import nn, Tensor
from einops import rearrange
from huggingface_hub import hf_hub_download

from titok.modeling.titok import PretrainedTokenizer


class PretrainedVQGAN(nn.Module):
    """MaskGIT-VQGAN wrapper compatible with test time opt"""

    def __init__(self):
        super().__init__()

        filename = Path("pretrained/maskgit-vqgan-imagenet-f16-256.bin")
        if not filename.exists():
            hf_hub_download(
                repo_id="fun-research/TiTok",
                filename=str(filename),
                local_dir=filename.parent,
            )
        self.tok = PretrainedTokenizer(filename)

        # Compatibility with TiTok module
        self.quantize_mode = "vq"
        self.latent_tokens = None

    def encoder(
        self, pixel_values: Float[Tensor, "b c h w"], latent_tokens
    ) -> Float[Tensor, "b d 1 n"]:
        assert latent_tokens is None # just for compatibility with TiTok interface
        z = self.tok.encoder(pixel_values)
        return rearrange(z, "b d h w -> b d 1 (h w)")

    def quantize(self, z: Float[Tensor, "b d 1 n"]) -> tuple[Float[Tensor, "b d 1 n"], dict]:
        z = rearrange(z, "b d 1 (h w) -> b d h w", h=16, w=16)
        # return_loss needed for gradients
        z_quant, codebook_indices, codebook_loss = self.tok.quantize(z, return_loss=True)
        z_quant = rearrange(z_quant, "b d h w -> b d 1 (h w)")
        return z_quant, {"indices": codebook_indices, "loss": codebook_loss}

    def decode(self, z_quant: Float[Tensor, "b d 1 n"]) -> Float[Tensor, "b c h w"]:
        z_quant = rearrange(z_quant, "b d 1 (h w) -> b d h w", h=16, w=16)
        return self.tok.decoder(z_quant)
