from tto.ema import EMAModel
from tto.siglip import SigLIP
from tto.vqgan_wrapper import PretrainedVQGAN

from typing import cast, Callable, Literal
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
from einops import rearrange, einsum
from jaxtyping import Float
import open_clip

from titok.modeling.quantizer import DiagonalGaussianDistribution
from titok.modeling.titok import TiTok


@dataclass
class TestTimeOptConfig:
    titok_checkpoint: str = "yucornetto/tokenizer_titok_l32_imagenet"
    optimize_post_quantization_tokens: bool = False
    vae_deterministic_sampling: bool = True
    lr: float = 1e-1
    ema_decay: float = 0.
    token_noise: float | None = None
    reg_weight: float | None = None
    reg_type: None | Literal["seed", "zero"] = None
    num_iter: int = 600
    enable_amp: bool = False


@dataclass
class TestTimeOptInfo:
    i: int
    tokens: Float[Tensor, "b d 1 n"]
    img: Float[Tensor, "b c h w"]
    loss: Float[Tensor, "b"]


ObjectiveT = Callable[[Float[Tensor, "b c h w"]], Float[Tensor, "b"]]


class TestTimeOpt(nn.Module):
    def __init__(
        self,
        config: TestTimeOptConfig,
        objective: ObjectiveT,
    ):
        super().__init__()
        self.config = config
        self.objective = objective
        if config.titok_checkpoint == "maskgit-vqgan":
            print("Using pretrained MaskGIT-VQGAN!")
            self.titok = PretrainedVQGAN()
        else:
            self.titok = TiTok.from_pretrained(config.titok_checkpoint)
        self.eval()

    def decode(self, tokens: Float[Tensor, "b d 1 n"]) -> Float[Tensor, "b c h w"]:
        def _maybe_quantize(tokens):
            if self.config.optimize_post_quantization_tokens:
                return tokens
            else:
                if self.titok.quantize_mode == "vae":
                    assert isinstance(self.titok, TiTok)
                    tokens = self.titok.quantize(tokens)
                    return (
                        tokens.mean
                        if self.config.vae_deterministic_sampling
                        else tokens.sample()
                    )
                else:
                    return self.titok.quantize(tokens)[0] # type: ignore
        tokens = _maybe_quantize(tokens)
        dec = self.titok.decode(tokens)
        return dec

    def encode(self, img: Float[Tensor, "b c h w"]) -> Float[Tensor, "b d 1 n"]:
        tok = self.titok.encoder(pixel_values=img, latent_tokens=self.titok.latent_tokens)
        if self.config.optimize_post_quantization_tokens:
            if self.titok.quantize_mode == "vae":
                tok = DiagonalGaussianDistribution(tok)
                return tok.mean if self.config.vae_deterministic_sampling else tok.sample()
            else:
                assert isinstance(tok, Tensor)
                return self.titok.quantize(tok)[0] # type: ignore
        return tok

    def _token_noise_schedule(self, i):
        """Quadratic noise decay"""
        t = i / (self.config.num_iter - 1)
        t = max(0, min(1, 1.5 * t)) # make it ramp to 0 at 2/3 of num_iter
        return 0.5 * (1 + np.cos(np.pi * t))

    def forward(
        self,
        seed: Float[Tensor, "b c h w"] | None,
        seed_tokens: Float[Tensor, "b d 1 n"] | None=None,
        callback: Callable[[TestTimeOptInfo], bool | None] | None=None,
        token_reset_callback: Callable[[TestTimeOptInfo], Float[Tensor, "b d 1 n"] | None] | None=None,
    ):
        assert not self.training
        if seed is not None:
            if seed_tokens is not None:
                raise ValueError("must provide seed_tokens or seed but not both")
            with torch.no_grad():
                opt_tokens = self.encode(seed)
        else:
            if seed_tokens is None:
                raise ValueError("must provide either seed_tokens or seed")
            opt_tokens = seed_tokens.detach().clone()

        opt_tokens.requires_grad_(True)
        opt = torch.optim.Adam(params=[opt_tokens], lr=self.config.lr)
        scaler = torch.GradScaler(enabled=self.config.enable_amp)
        ema = EMAModel(
            [opt_tokens],
            decay=self.config.ema_decay,
            min_decay=self.config.ema_decay
        )
        orig_tokens = opt_tokens.detach().clone()

        for i in range(self.config.num_iter):
            if self.config.token_noise is not None:
                with torch.no_grad():
                    opt_tokens.add_(
                        self.config.token_noise
                        * self._token_noise_schedule(i)
                        * torch.randn_like(opt_tokens)
                    )
            with torch.autocast(
                orig_tokens.device.type,
                torch.float16,
                enabled=self.config.enable_amp,
            ):
                dec = self.decode(opt_tokens)
                loss = self.objective(dec)
                if self.config.reg_weight is not None:
                    assert self.config.reg_type is not None
                    if self.config.reg_type == "seed":
                        reg = self.config.reg_weight * torch.mean(
                            (opt_tokens - orig_tokens)**2, dim=(1, 2, 3)
                        )
                    elif self.config.reg_type == "zero":
                        reg = self.config.reg_weight * torch.mean(
                            opt_tokens**2, dim=(1, 2, 3)
                        )
                    else:
                        assert False
                else:
                    reg = 0
                sum_loss = torch.sum(loss + reg, dim=0)
            scaler.scale(sum_loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

            # Token reset
            with torch.no_grad():
                if token_reset_callback is not None and (
                    tokens_reset := token_reset_callback(TestTimeOptInfo(
                        i=i,
                        tokens=opt_tokens,
                        img=dec,
                        loss=loss,
                    ))
                ) is not None:
                    opt_tokens.copy_(tokens_reset.detach())

            ema.step()
            with ema.average_parameters(), torch.no_grad():
                if callback is not None and callback(TestTimeOptInfo(
                    i=i,
                    tokens=opt_tokens, # tokens with EMA
                    img=dec, # image decoded *without* EMA
                    loss=loss,
                )):
                    break
        with ema.average_parameters(), torch.no_grad():
            return torch.clamp(
                self.decode(opt_tokens), 0.0, 1.0
            )


class AugmentationHelper:
    def __init__(
        self,
        num_augmentations: int,
        img_size,
    ):
        self.num_augmentations = num_augmentations
        if num_augmentations >= 1:
            self.augmentations = v2.Compose([
                v2.RandomCrop(size=img_size),
                v2.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.augmentations = None

    def __call__(
        self,
        x: Float[Tensor, "b c h_in w_in"]
    ) -> Float[Tensor, "num_aug b c h w"]:
        if self.augmentations is None:
            return x.unsqueeze(0)
        else:
            return torch.stack(
                [self.augmentations(x) for _ in range(self.num_augmentations)]
            )


class CLIPObjective(nn.Module):
    device_indicator: Tensor

    def __init__(
        self,
        prompt: str | list[str] | None=None,
        neg_prompt: str | list[str] | None=None,
        cfg_scale: float = 1.,
        num_augmentations: int=0,
        pretrained: tuple[str, str]=("ViT-B-32", "laion2b_s34b_b79k"),
    ):
        super().__init__()

        self.register_buffer("device_indicator", torch.tensor(0))
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            pretrained[0], pretrained=pretrained[1]
        )
        self.clip_tokenizer = cast(open_clip.SimpleTokenizer, open_clip.get_tokenizer("ViT-B-32"))
        self.augment = AugmentationHelper(
            num_augmentations=num_augmentations,
            img_size=self.clip_model.visual.image_size,
        )
        self.eval()

        self._prompt = prompt
        self._prompt_feat = None
        self._neg_prompt = neg_prompt
        self._neg_prompt_feat = None
        self.neg_prompt_weight = 1 - cfg_scale

    @property
    def prompt(self):
        return self._prompt

    @prompt.setter
    def prompt(self, prompt):
        self._prompt_feat = None
        self._prompt = prompt

    @property
    @torch.no_grad
    def prompt_feat(self) -> Float[Tensor, "#b d"]:
        assert not self.training
        if self._prompt_feat is None:
            prompt_feat = self.clip_model.encode_text(self.tokenize(self.prompt))
            prompt_feat = prompt_feat / prompt_feat.norm(dim=-1, keepdim=True)
            self._prompt_feat = prompt_feat
        return self._prompt_feat

    @property
    def neg_prompt(self):
        return self._neg_prompt

    @neg_prompt.setter
    def neg_prompt(self, prompt):
        self._neg_prompt_feat = None
        self._neg_prompt = prompt

    @property
    @torch.no_grad
    def neg_prompt_feat(self) -> Float[Tensor, "#b d"]:
        assert not self.training
        if self._neg_prompt_feat is None:
            prompt_feat = self.clip_model.encode_text(self.tokenize(self.neg_prompt))
            prompt_feat = prompt_feat / prompt_feat.norm(dim=-1, keepdim=True)
            self._neg_prompt_feat = prompt_feat
        return self._neg_prompt_feat

    def preprocess(self, img):
        """apply (differentiable) transforms: normalization + resize to CLIP input size"""
        resize = self.clip_preprocess.transforms[0] # type: ignore
        normalize = self.clip_preprocess.transforms[4] # type: ignore
        if not (img.shape[-1] == img.shape[-2] == resize.size):
            img = F.interpolate(img, size=resize.size, mode="bilinear")
        img = normalize(img)
        return img

    def tokenize(self, text):
        if isinstance(text, str):
            text = [text]
        return self.clip_tokenizer(text).to(self.device_indicator.device)

    def forward(self, img: Float[Tensor, "b c h w"]) -> Float[Tensor, "b"]:
        assert not self.training
        augs = self.augment(img) # num_aug b c h w
        num_augs = augs.shape[0]
        augs = rearrange(augs, "n b c h w -> (n b) c h w")
        image_feats = self.clip_model.encode_image(self.preprocess(augs))
        image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
        image_feats = rearrange(image_feats, "(n b) d -> n b d", n=num_augs)
        similarity = torch.mean(
            einsum(image_feats, self.prompt_feat.mT, "n b d, d b -> n b"),
            dim=0
        )
        if self.neg_prompt is not None:
            neg_similarity = torch.mean(
                einsum(image_feats, self.neg_prompt_feat.mT, "n b d, d b -> n b"),
                dim=0
            )
            return -similarity - self.neg_prompt_weight * neg_similarity
        else:
            return -similarity


class SigLIPObjective(nn.Module):
    def __init__(
        self,
        prompt: str | list[str] | None=None,
        num_augmentations: int=0,
    ):
        super().__init__()
        self.siglip = SigLIP()
        self.augment = AugmentationHelper(
            num_augmentations=num_augmentations,
            img_size=224,
        )
        self.eval()

        self._prompt = prompt
        self._prompt_feat = None

    @property
    def prompt(self):
        return self._prompt

    @prompt.setter
    def prompt(self, prompt):
        self._prompt_feat = None
        self._prompt = prompt

    @property
    @torch.no_grad
    def prompt_feat(self) -> Float[Tensor, "#b d"]:
        assert not self.training
        if self._prompt_feat is None:
            self._prompt_feat = self.siglip.encode_text(self._prompt)
        return self._prompt_feat

    def preprocess(self, img):
        return F.interpolate(img, size=224, mode="bilinear")

    def forward(self, img: Float[Tensor, "b c h w"]) -> Float[Tensor, "b"]:
        assert not self.training
        augs = self.augment(img) # num_aug b c h w
        num_augs = augs.shape[0]
        augs = rearrange(augs, "n b c h w -> (n b) c h w")
        image_feats = self.siglip.encode_img(self.preprocess(augs), differentiable=True)
        image_feats = rearrange(image_feats, "(n b) d -> n b d", n=num_augs)
        return -torch.mean(
            self.siglip.similarity(
                image_embeds=image_feats,
                text_embeds=self.prompt_feat.unsqueeze(0)
            ),
            dim=0
        )


class MultiObjective(nn.Module):
    def __init__(self, objectives: list[nn.Module], weights: list[float]):
        super().__init__()
        self.weights = weights
        self.objectives = nn.ModuleList(objectives)

    def forward(self, img: Float[Tensor, "b c h w"]) -> Float[Tensor, "b"]:
        loss = torch.zeros_like(img[:, 0, 0, 0])
        for w, o in zip(self.weights, self.objectives):
            loss = loss + w * o(img)
        return loss
