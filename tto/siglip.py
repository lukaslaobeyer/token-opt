import torch
from torch import nn
from transformers import AutoModel, AutoProcessor
from einops import einsum
from torchvision.transforms.v2.functional import normalize


class SigLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

    def encode_text(self, text):
        input_ids = self.processor.tokenizer(
            text,
            padding="max_length",
            return_tensors="pt"
        )["input_ids"].to(self.model.device)
        text_embeds = self.model.text_model(
            input_ids=input_ids,
            output_attentions=False,
            output_hidden_states=False,
        )[1]
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        return text_embeds

    def encode_img(self, img, differentiable=False):
        if differentiable:
            img_proc = normalize(
                img,
                mean=self.processor.image_processor.image_mean,
                std=self.processor.image_processor.image_std,
            )
        else:
            img_proc = self.processor.image_processor(
                img,
                return_tensors="pt",
                do_rescale=False,
            )["pixel_values"].to(img.device)
        image_embeds = self.model.vision_model(
            pixel_values=img_proc,
            output_attentions=False,
            output_hidden_states=False,
        )[1]
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        return image_embeds

    def similarity(self, img=None, text=None, image_embeds=None, text_embeds=None):
        if image_embeds is None:
            assert img is not None
            image_embeds = self.encode_img(img)
        else:
            assert image_embeds is not None

        if text_embeds is None:
            assert text is not None
            text_embeds = self.encode_text(text)
        else:
            assert text_embeds is not None

        return (
            einsum(text_embeds, image_embeds.mT, "... n d, ... d n -> ... n")
            * self.model.logit_scale.exp()
            + self.model.logit_bias
        )
