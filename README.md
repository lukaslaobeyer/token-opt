# Highly Compressed Tokenzier Can Generate Without Training

Tokenizers with highly compressed latent spaces -- such as
[TiTok](https://arxiv.org/abs/2406.07550v1), which compresses 256x256
px images into just 32 discrete tokens -- can be used to perform
various image generative tasks **without training a dedicated
generative model at all**. In particular, we show that simple
test-time optimization of tokens according to arbitrary user-specified
objective functions can be used for tasks such as text-guided editing
or inpainting.

![CLIP-guided optimization](./assets/clip_opt.gif)
![Inpainting](./assets/inpaint.gif)

This repo includes the simple test-time optimization algorithm used in
our ICML 2025 paper, "Highly Compressed Tokenzier Can Generate Without
Training", under the `tto/` directory.

For convenience, we include the TiTok implementation copied from the
[official code release](https://github.com/bytedance/1d-tokenizer)
under `titok/`.

## Examples

 * **Text-guided image editing:**
   [`notebooks/clip_opt.ipynb`](./notebooks/clip_opt.ipynb) for
   test-time optimization with CLIP objective.

   <a target="_blank" href="https://colab.research.google.com/github/lukaslaobeyer/token-opt/blob/master/notebooks/clip_opt.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>

 * **Inpainting:**
   [`notebooks/inpainting.ipynb`](./notebooks/inpainting.ipynb) for
   inpainting via reconstruction objective.

   <a target="_blank" href="https://colab.research.google.com/github/lukaslaobeyer/token-opt/blob/master/notebooks/inpainting.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>

**Running Locally:** If you use [Nix](https://nixos.org), you can enter a
shell with all dependencies via `nix develop`.
