# USD
Optimized View and Geometry Distillation from Multi-view Diffuser
## [Paper](https://arxiv.org/pdf/2312.06198.pdf) | [Project page](https://youjiazhang.github.io/USD/)

<!-- ![]() -->
<img src="assets/teaser.jpg" width="50%" height="50%">

Our technique produces multi-view images and geometries that are comparable, sometimes superior particularly for irregular camera poses, when benchmarked against concurrent methodologies such as SyncDreamer and Wonder3D, without training on large-scale data.

## Denoising with unconditional noise

<img src="assets/unconditional_noise.gif" width="50%" height="50%">

The unconditional noise predicted by Zero-1-to-3 model tends to be biased. The right subfigure shows the averaged difference between the predicted noise and the added noise. We take the unconditional noise predicted by Zero-1-to-3 to remove noise from the noisy input and recover the original image. We can see that even though a very low level of noise has been added, the denoised result deviates from the original image largely. In contrast, if we use the unconditional noise predicted by Stable Diffusion for denoising, only subtle details change while the main structure and identity of ‘Mario’ are preserved.

## Unbiased Score Distillation
<!-- ![](assets/main_idea.jpg){:height="10%"} -->
<img src="assets/main_idea.jpg" width="50%" height="50%">

## Quick Start
```
# USD image-to-3D 
python launch.py --config configs/usd-patch.yaml --train --gpu 0

# SDS Loss (lambda=0)
python launch.py --config configs/usd-text-to-3D-patch.yaml --train --gpu 0 system.prompt_processor.prompt="A model of a house in Tudor style"
```
