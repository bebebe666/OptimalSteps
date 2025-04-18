# Optimal Stepsize for Diffusion Sampling(OSS)

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2503.18948-b31b1b.svg)](https://arxiv.org/abs/2503.21774)&nbsp;

Official PyTorch implementation for our paper **[Optimal Stepsize for Diffusion Sampling](https://arxiv.org/abs/2503.21774)**, a plug-and-play algorithm to search the optimal sampling stepsize in diffusion sampling.

  <p align="center">
    <img src="./teaser.png" alt="teaser" width="820px">
    <br/>
    <em>Left to Right: FLUX-100steps; FLUX+OSS-10steps; FLUX-10steps.</em>
  </p>




https://github.com/user-attachments/assets/d903c467-7050-4e79-9659-47bb672f001c
<p align="center">
    <em>Left to Right: Wan-100steps; Wan+OSS-20steps; Wan-20steps.</em>
</p>

## :fire: Latest News!!

* Apr 13, 2025: :clap: OSS is integrated into [ComfyUI](https://github.com/comfyanonymous/ComfyUI) as OptimalStepsScheduler! It supports the fast sampling of text-to-image model FLUX and text-to-video model Wan. Here, we give a comparision of FLUX workflow with 10 sampling steps:
![pr](https://github.com/user-attachments/assets/3cd26404-3873-40a0-9773-3bddcdce6cb1)
Here are the demo workflows for [FLUX-workflow](https://github.com/user-attachments/files/19720395/Flux-Dev-ComfyUI-Workflow.json)  and [Wan-workflow](https://github.com/user-attachments/files/19720398/text_to_video_wan.json). 10 steps for FLUX and 20 steps for Wan are highly recommended. More details are included in our pull request [here](https://github.com/comfyanonymous/ComfyUI/pull/7584).




## :smiley: Overview
In this repo, we provide some examples of using our algorithm based on the [DiT](https://github.com/facebookresearch/DiT), [FLUX](https://github.com/black-forest-labs/flux), [Open-Sora](https://github.com/hpcaitech/Open-Sora/tree/opensora/v1.2), and [Wan2.1](https://github.com/Wan-Video/Wan2.1). 

Note that OSS is not limited to these, we also provide a guidance to adapt it to other diffusion models.

## :rocket: Quick Start Guide

### Step 0: Prepare the environment
Prepare the environment for the target model you want to use, such as [DiT](https://github.com/facebookresearch/DiT), [FLUX](https://github.com/black-forest-labs/flux), [Open-Sora](https://github.com/hpcaitech/Open-Sora/tree/opensora/v1.2), [Wan2.1](https://github.com/Wan-Video/Wan2.1) or other models.


### Step 1: Clone the repository
```bash
git clone https://github.com/bebebe666/OSS.git
cd OSS
```

### Step 2: Run inference
```bash
# DiT 
bash scripts/dit.sh
# FLUX
bash scripts/flux.sh
# Open-Sora
bash scripts/opensora.sh
# Wan2.1
bash scripts/wan.sh

```

## :airplane: Appling for other models
### Model-Preparing
Before using our algorithm, you need to wrap your diffusion model to a unified format, this should satisfies:
- The output of the model should be the v-pred same as the Flow Matching.
- The sampling trajectory should be straight, follows $x_j = x_i + v(t_j - t_i)$.

You can refer to examples we gave in the `model_wrap.py`.

### Searching
We provide the searching functions as follows:
```
oss_steps = search_OSS(model, z, search_batch, context, device, teacher_steps=10, student_steps=5, model_kwargs=model_kwargs)
```
 - the `z` is the input noise;
 - the `search_batch` is the number of images you want to search;
 - the `context` is the class embedding in class conditional image generation and prompt embedding in text-to-image generation.

We provide the function `search_OSS_video` for video generation model searching, which supports the selection of frame and channel. As default, `cost_type="all"` and `channel_type="all"` means using all the frames and channels for cost calculation. You can pass any number (remember not greater than the total) as a string format like `"4"` to use part of them.



### Inference
After getting the `oss_steps`, you can pass it to the inference function to get the sampling results.
```
samples = infer_OSS(oss_steps, model, z, context, device, model_kwargs=model_kwargs)
```


## 🙏 Acknowledgments

This codebase benefits from the solid prior works: [DiT](https://github.com/facebookresearch/DiT), [FLUX](https://github.com/black-forest-labs/flux), [Open-Sora](https://github.com/hpcaitech/Open-Sora/tree/opensora/v1.2), and [Wan2.1](https://github.com/Wan-Video/Wan2.1) for their excellent generation ability.

---
## 📖 Citation

If you find this project helpful for your research or use it in your own work, please cite our paper:
```bibtex
@misc{pei2025optimalstepsizediffusionsampling,
      title={Optimal Stepsize for Diffusion Sampling}, 
      author={Jianning Pei and Han Hu and Shuyang Gu},
      year={2025},
      eprint={2503.21774},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.21774}, 
}
```

---


⭐️ If this repository helped your research, please star 🌟 this repo 👍!
