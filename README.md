## This is a fork of PyTorch implementation of author paper at ICCV2021:

**Knowledge Enriched Distributional Model Inversion Attacks** \[[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Knowledge-Enriched_Distributional_Model_Inversion_Attacks_ICCV_2021_paper.pdf)\]  \[[arxiv](https://arxiv.org/abs/2010.04092)\]

Authors propose a novel **'Inversion-Specific GAN'** that can better distill knowledge useful for performing attacks on private models from public data. Moreover,  they propose to *model a private data distribution* for each target class which refers to **'Distributional Recovery'**.

## Experiment Goal
* Test different black-box model security measures to prevent private training data recovery.

## Getting Started
* ensure you have an NVIDIA GPU and cuda installed.
* execute `prepare_for_experiment.sh` to download CelebA dataset and necessary python libraries.
* Download pre-trained FaceNet_95.88.tar evaluator model parameters from: https://drive.google.com/drive/folders/1U4gekn72UX_n1pHdm9GQUQwwYVDvpTfN. Place the file inside ./target_model/target_ckp


## Replicate the experiment
* execute `run_experiment.sh`. It runs multiple Model Inversion attacks, each one utilzing a different victim security measure (ex: flatten, fixed_top5).
* `run_experiment.sh` produces results under `protection_results/` folder.
* visualize results by running `plot_recovery_results.ipynb`.

## Modifications to original code
apart from logging and plotting functions, code has been only modified
* revovery.py (implementation of different script run modes)
* attack.py (function dist_inversion() contains modification to victim output based on the different run modes)
