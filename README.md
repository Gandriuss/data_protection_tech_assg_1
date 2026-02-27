This is a fork of PyTorch implementation of author paper at ICCV2021:

**Knowledge Enriched Distributional Model Inversion Attacks** \[[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Knowledge-Enriched_Distributional_Model_Inversion_Attacks_ICCV_2021_paper.pdf)\]  \[[arxiv](https://arxiv.org/abs/2010.04092)\]

Authors propose a novel **'Inversion-Specific GAN'** that can better distill knowledge useful for performing attacks on private models from public data. Moreover,  they propose to *model a private data distribution* for each target class which refers to **'Distributional Recovery'**.

## Forked Repository Experiment Goal
* Test different black-box model security measures to prevent private training data recovery.

## Getting Started
* execute `prepare_for_experiment.sh` to download CelebA dataset and necessary python libraries.
* Download pretrained binary GANs, victim & evaluator models from: https://drive.google.com/drive/folders/1L3frX-CE4j36pe5vVWuy3SgKGS9kkA70?usp=sharing.


## Run the experiment
* execute `run_experiment.sh`
* find results under `protection_results/` folder
* visualize results by running `plot_recovery_results.ipynb`
