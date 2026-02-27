This is a fork of PyTorch implementation of author paper at ICCV2021:

**Knowledge Enriched Distributional Model Inversion Attacks** \[[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Knowledge-Enriched_Distributional_Model_Inversion_Attacks_ICCV_2021_paper.pdf)\]  \[[arxiv](https://arxiv.org/abs/2010.04092)\]

Authors propose a novel **'Inversion-Specific GAN'** that can better distill knowledge useful for performing attacks on private models from public data. Moreover,  they propose to *model a private data distribution* for each target class which refers to **'Distributional Recovery'**.

## Experiment Goal
* Test different black-box model security measures to prevent private training data recovery.

## Getting Started
* execute `prepare_for_experiment.sh` to download CelebA dataset and necessary python libraries.
* Download pre-trained FaceNet_95.88.tar evaluator model parameters from: https://drive.google.com/drive/folders/1U4gekn72UX_n1pHdm9GQUQwwYVDvpTfN. Place the file inside ./target_model/target_ckp


## Replicate the experiment
* execute `run_experiment.sh`. It runs the Model Inversion attack multiple times, each one utilzing a different security strategy.
* `run_experiment.sh` produces results under `protection_results/` folder.
* visualize results by running `plot_recovery_results.ipynb`.
