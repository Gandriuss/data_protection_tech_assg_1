## This is a fork of PyTorch implementation of author paper at ICCV2021:

**Knowledge Enriched Distributional Model Inversion (MI) Attacks** \[[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Knowledge-Enriched_Distributional_Model_Inversion_Attacks_ICCV_2021_paper.pdf)\]  \[[arxiv](https://arxiv.org/abs/2010.04092)\]

Authors propose a novel **'Inversion-Specific GAN'** that can better distill knowledge useful for performing attacks on private models from public data. Moreover, they propose to *model a private data distribution* for each target class which refers to **'Distributional Recovery'**.



## Experiment Goal
* Test multiple 'black-box' ML model security measures to prevent private training data recovery.

## Getting Started
0. ensure you have an NVIDIA GPU and cuda installed. Experiment was performed with:
  - python 3.10 
  - cuda 11
  - torch 2.1.0
1. execute `prepare_for_experiment.sh` - installs necessary python libraries.
2. Download pre-trained `FaceNet_95.88.tar` evaluator model parameters from: `https://drive.google.com/drive/folders/1U4gekn72UX_n1pHdm9GQUQwwYVDvpTfN`. Place the .tar file inside `./target_model/target_ckp`

## Replicate the experiment
1. execute `run_experiment.sh`:
  - runs multiple Model Inversion attacks, each one utilzing a different victim security measure (ex: flatten, fixed_top5).
  - produces results under `./protection_results/`.
2. visualize attack results by executing `plot_recovery_results.ipynb`.

## Modifications to the original author repository:
* added author pre-trained victim and attack model parameter files.
* 2 new bash scripts(`prepare_for_experiment.sh`, `run_experiment.sh`) for experiment replication.
* `revovery.py` (added parameters which set victim model's safety measures).
* `attack.py` (updated function `dist_inversion()` - simulates different victim's safety features during MI attack).
* added `csv_logging.py` for persistent logging.
* experimental results, used in report's experiment, are saved under `./report_protection_results/`.
* added `plot_recovery_results.ipynb`.
