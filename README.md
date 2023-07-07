# BadDiffusion
Official repo to reproduce the paper "How to Backdoor Diffusion Models?" published at CVPR 2023

Paper link: https://arxiv.org/abs/2212.05400

## Environment

- Python 3.8.5
- PyTorch 1.10.1+cu11 or 1.11.0+cu102

## Usage

### Install Require Packages and Prepare Essential Data

Please run

```bash
bash install.sh
```

### Wandb Logging Support

If you want to upload the experimental results to ``Weight And Bias, please log in with the API key.

```bash
wandb login --relogin --cloud <API Key>
```

### Prepare Dataset

- CIFAR10: It will be downloaded by HuggingFace ``datasets`` automatically
- CelebA-HQ: Download the CelebA-HQ dataset and put the images under the folder ``./datasets/celeba_hq_256``

### Run BadDiffusion

Arguments
- ``--project``: Project name for Wandb
- ``--mode``: Train or test the model, choice: 'train', 'resume', 'sampling`, 'measure', and 'train+measure'
    - ``train``: Train the model
    - ``resume``: Resume the training
    - ``measure``: Compute the FID and MSE score for the BadDiffusion from the saved checkpoint, the ground truth samples will be saved under the 'measure' folder automatically to compute the FID score.
    - ``train+measure``: Train the model and compute the FID and MSE score
    - ``sampling``: Generate clean samples and backdoor targets from a saved checkpoint
- ``--dataset``: Training dataset, choice: 'MNIST', 'CIFAR10', and 'CELEBA-HQ'
- ``--batch``: Training batch size. Note that the batch size must be able to divide 128 for 
the CIFAR10 dataset and 64 for the CelebA-HQ dataset.
- ``--sched``: Choose sampler algorithm, choice: "DDPM-SCHED", "DDIM-SCHED", "DPM_SOLVER_PP_O1-SCHED", "DPM_SOLVER_O1-SCHED", "DPM_SOLVER_PP_O2-SCHED", "DPM_SOLVER_O2-SCHED", "DPM_SOLVER_PP_O3-SCHED", "DPM_SOLVER_O3-SCHED", "UNIPC-SCHED", "PNDM-SCHED", "DEIS-SCHED", "HEUN-SCHED"
- ``--eval_max_batch``: Batch size of sampling, default: 256
- ``--epoch``: Training epoch num, default: 50
- ``--learning_rate``: Learning rate, default for 32 * 32 image: '2e-4', default for larger images: '8e-5'
- ``--poison_rate``: Poison rate
- ``--trigger``: Trigger pattern, default: 'BOX_14', choice: 'BOX_18', 'BOX_14', 'BOX_11', 'BOX_8', 'BOX_4', 'STOP_SIGN_18', 'STOP_SIGN_14', 'STOP_SIGN_11', 'STOP_SIGN_8', 'STOP_SIGN_4', 'GLASSES'
- ``--target``: Target pattern, default: 'CORNER', choice: 'TRIGGER', 'SHIFT', 'CORNER', 'SHOE', 'HAT', 'CAT'
- ``--gpu``: Specify GPU device
- ``--ckpt``: Load the HuggingFace Diffusers pre-trained models or the saved checkpoint, default: 'DDPM-CIFAR10-32', choice: 'DDPM-CIFAR10-32', 'DDPM-CELEBA-HQ-256', or user specify checkpoint path
- ``--fclip``: Force to clip in each step or not during sampling/measure, default: 'o'(without clipping)
- ``--output_dir``: Output file path, default: '.'

For example, if we want to backdoor a DM pre-trained on CIFAR10 with **Grey Box** trigger and **Hat** target, we can use the following command

```bash
python baddiffusion.py --project default --mode train+measure --dataset CIFAR10 --batch 128 --epoch 50 --poison_rate 0.1 --trigger BOX_14 --target HAT --ckpt DDPM-CIFAR10-32 --fclip o -o --gpu 0
```

If we want to generate the clean samples and backdoor targets from a backdoored DM, use the following command
Or simply generate the samples

```bash
python baddiffusion.py --project default --mode sampling --ckpt res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT --fclip o --gpu 0
```

### Run Adversarial Neuron Pruning (ANP)

Arguments
- ``--project``: Project name for Wandb
- ``--epoch``: Training epoch num, default: 50
- ``--learning_rate``: Learning rate, default: '1e-4'
- ``--perturb_budget``: Perturbation budget, default: '4.0'
- ``--gpu``: Specify GPU device
- ``--ckpt``: Load the HuggingFace Diffusers pre-trained models or the saved checkpoint
- ``--output_dir``: Output file path, default: '.'

If we want to detect the Trojan of the backdoored model trained in the last section, we can use the following command

```bash
python anp_defense.py --project default --epoch 5 --learning_rate 1e-4 --perturb_budget 4.0 --ckpt res_DDPM-CIFAR10-32_CIFAR10_ep50_c1.0_p0.1_BOX_14-HAT --gpu 0
```
