pip install pyarrow==6.0.1
# pip install accelerate comet-ml matplotlib datasets tqdm tensorboard tensorboardX torchvision tensorflow-datasets einops pytorch-fid joblib PyYAML kaggle wandb torchsummary torchinfo
pip install -r requirements.txt

cd diffusers
pip install .
cd ..

mkdir measure
mkdir datasets
mkdir measure/CELEBA-HQ
mkdir measure/CIFAR10
mkdir datasets/celeba_hq_256