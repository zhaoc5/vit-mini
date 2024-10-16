conda create -n dino python=3.10
conda activate dino
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 cuda-toolkit=11.8 -c pytorch -c nvidia
pip install numpy==1.26.4 xformers==0.0.19 timm==0.9.16 torchmetrics==0.10.3 six lmdb fvcore omegaconf
