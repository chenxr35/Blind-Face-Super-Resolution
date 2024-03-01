## Decription of Files

srresnet_ffhq_300k.py: Config file of SRResNet model

sr_inference.py: Used to generate inferences with the trained model

20230421_034944.log: Training log file

20230421_034944.log: Training log data

loss.png: Training loss curve

psnr.png: PSNR on the validation set

AI6126_Project_2.pdf: Report of final solution

## Third-party Libraries

Pytorch, MMEngine, MMCV

## Usage

# Training

Step 1: Install Pytorch

Step 2: Install MMCV by
> pip3 install openmim
> mim install 'mmcv>=2.0.0'

Step 3: Install MMEditing by
> git clone https://github.com/open-mmlab/mmediting.git
> cd mmediting
> pip3 install -e .

Step 4: Download the given dataset(train, val, test) and place it in the "mmediting/data" folder

Step 5: Place "srresnet_ffhq_300k.py" in the "mmediting/configs" folder

Step 6: Train the model by
> python3 tools/train.py configs/srresnet_ffhq_300k.py

# Inference

Step 1: Place "sr_inference.py" in the "mmediting/tools" folder

Step 2: Place "best_PSNR_iter_290000.pth" in the "mmediting/work_dirs/srresnet_ffhq_300k_L2" folder

Step 3: Predict HQ images on the 400 test images by
> python3 tools/sr_inference.py configs/srresnet_ffhq_300k.py work_dirs/srresnet_ffhq_300k_L2/best_PSNR_iter_290000.pth data/test/LQ outputs/srresnet_ffhq_300k_L2/test

The predicted HQ images can now be found in the "mmediting/outputs/srresnet_ffhq_300k_L2/test" folder