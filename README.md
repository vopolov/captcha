# captcha
## Description
Convolutional network for reading short captcha phrases. Written in PyTorch with Torchvision.
In current state all inputs supposed to be 3-channel RGB with sizes 330x150. Inner frame containing label supposed to be 300x120 starting at (x, y) = (8, 10).\
This size constaints are imposed by training dataset and model architecture. See sample.png for reference.
## Contents
### data.py
#### contains CaptchaDataset class compatible with torch.utils.data.DataLoader
### data_notes.ipynb
#### various notes about training set, also contains mean and bounding box calculations
### inference.py
#### calculates per sample accuracy on test dataset provided by positional argument DATA_DIR
This script uses CaptchaFixedModel.\
Optionally you can pass --weights path to model weights and save your results to --save_preds path to save file.\
Typical usage: "python inference.py data_dir --weights weights.pt --device cpu --save_preds results.json"\
For all available options call inference.py -h
### model.py
#### Contains CaptchaModel and CaptchaFixedModel
All models use randomly initialized ResNet18 as backbone and differ at feature processing step.\
CaptchaModel can work with arbitrary length phrases and don't have image width restriction. This model outputs log probabilities for letters at each scan line position in shape of:
[predictions_length >= target_length, batch_size, log_probs for each letter from dataset vocabulary + blank character for ctc loss]\
Model supposed to be optimized with torch.nn.CTCLoss.\
Currently this model hasn't been successfully optimized and may be bugged.\
CaptchaFixedModel can only predict fixed number of letters and requires inputs of a fixed label length and image width.\
It's outputs are the same log probabilities and supposed to be passed to torch.nn.NLLLoss.\
Outputs have shape of [batch_size, log_probabilities, label_size].
### train.py
#### Used to train both models, can save checkpoints and store data for Tensorboard visualization and monitoring
### util.py
#### Contains only ctc_naive_decoder to decode outputs from CaptchaModel
## Running with Docker
Docker image can be built with: "docker build -t captcha-docker ."\
Docker build requires weights.pt file to be included in the project's root directory. This file must contain weights for trained CaptchaFixedModel.\
After building container can be run with: "docker run -v DATA_DIR:/captcha/test captcha-docker test --device cpu" on cpu or\
"nvidia-docker run -v DATA_DIR:/captcha/test captcha-docker test --device cuda" on gpu if you have the nvidia-docker extension installed.\
DATA_DIR is full path to the test directory.

