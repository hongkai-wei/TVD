
# ViTD
## Introduction
Trajectory Spatiotemporal Diagram Guided Detection of Road Debris
## CUDA Requirements
- CUDA Version: 11.1
- PyTorch Version: 1.9.1
## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Build and install the project:
```bash
python setup.py build develop
```
## Data Preparation
Our dataset is organized in the COCO format. Ensure your dataset follows this structure:
```
coco
 ├── annotations
 │   ├── val_your_dataset.json
 │   ├── train_your_dataset.json
 │   └── ...
 ├── train_part
 │   ├── XXXXX.jpg
 │   ├── XXXXX.jpg
 │   └── ...
 └── val_part
     ├── XXXXX.jpg
     ├── XXXXX.jpg
     └── ...
```
## Training and Evaluation
### Training Script
To start the training process, use the following command:
```bash
python tools/trainnet.py --num-gpus 8 --config-file <CONFIGPATH> --opts MODEL.WEIGHTS <WEIGHTSPATH>
```
Parameters:
- `--config-file`: Specify the path to the configuration file.
- `--opts`: Optional parameters to override settings in the configuration file. For example, specify the path to the model weights.
### Testing Script
To perform testing and evaluation, use the following command:
```bash
python tools/testnet.py --num-gpus 8 --config-file <CONFIGPATH> --eval-only
```
Parameters:
- `--num-gpus`: Specify the number of GPUs to use.
- `--config-file`: Specify the path to the configuration file.
- `--eval-only`: Specify to only perform evaluation, not training.

## Contribution Guidelines
If you wish to contribute to this project, please follow these guidelines:
1. Fork the project to your GitHub account.
2. Create a new branch for development.
3. Submit a Pull Request and describe your changes.
## Acknowledgments
Thank all individuals and organizations that have contributed to this project.

