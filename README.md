# Vision-and-Trajectory Detection (ViTD): A New Paradigm for Detecting Highway Abandoned Objects
 
 <p align="center">
     <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
 </p>
 
- [Vision-and-Trajectory Detection (ViTD): A New Paradigm for Detecting Highway Abandoned Objects](#vision-and-trajectory-detection-vitd-a-new-paradigm-for-detecting-highway-abandoned-objects)
  - [💬 Task: ViTD](#-task-vitd)
  - [🗃️ Dataset: ViTD-Highway](#️-dataset-vitd-highway)
    - [Data Preparation](#data-preparation)
  - [💡 Architecture: ViTD-FL](#-architecture-vitd-fl)
  - [🔍 Results](#-results)
  - [👁️ Visualization](#️-visualization)
  - [🏷️ Requirement](#️-requirement)
  - [📄 Preparation](#-preparation)
    - [Training and Evaluation](#training-and-evaluation)
    - [Contribution Guidelines](#contribution-guidelines)
  - [✨ Acknowledgments](#-acknowledgments)
 
 <p align="center">
     <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
 </p>
 
 ## 💬 Task: ViTD
 
 The ViTD task focuses on monocular image-based abandoned objects detection guided by vehicle trajectory analysis, fusing visual features and spatio-temporal trajectory information to enable more robust detection decisions in dynamic traffic scenarios. By jointly modeling visual and trajectory modalities, this task advances beyond conventional visual-centric detection approaches towards holistic traffic scene understanding required by ITS. To the best of our knowledge, we are the first to propose and implement vision-and-trajectory fusion for AOD, establishing a new paradigm that unifies static image perception with dynamic traffic scene understanding.
 
 <div align="center">
   <img src="images/ViTD.png"/>
 </div>
 
 ## 🗃️ Dataset: ViTD-Highway
 To facilitate research on this newly introduced task, we construct and release a dedicated dataset for highway abandoned objects detection, called ViTD-Highway. Sourced from real-world surveillance footage across multiple Chinese highways, the dataset comprises the final frames of 6,435 video sequences paired with spatio-temporal trajectory maps. These maps, generated via our custom trajectory rendering script and professionally calibrated, provide precise spatio-temporal context relevant to abandoned objects locations in the final video frames. 
 
 Download our dataset. We build the first dataset for ViTD, termed ViTD-Highway, which can be downloaded from our [Web](https://1drv.ms/u/c/c251f33a7b0ec098/EcNmOlVLxN9JmdH2gKoHQDsBLv7R0EYx5SI37JFr18bFyg?e=lCkOac). The download link is available below:
 ```
 https://1drv.ms/u/c/c251f33a7b0ec098/EcNmOlVLxN9JmdH2gKoHQDsBLv7R0EYx5SI37JFr18bFyg?e=lCkOac
 ```
 
 <div align="center">
   <img src="images/ViTD-Highway.png"/>
 </div>
 
 
### Data Preparation
Our dataset is organized in the COCO format. Ensure your dataset follows this structure:
```
coco
 ├── Final_Frame
 │   ├── XXXXX.jpg
 │   ├── XXXXX.jpg
 │   └── ...
 └── Trajectory
     ├── XXXXX.jpg
     ├── XXXXX.jpg
     └── ...
```

 ## 💡 Architecture: ViTD-FL
 
 <div align="center">
   <img src="images/ViTD-FL.png"/>
 </div>
 The Vision-and-Trajectory Feature Preprocessing extracts initial visual and trajectory features using dedicated backbones, FPN, RPN, and RoI Pooling. The Integration Extractor leverages a Linked Memory Token Turing Machine (LMTTM) and multi-scale alignment module to fuse the multi-modal features progressively. The ViTD Encoder performs standard detection tasks, including bounding box regression and object classification. A parallel Contrastive Head enhances feature discriminability by learning compact proposal embeddings through contrastive learning, optimized  with the Contrastive Proposal Encoding (CPE) loss.
 
 ## 🔍 Results
 
 Comparison ViTD-FL with baselines on ViTD-Highway Dataset.
 <div align="center">
   <img src="images/Experiments.jpg"/>
 </div>
 
 ## 👁️ Visualization
 
 Visualizations from our ViTD-FL on our ViTD-Highway Dataset.
 
 <div align="center">
   <img src="images/ViTD-ksh1.png"/>
 </div>
 In the figure, the red bounding boxes represent ground truth boxes,the green bounding boxes represent predicted boxes, and the yellow regions are zoomed-in views of Abandoned Objects.Scene 1 contains multiple Abandoned Objects, Scene 2 represents a low-visibility scenario, while Scene 3 and Scene 4 are normal scenes.
 
 <div align="center">
   <img src="images/ViTD-ksh2.png"/>
 </div>
  Feature heatmap corresponding to the Abandoned
 Object image region.
 
 
 ## 🏷️ Requirement
 
 | Name  |Version   |
 | ------------ | ------------ |
 |  Python | 3.9  |
 |  CUDA | >=11.1  |
 |  Pytorch | 1.12.1  |
 
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Build and install the project:
```bash
python setup.py build develop
```
 
 ## 📄 Preparation
 
 Clone the repository
 ```shell
 git clone <repository url>
 ```

### Training and Evaluation
1. Training Script
To start the training process, use the following command:
```bash
python tools/trainnet.py --num-gpus 8 --config-file <CONFIGPATH> --opts MODEL.WEIGHTS <WEIGHTSPATH>
```
Parameters:
- `--config-file`: Specify the path to the configuration file.
- `--opts`: Optional parameters to override settings in the configuration file. For example, specify the path to the model weights.
2. Testing Script
To perform testing and evaluation, use the following command:
```bash
python tools/testnet.py --num-gpus 8 --config-file <CONFIGPATH> --eval-only
```
Parameters:
- `--num-gpus`: Specify the number of GPUs to use.
- `--config-file`: Specify the path to the configuration file.
- `--eval-only`: Specify to only perform evaluation, not training.

### Contribution Guidelines
If you wish to contribute to this project, please follow these guidelines:
1. Fork the project to your GitHub account.
2. Create a new branch for development.
3. Submit a Pull Request and describe your changes.
   
## ✨ Acknowledgments
Thank all individuals and organizations that have contributed to this project.

