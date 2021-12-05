# Introduction
With the rise of population and ever-growing megacities, it is essential for the authorities to monitor the crowd movements and behaviour. In the current context of global pandemics, crowd monitoring in order to maintain social distancing is becoming extremely important. Using surveillance cameras for this is not feasible because of the large area that needs to be covered. Using Unmanned Aerial Vehicles (UAVs) for aerial surveillance is one solution to this problem. However, using Unmanned Aerial Vehicles has its own challenge because of the area covered is large. This can be somewhat overcome by using higher resolution images, but this will result in longer processing times and higher computing resources. There are several other challenges such as viewpoint and scale variations and background clutter in the UAV based crowd monitoring. Crowd monitoring can be categorised into three areas. They are, Crowd Counting, Crowd Localization and Crowd Behaviour Detection. In this thesis we focus on crowd behaviour monitoring in the health and safety and social distancing aspects. The motivation for this research came from the recent Coronavirus Disease 2019 (COVID-19) pandemic and the need of enforcing social distancing. We have selected two aspects of enforcing social distancing for this project. They are Enforcing isolation and lock down in certain areas and Detecting highly dense crowd in public areas. In order to the achieve first task we use a human action detection and people detection method based of You Only Look Once (YOLO) deep neural network. To achieve the second, we used the deep neural network SFANet with a modified loss function called Bayesian Loss which was proposed in literature. Our proposed model for human action detection and people detection outperformed the existing models with Mean Average Precision of 20.10% for the Okutama-Action dataset. Our proposed model for detecting crowd density outperformed the existing models with a mean squared error of 151.3 on the UCF-QNRF dataset.

# GitHub URL https://github.com/akilaw/crowd-behaviour
# Instructions to Setup the Project
## System Requirements 
- In order to run the project the system should have at least a 4 core CPU, 16 GB memory and a Nvidia GPU with Kepler or better architecture with 4GB or larger GPU memory.
Operating system needed to run the project is Ubuntu 20.04 LTS
## Installing Prerequisite Software
### Installing Miniconda
- `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
- `chmod 755 Miniconda3-latest-Linux-x86_64.sh`

### Install CUDA
- `wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin`
- `sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600`
- `sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub`
- `sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"`
- `sudo apt-get update`
- `sudo apt-get -y install cuda`
- `sudo apt-get install nvidia-driver-460`
- Reboot the system
### Install Pytorch & Torch Vision
- conda activate
- conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

## Cloning the Repository and Setting Up the Project
- Clone the repository at https://github.com/akilaw/crowd-behaviour
### Setting up Crowd Density Prediction
- Download the UCF-QNRF Dataset
    - `wget --no-check-certificate https://www.crcv.ucf.edu/data/ucf-qnrf/UCF-QNRF_ECCV18.zip`
- Move the UCF-QNRF_ECCV18.zip to crowd-behaviour/density/data and extract
    - `mv UCF-QNRF_ECCV18.zip crowd-behaviour/density/data`
    - `unzip UCF-QNRF_ECCV18.zip`
- Preprocess the files
    - `cd crowd-behaviour/density/`
    - `cnn/python preprocess_dataset.py --origin_dir data/UCF-QNRF_ECCV18 --data_dir data/UCF-QNRF_ECCV18_PROCESSED`
- Train on Data
    - `python cnn/train.py --data-dir data/UCF-QNRF_ECCV18_PROCESSED --save-dir cnn/output`  
- Validation and Testing
    - `python val.py`
    - `python test.py --data-dir data/UCF-QNRF_ECCV18_PROCESSED --save-dir cnn/output`
### Setting up Crowd Behaviour Detection
- Downloading and Copying Data
    - `Download the dataset from http://okutama-action.org/`
    - `Copy the train and test set to isolation/data/train/ and isolation/data/test`
- Preprocessing Data
    - `cd crowd-behaviour/isolation/utils`
    - `python frames.py`
    - `python annotations.py`
- Training the Model
    - `cd crowd-behaviour/isolation/yolov5`
    - `python train.py --img 640 --batch 16 --epochs 20 --data crowd.yaml --weights yolov5m.pt`
- Validation and Testing the Model
    - `cd crowd-behaviour/isolation/yolov5`
    - `python val.py --img 640 --data crowd.yaml --weights runs/train/exp6/weights/best.pt`
