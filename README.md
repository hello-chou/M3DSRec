# M3DSRec
# Introduction
This work is currently under review at TOIS (ACM Transactions on Information Systems).

Note: The file alltoone.py corresponds to the implementation of M3DSRec.




# Data Preparation
Please place your datasets in the dataset folder before running the training scripts.

# Training Guide

M3DSRec has two-stage process: teacher network training and student network training.

## 1. teacher network training

Run the following command to start teacher network training:

bash
python finetune.py -m alltoone -d dataset_name -props=‘props/alltoone.yaml,props/finetune.yaml’ --stage=pretrain --modelmethod=[method]


## 2. student network training

Run the following command to start student network training:

bash
python finetune.py -m alltoone -d dataset_name -props=‘props/alltoone.yaml,props/finetune.yaml’ --stage=finetune --warm_up=path/to/your_teacher_model.pth --modelmethod=[method]



# Sequence Modeling Methods
You can specify the sequence modeling method using the --modelmethod argument. The available options are:

sasrec: Corresponds to SASRec
mixer: Corresponds to MLP-Mixer
mamba: Corresponds to Mamba4Rec


**Note:**
*   `dataset_name`: Replace with the name of your dataset.
*   `path/to/your_teacher_model.pth`: Replace with the actual path to the trained teacher model weights.
*   `[method]`: Replace with `sasrec`, `mixer`, or `mamba`.


