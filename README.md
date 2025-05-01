# Deep Learning Final Project

This repository contains our team's final project for the Deep Learning course.

## Team Members

- Abhilasha Singh
- Pranjal Wakpaijan
- Shabnam Rafat Ummey Sawda

## Project Overview

This project explores the application of deep learning techniques to solve complex problems in [brief description of your project focus area]. We implement and evaluate various neural network architectures to [main objective of your project].

## Repository Structure

```
DeepLearning_FinalProject/
├── .idea/                           # IDE configuration files
├── App/                             # Application implementation
├── Final-Group-Presentation/        # Presentation materials for the final project
├── Final-Group-Project-Report/      # Final project report documents
├── Group-Proposal/                  # Initial project proposal materials
├── Vision_transformer/              # An attempted script, not used later
├── dataset-loading/                 # Scripts for data loading and processing datasets
│   └── (data loader script Data_Preprocessing.py has only been used finally)
├── train/                           # Model training implementation
│   └── (Updated Base model, Architecture 1 and 2 implementation)
├── train_vit/                       # Vision transformer training
│   └── (Segmentation Transfromer implementation)
├── utils/                           # Utility functions and helper scripts
    
```



# Download the Data files

Source:
https://www.a2d2.audi/a2d2/en/download.html

curl -O https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/camera_lidar_semantic.tar
curl -O https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/camera_lidar_semantic_instance.tar
curl -O https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/README-SemSeg.txt


# Extract the tar file
# The tar command handles extraction of .tar files
tar -xvf camera_lidar_semantic.tar
tar -xvf camera_lidar_semantic_instance.tar


## Notes:

- Data_Preprocessing.py : A custom data-loader created from scratch to generate map from raw images and mask image. It needs to be executed at first.
- utils_1.py : Imported at the beginning of "train_model.py", "train_model_updated_architecture.py", "train_model_updated_architecture2.py"



