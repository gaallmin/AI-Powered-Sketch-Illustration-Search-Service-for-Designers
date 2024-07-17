# AI-Powered-Sketch-Illustration-Search-Service-for-Designers
<img src="https://github.com/gaallmin/AI-Powered-Sketch-Illustration-Search-Service-for-Designers/blob/main/prize.jpg" width="300"/>

## Introduction 
This project, which secured second place in a national data hackathon organized by the Ministry of Science and ICT of Korea, aims to develop an AI service for efficiently searching sketch illustrations, supporting designers in the growing design industry. We created an effective classifier for illustrations drawn using computer tools.

## Project Overview

### Mission #1: Exploratory Data Analysis (EDA)
We conducted a thorough EDA to identify data imbalances and variances in image formats and resolutions. Using bar plots, we visualized class distributions and resized images to a uniform 3x224x224 resolution. Random samples from each class were extracted to analyze image characteristics, leading to the decision to remove real images for better classification accuracy.

### Mission #2: Data Cleansing
Utilizing DBSCAN and Agglomerative Clustering, we cleansed the data by identifying and removing noise (real images). Agglomerative Clustering with K=2 provided the best results, separating images into real and other types, such as sketches and illustrations. We then split the dataset into training, testing, and validation sets in a 6:2:2 ratio.

### Mission #3: Hyperparameter Tuning and Model Selection
We tested various models, including VGG 16 and ResNet 18, and implemented focal loss to handle data imbalance. VGG 16 outperformed other models and was further optimized to VGG 16_V2. This final model, trained with augmented datasets, demonstrated superior classification performance.

This project effectively addresses the challenges of data imbalance and diverse image formats, resulting in a robust classifier for sketch illustration search services.


