# Reproduce-Deepfake-paper

## Overview
The primary objective of this project is to replicate and validate the research findings of Xia et al. in the domain of DeepFake video forensics.


## Steps
To illustrate the process of reproducing the findings outlined in the aforementioned paper, I provide a detailed demonstration of the methodology employed:



Step 1: Perform simple random sampling to select 30 images from each video. (In this repo used system sampling)

Step 2: Divide each image into $256*256$ blocks, and due to equipment constraints, reduce the image size to $128*128$.

Step 3: Calculate the first-order differential of each block and perform Min-Max normalization.

Step 4: $F_{ij} = f(b_{ij}) - f(b_{ij+1})$ and round off to the nearest integer.

Step 5: Set the threshold value $T (=2)$. If $F_{ij}$ is greater than or equal to 2, it is set to 2, and if $F_{ij}$ is less than or equal to -2, it is set to -2.

Step 6: Calculate the co-occurrence matrix with 0°, 90°, 180°, and 270° rotations using 1-step and 2-step combinations.

Step 7: Only calculate the co-occurrence matrix of R, G, B, V, and Y to obtain a total of 375 variables.

Step 8: Perform SVM with the 375 variables, and use them as inputs for Logistic Regression, Naive Bayes, Kernel SVM, and Random Forest in this study.


## Data Source
One of dataset in reproduced paper is [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics). You can get more detailed information in below link: 


## Reference 
[1] Xia, Z., Qiao, T., Xu, M., Zheng, N., Xie, S. (2022). Towards DeepFake video forensics based on facial textural disparities in multi-color channels. Information Sciences, 607, 654–669.