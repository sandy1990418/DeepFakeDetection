# Towards DeepFake video forensics based on facial textural disparities in multi-color channels

## Overview
The primary objective of this project is to replicate, validate the effectiveness and reliability of the DeepFake detection methodology proposed by Xia et al. (2022). in the domain of DeepFake video forensics.

<br>

## Environments

Using CUDA11.3, PyTorch 1.13.1, python3.9

```bash
pip install -r requirements.txt
```

<br>

## Steps
To reproduce the findings outlined in the aforementioned paper, the following steps are undertaken:


1. **Data Sampling**: Employ simple random sampling technique to extract 30 images from each video dataset. Utilize system sampling within this repository to ensure consistency and reproducibility.

2. **Image Preprocessing**: Utilize advanced image processing algorithms to partition each image into $256 \times 256$ blocks. Due to hardware constraints, resize the image dimensions to $128 \times 128$ while preserving critical information.

3. **Feature Engineering**: Compute the first-order differential of each image block and subsequently perform Min-Max normalization to standardize the resulting values, ensuring consistent feature scaling across the dataset.

4. **Feature Transformation**: Derive the feature difference $F_{ij}$ by subtracting the function values of adjacent blocks $(f(b_{ij}) - f(b_{ij+1}))$. Round off the result to the nearest integer to maintain numerical stability.

5. **Thresholding**: Establish a threshold value $T (=2)$. If the calculated feature difference $F_{ij}$ exceeds this threshold, set it to 2; conversely, if it falls below -2, set it to -2. This step helps mitigate noise and enhance feature discriminability.

6. **Co-occurrence Matrix Calculation**: Compute the co-occurrence matrix incorporating rotations at 0°, 90°, 180°, and 270° using both 1-step and 2-step combinations. This step captures spatial relationships between image pixels and enhances feature representation.

7. **Feature Selection**: Focus solely on computing the co-occurrence matrix for Red (R), Green (G), Blue (B), Value (V), and Luminance (Y) channels to obtain a total of 375 variables. This selective feature extraction strategy ensures the inclusion of informative features while reducing computational complexity.

8. **Model Training and Evaluation**: Utilize Support Vector Machine (SVM) with the extracted 375 variables as inputs. Additionally, leverage Logistic Regression, Naive Bayes, Kernel SVM, and Random Forest classifiers for comprehensive model evaluation. This step assesses the performance of the proposed methodology across diverse classification frameworks, ensuring robustness and generalizability.

<br>

## Usage
```bash
python src/main.py \
       --file_path YOUR_DATA_FILE_PATH\
       --storage_dir WHERE_YOU_WANT_TO_STORE_EXTRACT_RESULT

```

<br>

## Data Source
One of dataset in reproduced paper is [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics). You can get more detailed information in link.

<br>


## Experiment Result
The table below presents the results of the conducted experiments:

|Method             | Block Number  | IF Background  | Logistic Regression ACC | Naive Bayes ACC | Kernel SVM ACC |Random Forest ACC |
|-------------------|:-------------:|:-------------:|:-----------------------:|:----------------:|:----------------:|:----------------:|
| Xia et al. (2022) | $128\times128$| Extract Face  | 0.66                    |0.60             |0.69|0.81|
| Xia et al. (2022) | $128\times128$| Keep Background  | 0.57                    |0.54             |0.60|0.80|

***Note:*** `IF Background` indicates whether the background should be retained or removed if extracting a face from an image.


<br>

## Reference 
[1] Xia, Z., Qiao, T., Xu, M., Zheng, N., Xie, S. (2022). Towards DeepFake video forensics based on facial textural disparities in multi-color channels. Information Sciences, 607, 654–669.
