# 🧠 Face Image Age and Gender Prediction

## 📊 Overview

In this project, we explore the multi-object task of age and gender prediction \
using a convolutional neural network (CNN) on a comprehensive face image dataset. \
This dataset comprises over 20,000 face images, each annotated with age, gender, \
and ethnicity, and a wide variety of poses, facial expressions, lighting conditions, \
occlusions, and resolutions.

Our primary focus is on predicting age and gender while analyzing the associated \
biases and accuracy of these predictions. Additionally, we will discuss the \
ethical implications of the AI-generated results and the potential risks associated \
with deploying this model in real-world applications.

## 📚 Dataset

The analysis uses the [UTKFace](https://www.kaggle.com/datasets/jangedoo/utkface-new) dataset from Kaggle, which includes:

- Over 20,000 face images
- Annotations for age, gender, and ethnicity
- Various poses, facial expressions, and lighting conditions
- Different levels of occlusions and resolutions

## 📗 Notebooks

- [1_data_cleaning.ipynb](https://github.com/MeiChieh/face-image-age-and-gender-prediction/blob/main/1_data_cleaning.ipynb)
- [2_eda.ipynb](https://github.com/MeiChieh/face-image-age-and-gender-prediction/blob/main/2_eda.ipynb)
- [3_modeling.ipynb](https://github.com/MeiChieh/face-image-age-and-gender-prediction/blob/main/3_modeling.ipynb)

## 📈 Analysis Structure

### 1. Data Cleaning & Preprocessing

- Detection of broken/corrupted files
- Duplicate detection using dHash comparison
- Brightness anomaly detection
- Blurriness detection using Laplacian variance

### 2. Exploratory Data Analysis

- Basic overview
  - Sample images analysis
  - File format and dimension analysis
  - Aspect ratio distribution
- Target and feature distribution
  - Age and gender distribution
  - Race feature distribution
- Color and saturation analysis
  - Brightness and saturation patterns
  - Common color palette identification
- Structure and texture detection
- Stratified data splitting

### 3. Model Development

- Model preparation and training
- Prediction analysis
  - Baseline model comparison
  - Gender classification metrics
  - Age prediction residual analysis
- Error analysis
  - Gender classification errors
  - Age prediction errors
- Ethical implications and bias analysis

## ⭐ Key Findings

- Analysis of model performance across different demographic groups
- Identification of potential biases in age and gender prediction
- Evaluation of model accuracy in various conditions
- Insights into ethical implications of AI-based demographic prediction

## 📁 Project Structure

```
├── 1_data_cleaning.ipynb    # Data cleaning and preprocessing
├── 2_eda.ipynb             # Exploratory data analysis
├── 3_modeling.ipynb        # Model development and evaluation
└── README.md               # Project documentation
```

## 🛠️ Technologies Used

- Python
- TensorFlow/Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib/Seaborn

## 🚀 Setup and Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the UTKFace dataset from Kaggle
4. Run the notebooks in sequence:
   - Data cleaning
   - EDA
   - Modeling


## 🔄 Future Improvements

1. Implement additional data augmentation techniques
2. Explore more advanced CNN architectures
3. Add real-time prediction capabilities
4. Develop a user-friendly interface
5. Implement bias mitigation techniques

## 📦 Dependencies

Key dependencies include:

- tensorflow
- opencv-python
- numpy
- pandas
- matplotlib
- seaborn

For a complete list of dependencies, see `requirements.txt`.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Mei-Chieh Chien
