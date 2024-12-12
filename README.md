# Advanced Analytics Edge: Detecting Problematic Internet Usage

## Project Overview
This project aims to predict and detect early indicators of problematic internet usage among children and adolescents using sparse data from the Healthy Brain Network (HBN) dataset. The dataset includes physical activity levels, medical data, and survey data. The primary objective is to predict the Severity Impairment Index (`SII`), which measures the level of problematic internet usage.

## Authors
- **Andres Camarillo**
- **Shane Epstein-Petrullo**
- **Azfal Peermohammed**
- **Sabal Ranabhat**
- **Victor Zhuang**

## Key Objectives
- **Data Imputation**: Handle extremely sparse data with over 100,000 missing values.
- **Model Comparison**: Evaluate various multi-class classification models and ensemble models.
- **Predictive Metric**: Use the Quadratic Weighted Kappa (QWK) to assess model performance.

## Dataset
The HBN dataset includes:
- **Tabular Data**: Demographics, BMI, physical fitness tests, and survey responses.
- **Time Series Data**: Accelerometer data tracking movement and light exposure for 998 children.

## Challenges
- **Sparse Data**: Both tabular and time series data have a large number of missing observations.
- **Noisy Measurements**: The `SII` is subject to variability due to self-reported questionnaires.
- **Data Integration**: Combining available tabular data with time series information.

## Methodology
- **Data Preprocessing**: Imputation techniques to handle missing values.
- **Feature Extraction**: Extract relevant features from time series data.
- **Model Training**: Train and compare multiple classification models.
- **Evaluation**: Use QWK to measure the agreement between predicted and actual `SII` values.

## Dependencies
- Python 3.9.18
- Libraries: pandas, numpy, scikit-learn, hyperopt, etc.

## Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo-url.git
   cd your-repo-directory
   ```

2. **Install dependencies**:
   ```bash
    pip install -r requirements.txt
    ```

3. **Download the data**

    Please download the data into a `data` folder.

    ```bash
    kaggle competitions download -c child-mind-institute-problematic-internet-use -p /path/to/data
    ```

4. **Run the analysis**: 
Please indicate the path of the training set, as well as the name of the data used.
    ```bash
    python main.py training_sets/imputed_train_grid.csv optimal_imputation
    ```
