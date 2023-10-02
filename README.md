# Customer Churn Prediction using Machine Learning

This repository demonstrates the process of predicting customer churn in a telecommunications company using machine learning techniques. We'll walk through the steps of importing necessary libraries, loading the dataset, data preprocessing, model training and evaluation, and visualization.

## Introduction

Customer churn is a critical concern for businesses, including telecommunications companies like Sprint. Predicting which customers are likely to leave (churn) can help companies take proactive measures to retain customers and reduce revenue loss. In this project, we'll use a machine learning model to predict customer churn based on various features.

### Prerequisites

Before running the code in this repository, make sure you have the following prerequisites installed:

 - `Python` (3.x recommended)
 - `Jupyter Notebook` (for code execution)
 - Required Python libraries (install using `pip` or `conda`):
   - `pandas`
   - `numpy`
   - `matplotlib`
   - `scikit-learn`

You can install the required libraries using the following command:

``````py
pip install pandas numpy matplotlib scikit-learn
``````

Alternatively:
``````py
pip install pyforest # it has multiple data science libraries i.e pandas, numpy, matplotlib, e.t.c. 
``````

These libraries provide tools for data manipulation, visualization, model creation, and evaluation.

## Usage

To use this code for customer churn prediction, follow these steps:

 1.	Clone this repository to your local machine.
 2.	Ensure you have met the prerequisites mentioned above.
 3.	Open a terminal and navigate to the project directory.
 4.	Start Jupyter Notebook 
 5.	Open the Jupyter Notebook file named `customer_churn.ipynb`.
 6.	Execute each cell in the notebook in order to load the dataset, preprocess the data, train a Random Forest Classifier, make predictions, and evaluate the model.

## Process
The code in the Jupyter Notebook follows these main steps:

 + Import Necessary Libraries: Import required Python libraries for data analysis and machine learning.
 + Load the Dataset: Load the customer churn dataset from a CSV file using pandas.
 + Data Preprocessing: Select relevant features for churn prediction, split the data into training and testing sets, and standardize features.
 + Train a Machine Learning Model: Train a Random Forest Classifier to predict customer churn. You can choose a different model based on your requirements.
 + Make Predictions: Use the trained model to make predictions on the test data.
 + Evaluate the Model: Calculate accuracy, generate a confusion matrix, and create a classification report to assess the model's performance.
 + Visualize Feature Importances: Visualize the importance of each feature in predicting churn using a bar chart.

## Results

The results of the churn prediction model, including accuracy, confusion matrix, and classification report, will be displayed in the Jupyter Notebook. Additionally, a bar chart will show the feature importances.


