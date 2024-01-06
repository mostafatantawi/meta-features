# Meta Features Analysis

This Visual Studio Code project performs the Meta feature analysis on a given dataset, focusing on meta-features such as skewness, kurtosis, correlation, covariance, and mutual information. It utilizes the scikit-learn and pandas libraries for data manipulation and feature selection.

## Prerequisites

Make sure you have the necessary Python packages installed. You can install them using the following:

```bash
pip install pandas scikit-learn numpy
```

## Description
The script fetches a dataset from OpenML (credit-g dataset) and identifies the target variable. It then extracts various meta-features, including skewness, kurtosis, correlation matrix, covariance matrix, and mutual information. The calculated features are printed to the console.

