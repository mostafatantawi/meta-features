import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import entropy 


def extract_meta_features(dataset):
    # Fetch the dataset from OpenML
    dataset = datasets.fetch_openml('credit-g', version=1, parser='auto', as_frame=True)
    # Identify the target variable
    target_variable = 'class'
    # Extract meta-features
    num_instances = dataset.frame.shape[0]
    num_features = dataset.frame.shape[1]
    num_classes = len(dataset.target.unique())
    correlation_matrix = dataset.frame.corr(numeric_only=True)
    covariance_matrix = dataset.frame.cov(numeric_only=True)
    skewness = dataset.frame.skew(numeric_only=True)
    kurtosis = dataset.frame.kurt(numeric_only=True)
    min_values = dataset.frame.min(numeric_only=True)
    max_values = dataset.frame.max(numeric_only=True)
    mean_values = dataset.frame.mean(numeric_only=True)
    median_values = dataset.frame.median(numeric_only=True)

    sd_ratio = (dataset.frame.std(numeric_only=True) / dataset.frame.mean(numeric_only=True)).mean()  

    class_entropy = entropy(dataset.frame[target_variable].value_counts(normalize=True), base=2)
    normal_entropy = class_entropy / np.log2(len(dataset.target.unique()))

    # Print meta-features
    print('Number of instances: ' + str(num_instances))
    print('Number of features: ' + str(num_features))
    print('Number of classes: ' + str(num_classes))
    print('Correlation matrix: ' + str(correlation_matrix))
    print('Covariance matrix: ' + str(covariance_matrix))
    print('Skewness: ' + str(skewness))
    print('Kurtosis: ' + str(kurtosis))
    print('Min values: ' + str(min_values))
    print('Max values: ' + str(max_values))
    print('Mean values: ' + str(mean_values))
    print('Median values: ' + str(median_values))
    print('Standard Deviation Ratio: ' + str(sd_ratio))

    print('Class Entropy:' + str(class_entropy.item()))
    print('Normal Entropy: ' + str(normal_entropy))

    # Extract features and target variable for the Mutual information features
    X = dataset.frame.drop(columns=[target_variable])
    y = dataset.frame[target_variable]

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X)

    # Calculate mutual information for each feature
    mutual_information_values = mutual_info_classif(X_encoded, y)

    # Print the results for all the features
    for feature, value in zip(X_encoded.columns, mutual_information_values):
        print(f'Mutual Information for {feature} : {value}')


extract_meta_features('credit-g')