from sklearn import datasets
from scipy.stats import entropy
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


#print(automl_datasets['name'].to_numpy())

# List of AutoML datasets (you can replace this with your list)
automl_datasets = ['kr-vs-kp', 'letter', 'balance-scale', 'mfeat-factors', 'mfeat-fourier', 'breast-w',
                   'mfeat-karhunen', 'mfeat-morphological', 'mfeat-zernike', 'cmc', 'optdigits', 'credit-approval',
                   'credit-g', 'pendigits', 'diabetes', 'sick', 'spambase', 'splice', 'tic-tac-toe', 'vehicle',
                   'electricity', 'satimage', 'eucalyptus', 'isolet', 'vowel', 'analcatdata_authorship',
                   'analcatdata_dmft', 'mnist_784', 'pc4', 'pc3', 'jm1', 'kc2', 'kc1', 'pc1', 'bank-marketing',
                   'banknote-authentication', 'blood-transfusion-service-center', 'cnae-9', 'first-order-theorem-proving',
                   'har', 'ilpd', 'madelon', 'nomao', 'ozone-level-8hr', 'phoneme', 'qsar-biodeg', 'wall-robot-navigation',
                   'semeion', 'wdbc', 'adult', 'Bioresponse', 'PhishingWebsites', 'GesturePhaseSegmentationProcessed',
                   'cylinder-bands', 'dresses-sales', 'numerai28.6', 'texture', 'connect-4', 'dna', 'churn',
                   'Devnagari-Script', 'CIFAR_10', 'MiceProtein', 'car', 'Internet-Advertisements', 'mfeat-pixel',
                   'steel-plates-fault', 'wilt', 'segment', 'climate-model-simulation-crashes', 'Fashion-MNIST',
                   'jungle_chess_2pcs_raw_endgame_complete', 'JapaneseVowels']

# Loop through each dataset
for dataset_name in automl_datasets:
    try:
        # Fetch the dataset from scikit-learn
        dataset = datasets.fetch_openml(name=dataset_name, version=1, as_frame=True)

        # Identify the target variable
        target_variable = dataset.target.name

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
        print(f'Dataset: {dataset_name}')
        print('Number of instances:', num_instances)
        print('Number of features:', num_features)
        print('Number of classes:', num_classes)
        print('Correlation matrix:', correlation_matrix)
        print('Covariance matrix:', covariance_matrix)
        print('Skewness:', skewness)
        print('Kurtosis:', kurtosis)
        print('Min values:', min_values)
        print('Max values:', max_values)
        print('Mean values:', mean_values)
        print('Median values:', median_values)
        print('Standard Deviation Ratio:', sd_ratio)
        print('Class Entropy:', class_entropy.item())
        print('Normal Entropy:', normal_entropy)

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

        print('\n' + '-'*50 + '\n')

    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {e}")
