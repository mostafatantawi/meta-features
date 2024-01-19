from sklearn import datasets
from scipy.stats import entropy
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from meta import extract_meta_features

#print(automl_datasets['name'].to_numpy())

# List of AutoML datasets
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
        extract_meta_features(dataset_name)
    except Exception as e:
        print(e)
        print('Error while fetching dataset: ' + dataset_name)
        continue