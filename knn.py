from google.colab import userdata
email = userdata.get('email')
password = userdata.get('password')

from orkg import ORKG, Hosts

orkg = ORKG(host=Hosts.SANDBOX, creds=(email, password))

resource1 = orkg.resources.find_or_add(label="PermuteAttack: Counterfactual Explanation of Machine Learning Credit Scorecards", classes=['Paper'])

resource2 = orkg.resources.find_or_add(label="FairNeuron: improving deep neural network fairness with adversary games on selective neurons", classes=['Paper'])

resource3 = orkg.resources.find_or_add(label="Support vector machine with quantile hyper-spheres for pattern classification", classes=['Paper'])

resource4 = orkg.resources.find_or_add(label="GrAMME: Semi-Supervised Learning using Multi-layered Graph Attention Models", classes=['Paper'])

resource5 = orkg.resources.find_or_add(label="Interpretable Counterfactual Explanations Guided by Prototypes", classes=['Paper'])

import requests
import pandas as pd
from json.decoder import JSONDecodeError
import json

print(resource2.content)

api_base_url = 'https://sandbox.orkg.org/paper/'

response = requests.get(api_base_url, headers={'Content-Type': 'application/json', 'Accept': 'application/json'})

print(response)

def get_dataset_for_resource(resource_id):
    api_url = f'https://sandbox.orkg.org/api/resources/{resource_id}'
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            dataset = response.json()
            return dataset
        else:
            print(f"Error fetching dataset for resource {resource_id}. Status code: {response.status_code}")
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

import requests

url = 'https://sandbox.orkg.org/api/resources/R345024'

response = requests.get(url)

if response.status_code == 200:
    resource_data = response.json()
    properties = resource_data.get('properties', {})

    for prop_key, prop_value in properties.items():
        print(f"{prop_key}: {prop_value}")
else:
    print(f"Failed to fetch resource details. Status code: {response.status_code}")

properties

import pandas as pd

datasets_list = []

resource_ids = ['R345042', 'R343217', 'R345011', 'R345003', 'R345031', 'R346033', 'R346039', 'R346046', 'R116789', 'R346061']

for resource_id in resource_ids:
    dataset = get_dataset_for_resource(resource_id)

    if dataset:
        df = pd.DataFrame(dataset)

        if not df.empty:
            datasets_list.append(df)
        else:
            print(f"Empty DataFrame for resource {resource_id}")
    else:
        print(f"Dataset not found {resource_id}")

combined_df = pd.concat(datasets_list)

combined_df.head()

number_of_instance = orkg.statements.get_by_subject_and_predicate(subject_id='R343217', predicate_id='P39021')

statements = orkg.statements.get_by_subject(subject_id='R343217')

import json
import requests

resource_ids = ['R345042', 'R343217', 'R345011', 'R345003', 'R345031', 'R346033', 'R346039', 'R346046', 'R116789', 'R346061']
all_data = []
for resource_id in resource_ids:
    response = requests.get(f'https://sandbox.orkg.org/api/statements/subject/{resource_id}', headers={'Accept': 'application/json'})

    if response.status_code == 200:
        data = response.json()

        statements = data['content']

        filtered_statements = [statement for statement in statements if statement['predicate']['label'] != 'hasFeature']

        print(f"Filtered statements for resource ID {resource_id}:")
        all_data.extend(filtered_statements)
    else:
        print(f'Failed to fetch data for resource ID {resource_id}:', response.status_code)

df = pd.DataFrame(all_data)
df.head()

import pandas as pd
import requests

resource_ids = ['R345042', 'R343217', 'R345011', 'R345003', 'R345031', 'R346033', 'R346039', 'R346046', 'R116789', 'R346061']
all_dfs = []

for resource_id in resource_ids:
    response = requests.get(f'https://sandbox.orkg.org/api/statements/subject/{resource_id}', headers={'Accept': 'application/json'})

    if response.status_code == 200:
        data = response.json()
        statements = data['content']

        filtered_statements = [statement for statement in statements if statement['predicate']['label'] != 'hasFeature']

        df = pd.DataFrame({
            'resource_name': [statement['subject']['label'] for statement in filtered_statements],
            'predict_name': [statment['predicate']['label'] for statment in filtered_statements],
            'object_id': [statement['object']['id'] for statement in filtered_statements],
            'object_label': [statement['object']['label'] for statement in filtered_statements],
            'object_datatype': [statement['object'].get('datatype', None) for statement in filtered_statements]
        })

        all_dfs.append(df)

    else:
        print(f'Failed to fetch data for resource ID {resource_id}:', response.status_code)

result_df = pd.concat(all_dfs, ignore_index=True)

result_df

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np

result_df['object_label'] = pd.to_numeric(result_df['object_label'], errors='coerce')

scaler = StandardScaler()
X = scaler.fit_transform(result_df[['object_label']])

distances = np.sqrt(np.sum((X[:, np.newaxis] - X) ** 2, axis=2))

k = 3
neigh = NearestNeighbors(n_neighbors=k)
neigh.fit(X)

knn_indices = neigh.kneighbors(return_distance=False)

for i, resource_id in enumerate(resource_ids):
    print(f"K-nearest neighbors for resource {resource_id}:")
    for j, neighbor_index in enumerate(knn_indices[i]):
        neighbor_id = result_df.iloc[neighbor_index]['object_id']
        neighbor_label = result_df.iloc[neighbor_index]['object_label']
        neighbor_name = result_df.iloc[neighbor_index]['resource_name']
        pred_name = result_df.iloc[neighbor_index]['predict_name']
        print(f"Neighbor {j+1}: ID={neighbor_id}, Label={neighbor_label}, Resource Name={neighbor_name}, Predicate Name={pred_name}")
    print()