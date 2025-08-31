import os 
import math
import tarfile
from six.moves import urllib 
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy  as np

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
DATA_PATH = os.path.join("datasets","housing")
DATA_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
def fetch_data(data_url = DATA_URL,data_path=DATA_PATH):
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    tgz_path = os.path.join(data_path,"housing.tgz")
    urllib.request.urlretrieve(data_url,tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=data_path,filter="data")
    housing_tgz.close()
    os.remove(tgz_path)

fetch_data()

def load_data_to_pandas(data_path=DATA_PATH,file_name="housing.csv"):
    data_file_path = os.path.join(data_path,file_name)
    return pd.read_csv(data_file_path)

data = load_data_to_pandas() 

def test_train_sample_random(data,test_ratio=0.2):
    np.random.seed(42)
    total_len = len(data)
    random_indices = np.random.permutation(total_len)
    test_set_indices = math.floor(total_len*test_ratio)
    test_set = random_indices [:test_set_indices]
    train_set = random_indices[test_set_indices:]
    
    return data.iloc[train_set],data.iloc[test_set]

def save_test_train_samples(test_set,train_set,data_path=DATA_PATH):
    test_set_path  = os.path.join(data_path,"test_set.csv")
    train_set_path  = os.path.join(data_path,"train_set.csv")
    test_set.to_csv(test_set_path, index=False)
    train_set.to_csv(train_set_path, index=False)

def load_test_train_samples(data,data_path=DATA_PATH, test_file_name="test_set.csv",train_file_name="train_set.csv"):
    payload = {"test_set":[],"train_set":[]}
    test_set_path  = os.path.join(data_path,test_file_name)
    train_set_path  = os.path.join(data_path,train_file_name)
    if (not os.path.isfile(train_set_path)) or (not os.path.isfile(test_set_path)):
        sampled_data = test_train_sample_random(data)
        payload["test_set"] = sampled_data[0]
        payload["train_set"] = sampled_data[1]
        save_test_train_samples(payload["test_set"],payload["train_set"])
        return payload
    
    payload["test_set"] = pd.read_csv(test_set_path)
    payload["train_set"] = pd.read_csv(train_set_path)
    
    return payload

sample_state = load_test_train_samples(data)

train_set = sample_state["train_set"]
test_set = sample_state["test_set"]


def fill_missing_values_by_median(numeric_data):
    imputer = SimpleImputer(strategy="median")
    imputer.fit(numeric_data)
    transformed_array = imputer.transform(numeric_data)
    return imputer.statistics_ , transformed_array 

train_set_numeric_fields = train_set.drop("ocean_proximity",axis=1)

result = fill_missing_values_by_median(train_set_numeric_fields)
train_tr = pd.DataFrame(result[1],columns=train_set_numeric_fields.columns)


train_set_numeric_fields["bedrooms_per_rooms"] = train_set_numeric_fields["total_bedrooms"]/train_set_numeric_fields["total_rooms"]
train_set_numeric_fields["rooms_per_household"] = train_set_numeric_fields["total_rooms"]/train_set_numeric_fields["households"]
correlations_matrix = train_set_numeric_fields.corr()

print(correlations_matrix["median_house_value"].sort_values(ascending=False))
    

