from process_data import train_set,train_set_labels , test_set ,test_set_labels,data,ProcessInputData
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import numpy as np 

def train():
    model = RandomForestRegressor()
    model.fit(train_set,train_set_labels)
    return model

trained_model = train()
prepared_data = ProcessInputData(train_set.iloc[:5])
prepared_data_labels = train_set_labels.iloc[69:74]
predictions = trained_model.predict(prepared_data)
lin_rmse = root_mean_squared_error(prepared_data_labels,predictions)
print(lin_rmse)

