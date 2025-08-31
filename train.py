from process_data import train_set,train_set_labels , test_set ,test_set_labels,data,ProcessData
from sklearn.linear_model import LinearRegression 

model = LinearRegression()
model.fit(train_set,train_set_labels)

some_data = ProcessData(train_set[69:74])
some_data_labels = train_set_labels[69:74]
predictions = model.predict(some_data)
print("Predictions",predictions )
print("Labels",some_data_labels.to_numpy() )

