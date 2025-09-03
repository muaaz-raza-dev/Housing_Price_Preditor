from process_data import train_set, train_set_labels, test_set, test_set_labels, data, ProcessDataPipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.model_selection import cross_val_score, GridSearchCV 
from sklearn.metrics import mean_squared_error
import numpy as np
import os


MODELS_PATH = "models"


def linear_regression_model(fit=True):
    model = LinearRegression()
    if (fit == False):
        return model

    model.fit(train_set, train_set_labels)
    return model


def decision_tree_model(fit=True):
    model = DecisionTreeRegressor()
    if (fit == False):
        return model
    model.fit(train_set, train_set_labels)
    return model


def random_forest_model(fit=True):
    model = RandomForestRegressor()
    if (fit == False):
        return model
    model.fit(train_set, train_set_labels)
    return model


def cross_validation_model(model):
    scores = cross_val_score(
        model, train_set, train_set_labels, scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)
    return tree_rmse_scores


def display_scores(scores):
    print("Scores : ", scores)
    print("Scores Mean : ", scores.mean())
    print("Scores Deviation : ", scores.std())


# linear_reg_model = linear_regression_model(False)
# tree_model = decision_tree_model(False)
forest_model = random_forest_model(False)


# linear_reg_model_cv = cross_validation_model(linear_reg_model)
# print("Linear Regression Model ")
# display_scores(linear_reg_model_cv)

# tree_model_cv = cross_validation_model(tree_model)
# print("Decision Tree Model ")
# display_scores(tree_model_cv)

# random_forest_model_cv = cross_validation_model(forest_model)
# print("Random Forest Model ")
# display_scores(random_forest_model_cv)

# joblib.dump({"model":linear_reg_model,"params":linear_reg_model.get_params(),"cv":linear_reg_model_cv.mean(),},os.path.join(MODELS_PATH,"lin_reg_model.pkl"))
# joblib.dump({"model":tree_model,"params":tree_model.get_params(),"cv":tree_model_cv.mean(),},os.path.join(MODELS_PATH,"decision_tree_model.pkl"))
# joblib.dump({"model":forest_model,"params":forest_model.get_params(),"cv":random_forest_model_cv.mean(),},os.path.join(MODELS_PATH,"random_forest_model.pkl"))

#! HyperParameter tuning

def grid_search_model(model, param_grid, data, labels):
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
    grid_search.fit(data, labels)
    return grid_search


param_grid = [{'n_estimators': [30,40,50], 'max_features': [2, 4, 6, 8]},
              {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
              ]

random_forest_model_grid_search = grid_search_model(forest_model,param_grid,train_set,train_set_labels)
final_model = random_forest_model_grid_search.best_estimator_

def EvaluateTestSet(test_set,test_set_labels, model):
    predictions = model.predict(test_set)
    mse = mean_squared_error(predictions,test_set_labels)
    return np.sqrt(mse) 


print(EvaluateTestSet(test_set=test_set,test_set_labels=test_set_labels,model=final_model))