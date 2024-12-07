from hyperopt import hp
import numpy as np

search_space_xgb = {
    'max_depth': hp.quniform('max_depth', 3, 10, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)), 
    'n_estimators': hp.quniform('n_estimators', 50, 500, 10),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1), 
}

search_space_catboost = {
    'depth': hp.quniform('depth', 3, 10, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)), 
    'iterations': hp.quniform('iterations', 50, 500, 10),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1), 
}

search_space_lgbm = {
    'max_depth': hp.quniform('max_depth', 3, 10, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)), 
    'n_estimators': hp.quniform('n_estimators', 50, 500, 10),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1), 
}

search_space_rf = {
    'n_estimators': hp.quniform('n_estimators', 50, 500, 10),
    'max_depth': hp.quniform('max_depth', 3, 10, 1),
    'max_features': hp.uniform('max_features', 0.5, 1),
    'criterion': hp.choice('criterion', ["gini", "entropy"]),
}

search_space_logistic = {
    'C': hp.loguniform('C', np.log(0.01), np.log(100)),
    'penalty': hp.choice('penalty', ["l1", "l2"]),
}

search_space_SVC = {
    'C': hp.loguniform('C', np.log(0.01), np.log(100)),
    'kernel': hp.choice('kernel', ["linear", "rbf"]),
}

search_knn = {
    'n_neighbors': hp.quniform('n_neighbors', 3, 50, 1),
    'weights': hp.choice('weights', ["uniform", "distance"]),
}