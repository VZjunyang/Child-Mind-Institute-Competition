from hyperopt import hp
import numpy as np
from hyperopt.pyll import scope

search_space_xgb = {
    'max_depth': scope.int(hp.quniform('max_depth', 3, 10, 1)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)), 
    'n_estimators': scope.int(hp.quniform('n_estimators', 50, 500, 10)),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1), 
}

search_space_catboost = {
    'depth': scope.int(hp.quniform('depth', 3, 10, 1)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)), 
    'iterations': scope.int(hp.quniform('iterations', 50, 500, 10)),
}

search_space_lgbm = {
    'max_depth': scope.int(hp.quniform('max_depth', 3, 10, 1)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)), 
    'n_estimators': scope.int(hp.quniform('n_estimators', 50, 500, 10)),
    'subsample': hp.uniform('subsample', 0.5, 1),
}

search_space_rf = {
    'n_estimators': scope.int(hp.quniform('n_estimators', 50, 300, 10)),
    'max_depth': scope.int(hp.quniform('max_depth', 3, 10, 1)),
    'max_features': hp.uniform('max_features', 0.5, 1),
}

search_space_logistic = {
    'C': hp.loguniform('C', np.log(0.01), np.log(100)),
}

search_space_SVC = {
    'C': hp.loguniform('C', np.log(0.01), np.log(100)),
    'kernel': hp.choice('kernel', ["linear", "rbf"]),
}

search_knn = {
    'n_neighbors': scope.int(hp.quniform('n_neighbors', 3, 50, 1)),
}