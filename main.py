import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import sys
from utils.metrics import quadratic_weighted_kappa
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


search_space = {
    'max_depth': hp.quniform('max_depth', 3, 10, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)), 
    'n_estimators': hp.quniform('n_estimators', 50, 500, 10),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1), 
}



def objective(params):

    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])   
    model = xgb.XGBClassifier(**params)
    model.fit(x_train.to_numpy(), y_train.to_numpy())
    y_pred = model.predict(x_test.to_numpy())
    score = quadratic_weighted_kappa(y_test, y_pred)

    return {'loss': -score, 'status': STATUS_OK}



if __name__=="__main__":

    # read the path given in the terminal
    path = sys.argv[1]

    # read the data
    train = pd.read_csv(path)

    # preprocessing

    cat_c = ['Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 'Fitness_Endurance-Season', 
          'FGC-Season', 'BIA-Season', 'PAQ_A-Season', 'PAQ_C-Season', 'SDS-Season', 'PreInt_EduHx-Season']
    pciat = train.columns[train.columns.str.startswith('PCIAT-PCIAT')].tolist() + ['sii', "PCIAT-Season"]

    train_clean = pd.concat([train, pd.get_dummies(train[cat_c]).astype(int)], axis=1)
    to_drop = ["id"] + cat_c
    train_clean = train_clean.drop(to_drop, axis=1)
    train_clean = train_clean.dropna(subset=pciat)

    x_train, x_test, y_train, y_test = train_test_split(train_clean.drop(pciat, axis=1), train_clean['sii'], test_size=0.2, random_state=42)

    # Fine Tuning model

    trials = Trials()

    best_params = fmin(
    fn=objective,  # Objective function
    space=search_space,  # Hyperparameter search space
    algo=tpe.suggest,  # Tree-structured Parzen Estimator
    max_evals=50,  # Number of iterations
    trials=trials,  # Store trial results
    rstate=np.random.default_rng(42)  # For reproducibility
    )

    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['n_estimators'] = int(best_params['n_estimators'])

    # Train the model
    model = xgb.XGBClassifier(**best_params)
    model.fit(x_train.to_numpy(), y_train.to_numpy())

    # Predict on the test set
    y_pred = model.predict(x_test.to_numpy())

    # write a log file
    with open("logs/model_results.txt", "w") as f:
        f.write(f"QWK: {quadratic_weighted_kappa(y_test, y_pred)}\n")
        f.write("------------------------------------------------------------------\n")
        f.write(f"Accuracy: {model.score(x_test.to_numpy(), y_test.to_numpy())}\n")
        f.write("------------------------------------------------------------------\n")
        f.write(f"Model: {model.get_params()}\n")
        f.write("------------------------------------------------------------------\n")
        f.write(f"Data: {path}\n")
        f.write("------------------------------------------------------------------\n")
        f.write(f"Features: {train_clean.columns}\n")
        f.write("------------------------------------------------------------------\n")

    print("Done!")