import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import sys
from utils.metrics import quadratic_weighted_kappa
from utils.variables import search_space_xgb, search_space_lgbm, search_space_catboost, search_space_rf, search_space_logistic, search_knn, search_space_SVC
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from warnings import filterwarnings

filterwarnings('ignore')


dict_models = {
    'xgb': {"model": xgb.XGBClassifier, "search_space": search_space_xgb},
    'lgbm': {"model": LGBMClassifier, "search_space": search_space_lgbm},
    'catboost': {"model": CatBoostClassifier, "search_space": search_space_catboost},
    'rf': {"model": RandomForestClassifier, "search_space": search_space_rf},
}

if __name__=="__main__":

    train_path = sys.argv[1]
    name = sys.argv[2]

    # read the data
    train = pd.read_csv(train_path)
    # train = train.drop("PreInt_EduHx-Season.1", axis=1)

    # List of columns to impute (categorical variables that were not imputed in our soft impute preprocessing)
    to_impute = ['FGC-FGC_CU_Zone', 'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD_Zone',
                'FGC-FGC_PU_Zone', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR_Zone',
                'FGC-FGC_TL_Zone', 'BIA-BIA_Activity_Level_num', 'BIA-BIA_Frame_num']

    # Impute each column with its most common class
    for column in to_impute:
        if column in train.columns:
            most_common_class = train[column].mode()[0]  # Get the most frequent value
            train[column].fillna(most_common_class, inplace=True)

    # read the time series data
    ts = pd.read_csv("time_series_extraction/ts_extract_train_merged.csv")
    ts["id"] = ts["kid_id"]
    ts = ts.drop(columns=["kid_id", "index"])

    # Merge the time series data with the covariates
    train = pd.merge(train, ts, on="id", how="left")

    # preprocessing (one hot encoding of categorical variables)
    cat_c = ['Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 'Fitness_Endurance-Season', 
            'FGC-Season', 'BIA-Season', 'PAQ_A-Season', 'PAQ_C-Season', 'SDS-Season', 'PreInt_EduHx-Season']
    pciat = train.columns[train.columns.str.startswith('PCIAT-PCIAT')].tolist() + ['sii', "PCIAT-Season"]

    train_clean = pd.concat([train, pd.get_dummies(train[cat_c]).astype(int)], axis=1)
    to_drop = ["id"] + cat_c
    train_clean = train_clean.drop(to_drop, axis=1)
    train_clean = train_clean.dropna(subset=pciat)

    # Split the data into train, val and test
    x_train, x_test, y_train, y_test = train_test_split(train_clean.drop(pciat, axis=1), train_clean['sii'], test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    results = []



    for clf in dict_models.keys():

        print("Training model: ", clf)

        clf_model = dict_models[clf]['model']
        search_space = dict_models[clf]['search_space']

        def objective(params):
            """
            Objective function for optimizing a classifier model.
            This function trains a classifier model with the given parameters, 
            makes predictions on the validation set, and calculates the 
            quadratic weighted kappa score as the evaluation metric.
            Args:
                params (dict): Dictionary of parameters to be used for the classifier model.
            Returns:
                dict: A dictionary containing the negative quadratic weighted kappa score 
                      as 'loss' and the status 'STATUS_OK'.
            """

            if clf=="lgbm":
                params["verbose"] = -1
            if clf=="catboost":
                params["verbose"] = 0

            model = clf_model(**params)
            model.fit(x_train.to_numpy(), y_train.to_numpy())

            y_pred = model.predict(x_val.to_numpy())
            score = cohen_kappa_score(y_val, y_pred, weights='quadratic')

            return {'loss': -score, 'status': STATUS_OK}
        
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

        if clf == "rf":
            best_params["n_estimators"] = int(best_params["n_estimators"])
            best_params["max_depth"] = int(best_params["max_depth"])

        if clf == "catboost":
            best_params["verbose"] = 0

        if clf == "xgb":
            best_params["max_depth"] = int(best_params["max_depth"])
            best_params["n_estimators"] = int(best_params["n_estimators"])

        if clf == "lgbm":
            best_params["max_depth"] = int(best_params["max_depth"])
            best_params["n_estimators"] = int(best_params["n_estimators"])


        if clf == "knn":
            best_params["n_neighbors"] = int(best_params["n_neighbors"])

        # Train the model
        model = clf_model(**best_params)
        model.fit(x_train.to_numpy(), y_train.to_numpy())

        # Predict on the test set
        y_pred_test = model.predict(x_test.to_numpy())
        y_pred_val = model.predict(x_val.to_numpy())
        y_pred_train = model.predict(x_train.to_numpy())

        # results.append(pd.DataFrame(index = [clf], data = {'QWK': [cohen_kappa_score(y_test, y_pred, weights='quadratic')], 'params': [best_params], "Accuracy": [model.score(x_test.to_numpy(), y_test.to_numpy())]}))
        results.append(pd.DataFrame(index = [clf], data = {'QWK_test': [cohen_kappa_score(y_test, y_pred_test, weights='quadratic')], 
                                                           "QWK_val": [cohen_kappa_score(y_val, y_pred_val, weights='quadratic')], 
                                                           "QWK_train": [cohen_kappa_score(y_train, y_pred_train, weights='quadratic')],
                                                           'acc_test': [model.score(x_test.to_numpy(), y_test.to_numpy())],
                                                           'acc_val': [model.score(x_val.to_numpy(), y_val.to_numpy())],
                                                           'acc_train': [model.score(x_train.to_numpy(), y_train.to_numpy())],
                                                           'params': [best_params]}))

    results = pd.concat(results)

    # Save the results
    results.to_csv(f"final_results/{name}_results.csv")