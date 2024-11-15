import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import sys
from utils.metrics import quadratic_weighted_kappa

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

    # model

    model = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05)

    model.fit(x_train.to_numpy(), y_train.to_numpy())

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