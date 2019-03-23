from sklearn import svm
from preprocessing_utils import Preprocessing, ModelExporter
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from sklearn.model_selection import GridSearchCV


def hiperparameter_optimization_GridSerach(X_train, y_train, X_val, y_val):
    # construct the set of hyperparameters to tune
    params = {"C": 2 ** np.arange(-5, 15, 2,  dtype=float)}

    #it’s normally preferable to used Randomized Search over Grid Search in nearly all circumstances.

    # tune the hyperparameters via a cross-validated grid search
    print("[INFO] tuning hyperparameters via grid search")
    model = svm.LinearSVC()
    grid = GridSearchCV(model, params)
    start = time.time()
    grid.fit(X_train, y_train)

    # evaluate the best grid searched model on the testing data
    print("[INFO] grid search took {:.2f} seconds".format(time.time() - start))
    acc = grid.score(X_val, y_val)
    print("[INFO] grid search accuracy: {:.2f}%".format(acc * 100))
    print("[INFO] grid search best parameters: {}".format(grid.best_params_))

    return grid.best_params_

def get_split_data(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=3)

    return X_train, y_train, X_val, y_val

if not __name__ == '__main_':

    parser = argparse.ArgumentParser(description='Digits')
    parser.add_argument('--s_model', default=True, help='save trained model')

    args=parser.parse_args()

    n_classes = 10
    n_epochs = 100

    pre = Preprocessing('digits')
    pre.load_data(filename='train.csv', name='train')

    X_df = pre.get(name='train').drop(columns=['0'])
    y_df = pre.get(name='train')['0']

    X_train, y_train, X_val, y_val = get_split_data(X_df, y_df)

    #clf = svm.SVC((kernel='linear'))
    #clf.fit(X_df, y_df)

    #linear kernels

    #C = hiperparameter_optimization_GridSerach(X_train, y_train, X_val, y_val)

    clf_l = svm.LinearSVC(C=0.03125)
    clf_l.fit(X_train.values, y_train.values)

    #train accuracy
    y_pred = clf_l.predict(X_train.values)

    print(f'train accuracy: {(y_pred==y_train).mean()}')

    pre.load_data(filename='test.csv', name='test')

    X_test = pre.get(name='test').drop(columns=['0'])
    y_test = pre.get(name='test')['0']


    #test accuracy
    y_pred = clf_l.predict(X_test)

    print(f'test accuracy: {(y_pred==y_test).mean()}')
