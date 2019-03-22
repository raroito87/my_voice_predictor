from train import TrainClassifier


from preprocessing_utils import Preprocessing, ModelExporter
import numpy as np
import torch
import argparse
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch.nn as nn

import models
from models import LogReg

if not __name__ == '__main_':

    parser = argparse.ArgumentParser(description='IMDBData')
    parser.add_argument('--n_feat', default=1000, help='number of features')
    parser.add_argument('--s_model', default=False, help='save trained model')

    args=parser.parse_args()

    pre = Preprocessing('IMDB')

    n_classes = 2
    n_features = int(args.n_feat)
    n_epochs = 100
    pre.load_data(filename=f'training_data_{n_features}.csv', name='training_data')

    X_df = pre.get(name='training_data').drop(columns=['target'])
    y_df = pre.get(name='training_data')['target']

    model = LogReg('log_reg', n_features, n_classes)

    train_classifier = TrainClassifier(model, X_df, y_df)
    trained_model, optimizer, criterion, loss_hist, loss_validate_hist = train_classifier.run_train(n_epochs = n_epochs)
    pre.save_results(loss_hist, loss_validate_hist, f'log_reg_{100}')

    m_exporter = ModelExporter('IMDB')
    m_exporter.save_nn_model(trained_model, optimizer, n_features, n_classes, n_epochs)

    ##teeeeeest part
    pre.load_data(filename=f'test_data_{n_features}.csv', name='test_data')

    X_test_df = pre.get(name='test_data').drop(columns=['target'])
    y_test_df = pre.get(name='test_data')['target']


    dtype = torch.float
    device = torch.device("cpu")
    X_test = torch.tensor(X_test_df.values, device=device, dtype=dtype)
    y_test = torch.tensor(y_test_df.values, device=device, dtype=torch.long)

    y_pred = model(X_test).argmax(1)
    accuracy_soft = (y_pred == y_test).float().mean()
    print(f'test accuracy {accuracy_soft.item()}')
