from preprocessing_utils import Preprocessing
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

import numpy as np
import argparse

from models import LogReg
import time

def get_mini_batching(X_train, y_train, batch_size):
    idx = np.random.choice(range(len(y_train)), batch_size, replace=False)
    X_train_mini = X_train[idx, :][:batch_size, :]
    y_train_mini = y_train.view(len(y_train), 1)[idx, :]

    return X_train_mini, y_train_mini.view(len(y_train_mini))


if not __name__ == '__main_':
    parser = argparse.ArgumentParser(description='digits')
    parser.add_argument('--s_model', default=False, help='save trained model')

    args=parser.parse_args()

    pre = Preprocessing('digits')
    pre.load_data(filename='train.csv', name='train')

    X_train_df, X_val_df, y_train_df, y_val_df = train_test_split(pre.get('train').drop(columns=['0']),
                                                                  pre.get('train')['0'],
                                                                  test_size=0.01)

    #transfom to torch striuctures
    dtype = torch.float
    device = torch.device("cpu")

    X_train = torch.tensor(X_train_df.values, device=device, dtype=dtype)
    y_train = torch.tensor(y_train_df.values, device=device, dtype=dtype)

    # Softmax regression model

    n_features = X_train.size()[1]
    n_classes = len(np.unique(y_train.round().numpy()))

    model = LogReg(name='logreg', d_in=n_features, d_out=n_classes)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    size_batch = 100
    loss_hist = []

    # Train
    start=time.time()
    for t in range(5000):

        X_train_mini, y_train_mini = get_mini_batching(X_train, y_train, size_batch)

        # Berechne die Vorhersage (foward step)
        outputs = model(X_train_mini)

        # Berechne den Fehler (Ausgabe des Fehlers alle 50 Iterationen)
        loss = criterion(outputs, y_train_mini.long())

        # Berechne die Gradienten und Aktualisiere die Gewichte (backward step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Berechne den Fehler (Ausgabe des Fehlers alle 100 Iterationen)
        if t % 50 == 0:
            loss_hist.append(loss.item())
            #print(t, loss.item())

    print(time.time()-start)

    if args.s_model:
        name = f'logreg_{n_features}.sav'
        pre.save_model(model, name=name)