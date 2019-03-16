from app import Preprocessing
import pandas as pd
import numpy as np

import torch

import time


import torch.nn as nn

import pickle



def get_mini_batching(X_train, y_train, batch_size):
    #idx = np.random.randint(0, len(y_train),  batch_size)
    idx = np.random.choice(range(len(y_train)), batch_size, replace=False)
    X_train_mini = X_train[idx, :]
    y_train_mini = y_train.view(len(y_train), 1)[idx, :]#:batch_size, :]

    return X_train_mini, y_train_mini.view(len(y_train_mini))

if not __name__ == '__main_':

    pre = Preprocessing('IMDB')

    pre.load_data(filename='training_data_1000.csv', name='training_data')

    X_df = pre.get(name='training_data').drop(columns=['target'])
    y_df = pre.get(name='training_data')['target']

    print(X_df.tail())
    print(y_df.tail())

    #going to use torch
    dtype = torch.float
    device = torch.device("cpu")

    X = torch.tensor(X_df.values, device=device, dtype=dtype)
    y = torch.tensor(y_df.values, device=device, dtype=dtype)


    # Softmax regression model

    n_features = X.size()[1]
    n_classes = 2

    print('features: ', n_features)
    print('classes: ', n_classes)

    model = torch.nn.Sequential(
        torch.nn.Linear(n_features, n_classes),
        #torch.nn.Sigmoid()#softmax for multiclass
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    size_batch = 256
    loss_hist = []
    # Train
    start=time.time()

    for t in range(5000):
        X_mini, y_mini = get_mini_batching(X, y, size_batch)

        # Berechne die Vorhersage (foward step)
        outputs = model(X)

        # Berechne den Fehler (Ausgabe des Fehlers alle 50 Iterationen)
        loss = criterion(outputs, y.long())

        # Berechne die Gradienten und Aktualisiere die Gewichte (backward step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Berechne den Fehler (Ausgabe des Fehlers alle 100 Iterationen)
        if t % 50 == 0:
            loss_hist.append(loss.item())
            print(t, loss.item())

    print(time.time() - start)

    #plt.figure(1)
    #plt.plot(loss_hist)
    #plt.show()


    pre.load_data(filename='test_data_1000.csv', name='test_data')

    X_test_df = pre.get(name='test_data').drop(columns=['target'])
    y_test_df = pre.get(name='test_data')['target']


    X_test = torch.tensor(X_test_df.values, device=device, dtype=dtype)
    y_test = torch.tensor(y_test_df.values, device=device, dtype=torch.long)

    y_pred = model(X_test).argmax(1)

    accuracy_soft = (y_pred == y_test).float().mean()

    #results = (y_test == y_pred)

    #print(torch.sum(results).item()/len(results))
    print(accuracy_soft.item())
