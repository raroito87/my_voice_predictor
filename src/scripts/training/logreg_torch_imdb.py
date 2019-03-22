from preprocessing_utils import Preprocessing
import pandas as pd
import numpy as np
import torch
import argparse
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

    parser = argparse.ArgumentParser(description='IMDBData')
    parser.add_argument('--n_feat', default=1000, help='number of features')
    parser.add_argument('--s_model', default=False, help='save trained model')

    args=parser.parse_args()

    pre = Preprocessing('IMDB')

    n_features = int(args.n_feat)
    pre.load_data(filename=f'training_data_{n_features}.csv', name='training_data')

    X_df = pre.get(name='training_data').drop(columns=['target'])
    y_df = pre.get(name='training_data')['target']

    print(X_df.head())

    #going to use torch
    dtype = torch.float
    device = torch.device("cpu")

    X = torch.tensor(X_df.values, device=device, dtype=dtype)
    y = torch.tensor(y_df.values, device=device, dtype=dtype)


    # Softmax regression model

    #n_features = X.size()[1]
    n_classes = 2

    print('features: ', n_features)
    print('classes: ', n_classes)

    model = torch.nn.Sequential(
        torch.nn.Linear(n_features, n_classes),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    size_batch = 256
    loss_hist = []
    # Train
    start = time.time()

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

    if args.s_model:
        name = f'logreg_{n_features}.sav'
        pre.save_model(model, name=name)

    print(f'model trained with {n_features} features')
