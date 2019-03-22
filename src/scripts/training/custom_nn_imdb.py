from preprocessing_utils import Preprocessing
import numpy as np
import torch
import argparse
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch.nn as nn

from models import IMDB_NN_Model
from models import LogReg

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

    #Divide into train and validate

    X_train_df, X_val_df, y_train_df, y_val_df = train_test_split(X_df, y_df, test_size=0.20, random_state=42)

    #going to use torch
    dtype = torch.float
    device = torch.device("cpu")

    X_train = torch.tensor(X_train_df.values, device=device, dtype=dtype)
    y_train = torch.tensor(y_train_df.values, device=device, dtype=dtype)

    X_val = torch.tensor(X_val_df.values, device=device, dtype=dtype)
    y_val = torch.tensor(y_val_df.values, device=device, dtype=dtype)


    # Softmax regression model

    #n_features = X.size()[1]
    n_classes = 2

    H_0 = 150
    H_1 = 125
    n_iter = 500

    print('features: ', n_features)
    print('classes: ', n_classes)

    model = LogReg('logreg', n_features, n_classes)#IMDB_NN_Model('imdb', n_features, H_0, H_1, n_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    size_batch = 256
    loss_hist = []
    loss_val_hist = []
    acc_train = []
    acc_val = []
    # Train
    start = time.time()

    for t in range(n_iter):
        X_mini, y_mini = get_mini_batching(X_train, y_train, size_batch)

        # Berechne die Vorhersage (foward step)
        outputs = model(X_mini)

        # Berechne den Fehler (Ausgabe des Fehlers alle 50 Iterationen)
        loss = criterion(outputs, y_mini.long())

        # Berechne die Gradienten und Aktualisiere die Gewichte (backward step)
        optimizer.zero_grad()
        #print(f'gradient: {}')
        loss.backward()
        optimizer.step()

        y_pred = model(X_train).argmax(1)
        acc_train.append((y_pred == y_train.long()).float().mean())

        y_pred_val = model(X_val).argmax(1)
        acc_val.append((y_pred_val == y_val.long()).float().mean())

        loss_hist.append(criterion(model(X_train), y_train.long()))
        loss_val_hist.append(criterion(model(X_val), y_val.long()))
        if t % 50 == 0:
            print(t, loss.item())

    print(time.time() - start)

    y_pred = model(X_train).argmax(1)
    accuracy_soft = (y_pred == y_train.long()).float().mean()
    print(f'training accuracy: {accuracy_soft}')

    print(f'optimal iteration: {acc_val.index(max(acc_val))}')

    title = f'train_acc_val_acc_vs_iter_{n_iter}'
    plt.figure(1)
    plt.plot(acc_train)
    plt.plot(acc_val)
    plt.title = title
    pre.save_plt_as_image(plt, title)
    plt.close()

    title = f'train_loss_val_loss_vs_iter_{n_iter}'
    plt.figure(2)
    plt.plot(loss_hist)
    plt.plot(loss_val_hist)
    plt.title = title
    pre.save_plt_as_image(plt, title)

    #plt.show()

    if args.s_model:
        name = f'custom_{n_features}.pt'
        pre.save_model(model, name=name)
        #torch.save(model, name)

    print(f'model trained with {n_features} features')