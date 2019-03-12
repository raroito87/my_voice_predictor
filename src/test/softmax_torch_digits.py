import pickle

import torch
import torch.nn as nn


from app import Preprocessing



if not __name__ == '__main_':


    pre = Preprocessing('digits')
    pre.load_data(filename='test.csv', name='test')

    dtype = torch.float
    device = torch.device("cpu")

    X_test = torch.tensor(pre.get('test').drop(columns=['0']).values,  device=device, dtype=dtype)
    y_test = torch.tensor(pre.get('test')['0'].values,  device=device, dtype=dtype)
    #print(y_test)

    filename = "../data/digits/softmax_torch_digits_model.sav"
    model = pickle.load(open(filename, 'rb'))

    y_pred = model(X_test)

    softmax = torch.nn.Softmax(dim=1)
    y_pred = softmax(y_pred).argmax(1)
    #print(y_pred)

    result = y_test == y_pred.float()
    print(100*float(result.sum())/len(result), '% sucess')


    #criterion = nn.CrossEntropyLoss()

    #print('error value: {}'.format(criterion(y_pred, y_test.long()).item()))

