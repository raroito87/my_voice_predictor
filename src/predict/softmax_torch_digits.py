import pickle

import torch
import torch.nn as nn

from app import Preprocessing
from app import Image_Importer

from numpy import random

import argparse

import numpy as np

import imageio
from skimage.transform import rescale, resize


from sklearn import preprocessing


def get_num_and_data(args):

    if not args.num:
        print('use test data')
        pre = Preprocessing('digits')
        pre.load_data(filename='test.csv', name='test')

        X_test = torch.tensor(pre.get('test').drop(columns=['0']).values, device=device, dtype=dtype)
        y_test = torch.tensor(pre.get('test')['0'].values, device=device, dtype=dtype)

        index = random.randint(0, len(y_test))
        print('index {}'.format(index))

        num = int(y_test[index].item())
        print('real_numer {}'.format(num))
        data_to_predict = X_test[index:index + 1, :]
        return num, data_to_predict


    else:
        print('use written images')
        num = int(args.num)
        im_imp = Image_Importer('digits')
        im_imp.load_image_as_grey(num)

        print('real_numer {}'.format(num))
        data_to_predict = im_imp.get_image_as_256px_array(num)

        return num, data_to_predict


if not __name__ == '__main_':


    parser = argparse.ArgumentParser(description='Digits')

    parser.add_argument('--num', default=False, help='load number image')

    args = parser.parse_args()

    dtype = torch.float
    device = torch.device("cpu")

    filename = "../data/digits/softmax_torch_digits_model.sav"
    model = pickle.load(open(filename, 'rb'))

    real_number, data_to_predict = get_num_and_data(args)

    #real_number =# 0
    #data_to_predict = np.array([])
#
    #if not args.num:
    #    print('use test data')
    #    pre = Preprocessing('digits')
    #    pre.load_data(filename='test.csv', name='test')
#
    #    X_test = torch.tensor(pre.get('test').drop(columns=['0']).values, device=device, dtype=dtype)
    #    y_test = torch.tensor(pre.get('test')['0'].values, device=device, dtype=dtype)
#
    #    index = random.randint(0, len(y_test))
    #    print('index {}'.format(index))
#
    #    real_nummber = int(y_test[index].item())
    #    print('real_numer {}'.format(real_nummber))
    #    data_to_predict = X_test[index:index + 1, :]
#
    #else:
    #    print('use written images')
    #    real_number = args.num
    #    im_imp = Image_Importer('digits')
    #    im_imp.load_image_as_grey(real_number)
    #    data_to_predict = im_imp.get_image_as_256px_array(real_number)

    softmax = torch.nn.Softmax()
    y_pred = softmax(model(data_to_predict)).argmax(1)
    b = y_pred.item()

    print('echte zahl: {}'.format(real_number))
    print('vohergesagte zahl: {} '.format(b))
    print('real_number == b : ', real_number == b)

    if real_number == b:
        print(', yuhu! nimm ein Bier. Neue Fische bezahlt dafür')
    else:
        print('schade, pech gehabt, versuchs nochmal, du looser')
#
