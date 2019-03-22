from preprocessing_utils import Preprocessing, TextTransformer
import argparse
import pandas as pd
import numpy as np

import time

if not __name__ == '__main_':
    parser = argparse.ArgumentParser(description='IMDBData')

    parser.add_argument('--n_words', default=100, help='num words')
    parser.add_argument('--s_model', default=False, help='save transform model')

    args = parser.parse_args()

    n_words = args.n_words

    pre = Preprocessing('imdb')

    TRAIN_PATH_POS = 'train/pos/'
    TRAIN_PATH_NEG = 'train/neg/'

    pre.load_all_texts_from_directory(path=TRAIN_PATH_POS, name='raw_pos')
    pre.load_all_texts_from_directory(path=TRAIN_PATH_NEG, name='raw_neg')

    texts_list = pd.concat([pre.data['raw_pos'], pre.data['raw_neg']])

    print('fit & transform training data')

    text_transformer = TextTransformer(num_words=n_words)
    train_features = text_transformer.fit_and_transform(texts_list)
    train_features['target'] = np.append(np.ones(len(pre.data['raw_pos'])), np.zeros(len(pre.data['raw_neg'])))

    #shuffle datafram in-place
    train_features = train_features.sample(frac=1).reset_index(drop=True)

    pre.set('training_data' + '_' + str(n_words), train_features)
    pre.save('training_data' + '_' + str(n_words))

    #transform the test data

    TEST_PATH_POS = 'test/pos/'
    TEST_PATH_NEG = 'test/neg/'

    pre.load_all_texts_from_directory(path=TEST_PATH_POS, name='raw_pos_test')
    pre.load_all_texts_from_directory(path=TEST_PATH_NEG, name='raw_neg_test')

    print('transform test data')

    texts_list_test = pd.concat([pre.data['raw_pos_test'], pre.data['raw_neg_test']])
    test_features = text_transformer.transform(texts_list_test)
    test_features['target'] = np.append(np.ones(len(pre.data['raw_pos_test'])), np.zeros(len(pre.data['raw_neg_test'])))
    test_features = test_features.sample(frac=1).reset_index(drop=True)

    pre.set('test_data' + '_' + str(n_words), test_features)
    pre.save('test_data' + '_' + str(n_words))
    if args.s_model:
        name = f'word_transformer_{n_words}.sav'
        pre.save_model(text_transformer, name=name)
