from app import Preprocessing, TextTransformer
import argparse
import pandas as pd

import time





if not __name__ == '__main_':
    parser = argparse.ArgumentParser(description='IMDBData')

    parser.add_argument('--n_words', default=100, help='num words')

    args = parser.parse_args()

    n_words = args.n_words

    pre = Preprocessing('imdb')

    TRAIN_PATH_POS = 'train/pos/'
    TRAIN_PATH_NEG = 'train/neg/'

    #pre.load_all_texts_from_directory_as_words(path=TRAIN_PATH_POS, name='raw_pos')
    #pre.load_all_texts_from_directory_as_words(path=TRAIN_PATH_NEG, name='raw_neg')

    pre.load_all_texts_from_directory(path=TRAIN_PATH_POS, name='raw_pos')
    pre.load_all_texts_from_directory(path=TRAIN_PATH_NEG, name='raw_neg')

    texts_list = pre.data['raw_pos']#pd.concat([pre.data['raw_pos'], pre.data['raw_neg']])
    print(texts_list)


    text_transformer = TextTransformer(num_words=n_words)
    words_df, text_words_df = text_transformer._turn_texts_df_to_word_columns_df(texts_list)

    print(text_words_df)

    print(words_df)

    #text_transformer.fit(texts_list)
    #features = text_transformer.transform(texts_list)
    #print(features.head())
    #print(features.tail())



