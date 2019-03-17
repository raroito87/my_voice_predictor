import pandas as pd
import numpy as np


class TextTransformer:

    def __init__(self,*,
                 num_words=2000,
                 encoding='count',
                 text_length=True,
                 ):

        self.num_words = int(num_words)
        self.dict_features = {}

    def fit(self, X):
        #X is a DataFrame with a list of texts
        #assert isinstance(X, pd.DataFrame), 'input is not pd.DataFrame'

        bag_of_words = self._create_bag_of_words(X)

        # the following looks complicated
        # it just makes sure that in case word_features contains repeated words (which it should not)
        # then the dictionary would still keep a continuous index as value,
        idx_top = 0
        idx_dic = 0
        while len(self.dict_features) < self.num_words:
            #    #print(len(dictionary), len(top_adj))
            if bag_of_words[idx_top] not in self.dict_features:
                self.dict_features[bag_of_words[idx_top]] = idx_dic
                idx_dic = idx_dic + 1
            idx_top = idx_top + 1

        print('fit completed with features len {}: \n'.format(len(self.dict_features)))
        return self.dict_features

    def transform(self, X):
        #assert isinstance(X, pd.DataFrame), 'input is not pd.DataFrame'
        #assert len(self.dict_features), 'need to fit transform first!'
        #X is a DataFrame with a list of texts
        #We return a vector containing the frequency of the words in dict_features

        feature_matrix = []
        idx = 0
        for text in X.texts:
            encoded_words = np.zeros(len(self.dict_features))
            for word in text.split(' '):
                if word in self.dict_features:
                    encoded_words[self.dict_features[word]] += 1

            feature_matrix += [encoded_words]
            idx = idx + 1

        return pd.DataFrame(feature_matrix, columns=list(self.dict_features.keys()))

    def fit_and_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def _turn_text_df_to_words_list(self, texts_df):
        # text_df is pd.DataFram
        all_words = []
        for text in texts_df.texts:
            words = np.str.split(text, sep=' ')
            all_words += words

        return all_words

    def _turn_texts_df_to_word_columns_df(self, texts_df):
        all_words = self._turn_text_df_to_words_list(texts_df)
        return pd.DataFrame(all_words, columns=['words'])

    def _create_bag_of_words(self, text):
        all_words = self._turn_texts_df_to_word_columns_df(text)
        all_words['count'] = 1
        word_count = all_words.groupby('words').count()

        return list(word_count.sort_values('count', ascending=False).index)
