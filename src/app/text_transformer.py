import re
import pandas as pd
import numpy as np


class TextTransformer:

    def __init__(self,*,
                 punc_rm=True,
                 lower=True,
                 num_rm=True,
                 stop_rm=False,
                 num_words=2000,
                 encoding='count',
                 text_length=True,
                 ):
        self.punc_rm = punc_rm  # remove punctuation
        self.lower = lower  # set all letters to lower case
        self.num_rm = num_rm  # remove numbers
        self.stop_rm = stop_rm  # remove stopwords (a, the, ...)
        self.num_words = int(num_words)

        self.dict_features = {}

    def fit(self, X):
        #X is a DataFrame with a list of texts

        all_words, word_columns_df = self._turn_texts_df_to_word_columns_df(X)

        X = all_words
        X['count'] = 1
        word_count = X.groupby('words').count()
        print(word_count.tail())
        all_word_features = list(word_count.sort_values('count', ascending=False).index)

        # the following looks complicated
        # it just makes sure that in case word_features contains repeated words (which it should not)
        # then the dictionary would still keep a continuous index as value,
        idx_top = 0
        idx_dic = 0
        while len(self.dict_features) < self.num_words:
            #    #print(len(dictionary), len(top_adj))
            if all_word_features[idx_top] not in self.dict_features:
                self.dict_features[all_word_features[idx_top]] = idx_dic
                idx_dic = idx_dic + 1
            idx_top = idx_top + 1

        print('fit completed with features len {}: \n'.format(len(self.dict_features)))
        print(self.dict_features)
        return self.dict_features

    def transform(self, X):
        assert len(self.dict_features), 'need to fit transform first!'
        #X is a DataFrame with a list of texts
        #We return a vector containing the frequency of the words in dict_features

        all_words, word_columns_df = self._turn_texts_df_to_word_columns_df(X)
        X = word_columns_df

        feature_matrix = []
        idx = 0
        for text in X.texts:
            words_df = self._turn_text_to_words_df(text)
            words = self._clean_up_word_list(words_df)
            encoded_words = np.zeros(len(self.dict_features))
            for word in words:
                if word in self.dict_features:
                    encoded_words[self.dict_features[word]] += 1

            feature_matrix += [encoded_words]
            idx = idx + 1

        return pd.DataFrame(feature_matrix, columns=list(self.dict_features.keys()))


    def fit_and_transform(self, X):

    def _clean_up_word_list(self, word_colums_df):
        #word_colums_df is a dataframe. the columns are the text with splitetd words

        def functions(z):
            if self.punc_rm:
                z = re.sub(r'[^\w\s]', '', str(z))
            if self.lower:
                z = str(z).lower()
            if self.num_rm:
                z = re.sub(r'[0-9]', '', str(z))
            return z

        return word_colums_df.applymap(lambda z: functions(z))


    def _clean_up_word_list_np(self, words_np):

        def functions(z):
            if self.punc_rm:
                z = re.sub(r'[^\w\s]', '', str(z))
                z = re.sub(r'[\n]', '', str(z))
            if self.lower:
                z = str(z).lower()
            if self.num_rm:
                z = re.sub(r'[0-9]', '', str(z))
            return str(z)

        return np.apply_along_axis(functions, 0, words_np)#np.array2string(np.apply_along_axis(functions, 0, words_np))


    def _turn_texts_df_to_word_columns_df(self, texts_df):
        # text_df is pd.DataFram
        all_words = []
        word_columns = []
        texts = []
        for text in texts_df.texts:
            words = self._clean_up_word_list_np(np.str.split(text, sep=' '))
            texts += [words]
            #word_columns.append(words)

        clean_texts_df = pd.DataFrame(texts, columns=['texts'])
        for text in clean_texts_df.texts:
            words = np.str.split(text, sep=' ')
            all_words += words
            word_columns.append(words)

        all_words_df = pd.DataFrame(all_words, columns=['words'])
        text_df = pd.DataFrame(texts, columns=['texts'])#list(map(list, zip(*word_columns))))

        return all_words_df, text_df

    def _turn_text_to_words_df(self, text):
        #text is a string
        words = text.split(' ')
        return pd.DataFrame(words, columns=['words'])
