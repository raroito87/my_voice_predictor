import os
import pandas as pd
from sklearn import preprocessing


class Preprocessing:
    def __init__(self, name):
        self.name = name.lower()
        self.data = {}

        root_dir = os.path.dirname(__file__)
        directory_template = '{root_dir}/../../data/{name}/'
        self.directory = directory_template.format(root_dir=root_dir, name=name)

        if not os.path.exists(self.directory):
            print(f'Creating "{name}" directory for you!')
            os.makedirs(self.directory)

    def load_data(self, filename, filetype='csv', *, name, **kwargs):
        filepath = f'{self.directory}/{filename}'

        function_name = f'read_{filetype}'
        df = getattr(pd, function_name)(filepath, **kwargs)
        self.data[name] = df
        return df

    def save(self, name, filetype='csv', *, index=False, **kwargs):
        filepath = f'{self.directory}/{name}.{filetype}'
        getattr(self.data[name], f'to_{filetype}')(filepath, index=index, **kwargs)

    def cleanup(self, name, *, drop=None, drop_duplicates=False, dropna=None):
        data = self.data[name]

        if drop is not None:
            data = data.drop(columns=drop)

        if drop_duplicates is True:
            data = data.drop_duplicates()

        if dropna is not None:
            if 'axis' not in dropna:
                dropna['axis'] = 1

            data = data.dropna(**dropna)

        self.data['clean'] = data

    def label_encode(self, *, columns):
        if 'clean' not in self.data:
            print('Can not find clean data. Call .cleanup() first.')
            return

        data = self.data['clean']
        encoder = preprocessing.LabelEncoder()
        labels = pd.DataFrame()

        label_index = 0
        for column in columns:
            encoder.fit(data[column])
            label = encoder.transform(data[column])
            labels.insert(label_index, column=column, value=label)
            label_index += 1

        data = data.drop(columns, axis=1)
        data = pd.concat([data, labels], axis=1)
        self.data['clean'] = data

        return data

    def one_hot_encode(self, *, columns):
        if 'clean' not in self.data:
            print('Can not find clean data. Call .cleanup() first.')
            return

        data = self.data['clean']
        categorical = pd.get_dummies(data[columns], dtype='int')
        data = pd.concat([data, categorical], axis=1, sort=False)
        self.data['clean'] = data

        return data

    def get(self, name):
        return self.data[name]

    def set(self, name, value):
        self.data[name] = value
