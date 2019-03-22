from preprocessing_utils import Preprocessing
from sklearn.model_selection import train_test_split


if not __name__ == '__main_':

    pre_train = Preprocessing('digits')
    kwarg = {'header':None, 'sep':' '}
    pre_train.load_data(filename='zip.train', name='raw', **kwarg)

    pre_train.cleanup(name='raw', drop_duplicates=True, dropna={'axis': 1, 'thresh':2})

    print(pre_train.get('clean').head())

    #classes = ['0_0.0', '0_1.0', '0_2.0', '0_3.0', '0_4.0', '0_5.0', '0_6.0', '0_7.0', '0_8.0', '0_9.0']
    X = pre_train.get('clean').drop(columns=[0])
    y = pre_train.get('clean')[0]

    pre_train.set(name='train', value=pre_train.get('clean'))

    pre_train.save(name='train')

    pre_test = Preprocessing('digits')
    kwarg = {'header':None, 'sep':' '}
    pre_test.load_data(filename='zip.test', name='raw', **kwarg)

    pre_test.cleanup(name='raw', drop_duplicates=True, dropna={'axis': 1, 'thresh':2})


    print(pre_test.get('clean').head())

    #classes = ['0_0.0', '0_1.0', '0_2.0', '0_3.0', '0_4.0', '0_5.0', '0_6.0', '0_7.0', '0_8.0', '0_9.0']
    X = pre_test.get('clean').drop(columns=[0])
    y = pre_test.get('clean')[0]

    pre_test.set(name='test', value=pre_test.get('clean'))

    pre_test.save(name='test')
