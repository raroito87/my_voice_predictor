from app import Preprocessing
from sklearn.model_selection import train_test_split


if not __name__ == '__main_':

    pre = Preprocessing('digits')
    kwarg = {'header':None, 'sep':' '}
    pre.load_data(filename='zip.train', name='raw', **kwarg)

    pre.cleanup(name='raw', drop_duplicates=True, dropna={'axis': 1, 'thresh':2})

    #data = pre.get('clean')
    #data[0] = data[0].apply(lambda  x: str(x))
    #pre.one_hot_encode(columns=[0])

    #data = pre.get('clean').drop(columns=[0])
    #pre.set(name='clean', value=data)
    #pre.save(name='clean')

    print(pre.get('clean').head())

    #classes = ['0_0.0', '0_1.0', '0_2.0', '0_3.0', '0_4.0', '0_5.0', '0_6.0', '0_7.0', '0_8.0', '0_9.0']
    X = pre.get('clean').drop(columns=[0])
    y = pre.get('clean')[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    train = X_train.join(y_train)
    test = X_test.join(y_test)

    pre.set(name='train', value=train)
    pre.set(name='test', value=test)

    pre.save(name='train')
    pre.save(name='test')
