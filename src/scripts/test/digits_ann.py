from preprocessing_utils import Preprocessing, ModelImporter

import torch



if not __name__ == '__main_':

    pre = Preprocessing('digits')
    pre.load_data(filename='test.csv', name='test')

    X_df = pre.get(name='test').drop(columns=['0'])
    y_df = pre.get(name='test')['0']

    dtype = torch.float
    device = torch.device("cpu")

    model_name = 'ann_digits'

    m_importer = ModelImporter('digits')

    model = m_importer.load_nn_model(model_name, 0, 10, 100)

    X_test = model.reshape_data(torch.tensor(X_df.values, device=device, dtype=dtype))
    y_test = torch.tensor(y_df.values, device=device, dtype=torch.long)

    y_pred = model(X_test).argmax(1)

    accuracy_soft = (y_pred == y_test).float().mean()


    print(f'test accuracy {accuracy_soft.item()}')