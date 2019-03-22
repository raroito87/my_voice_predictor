from preprocessing_utils import Preprocessing
import torch
import argparse


if not __name__ == '__main_':
    parser = argparse.ArgumentParser(description='IMDBData')
    parser.add_argument('--n_feat', default=1000, help='number of features')
    args = parser.parse_args()
    n_feat = args.n_feat

    pre = Preprocessing('IMDB')
    pre.load_data(filename=f'test_data_{n_feat}.csv', name='test_data')

    X_test_df = pre.get(name='test_data').drop(columns=['target'])
    y_test_df = pre.get(name='test_data')['target']

    dtype = torch.float
    device = torch.device("cpu")
    X_test = torch.tensor(X_test_df.values, device=device, dtype=dtype)
    y_test = torch.tensor(y_test_df.values, device=device, dtype=torch.long)

    n_features = X_test.size()[1]
    name = f'logreg_{n_features}.pt'
    model = pre.load_model(name)

    y_pred = model(X_test).argmax(1)

    accuracy_soft = (y_pred == y_test).float().mean()

    #results = (y_test == y_pred)

    #print(torch.sum(results).item()/len(results))
    print(accuracy_soft.item())