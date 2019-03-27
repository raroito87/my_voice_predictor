from train import TrainClassifier


from preprocessing_utils import Preprocessing, ModelExporter
import torch
import argparse
from models import LogReg

import matplotlib.pyplot as plt

if not __name__ == '__main_':

    parser = argparse.ArgumentParser(description='Digits')
    parser.add_argument('--s_model', default=True, help='save trained model')

    args=parser.parse_args()

    n_classes = 10
    n_epochs = 200

    pre = Preprocessing('digits')
    pre.load_data(filename='train_encoded.csv', name='train')

    X_df = pre.get(name='train').drop(columns=['0'])
    y_df = pre.get(name='train')['0']

    dtype = torch.float
    device = torch.device("cpu")

    print(len(X_df.columns))

    model_name = 'logreg_digits_encoded'
    model = LogReg(model_name, len(X_df.columns), n_classes)

    learning_rate = 0.0001
    batch_size = 32

    train_classifier = TrainClassifier(model, X_df, y_df)
    trained_model , optimizer, criterion, loss_hist, loss_val_hist, best_param = train_classifier.run_train(n_epochs = n_epochs, lr=learning_rate, batch_size=batch_size)
    pre.save_results(loss_hist, loss_val_hist, f'{model_name}')

    trained_model.load_state_dict(state_dict=best_param)
    trained_model.eval()

    if args.s_model:
        m_exporter = ModelExporter('digits')
        m_exporter.save_nn_model(trained_model, optimizer, 0, n_classes, n_epochs, trained_model.get_args())
