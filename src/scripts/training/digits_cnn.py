from train import TrainClassifier


from preprocessing_utils import Preprocessing, ModelExporter
import torch
import argparse
from models import Cnn_Digits

if not __name__ == '__main_':

    parser = argparse.ArgumentParser(description='Digits')
    parser.add_argument('--s_model', default=True, help='save trained model')

    args=parser.parse_args()

    n_classes = 10
    n_epochs = 100

    pre = Preprocessing('digits')
    pre.load_data(filename='train.csv', name='train')

    X_df = pre.get(name='train').drop(columns=['0'])
    y_df = pre.get(name='train')['0']

    dtype = torch.float
    device = torch.device("cpu")

    #name, ch_in = 1, d_out = 10, size_im = [16, 16], n_patterns = 6,
    #kernel_s = 2, dtype = torch.float, device = 'cpu'
    model_name = 'cnn_digits'
    model = Cnn_Digits(model_name)

    learning_rate = 0.0001
    batch_size = 32

    train_classifier = TrainClassifier(model, X_df, y_df)
    trained_model , optimizer, criterion, loss_hist, loss_val_hist = train_classifier.run_train(n_epochs = n_epochs, lr=learning_rate, batch_size=batch_size)
    pre.save_results(loss_hist, loss_val_hist, f'{model_name}')

    if args.s_model:
        m_exporter = ModelExporter('digits')
        m_exporter.save_nn_model(trained_model, optimizer, 0, n_classes, n_epochs, trained_model.get_args())

