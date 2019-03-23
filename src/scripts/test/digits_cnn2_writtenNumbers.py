from preprocessing_utils import Image_Importer, ModelImporter, Preprocessing
import matplotlib.pyplot as plt
import torch
import argparse

if not __name__ == '__main_':
    parser = argparse.ArgumentParser(description='Digits')
    parser.add_argument('--num', default=0, help='image to predict')
    args = parser.parse_args()

    img_imp = Image_Importer('digits')
    img = img_imp.load_image_as_grey(args.num)
    img_array = img_imp.get_image_as_256px_array(args.num)

    dtype = torch.float
    device = torch.device("cpu")

    model_name = 'cnn_digits_2'
    m_importer = ModelImporter('digits')
    model = m_importer.load_nn_model(model_name, 0, 10, 100)

    image = model.reshape_data(torch.tensor(img_array, device=device, dtype=dtype))
    y_test = torch.tensor(int(args.num), device=device, dtype=torch.long)

    y_pred = model(image).argmax(1)

    accuracy_soft = (y_pred == y_test).float().mean()

    print(f'number predicted {y_pred.item()}')

    pre = Preprocessing('digits')

    detected_patterns = model.get_detected_patterns1()
    plt.figure(1, figsize=(20, 10))
    for p in range(model.n_patterns1):
        pattern = detected_patterns[0][p].reshape(detected_patterns.shape[2], detected_patterns.shape[3])
        patern_np = pattern.detach().numpy().reshape(8, 8)
        plt.subplot(2, 3, 1 + p)
        plt.imshow(patern_np, cmap='hot', interpolation='none')
    pre.save_plt_as_image(plt, f'written_number {args.num}_patterns')


    detected_patterns = model.get_detected_patterns2()
    plt.figure(1, figsize=(20, 20))
    for p in range(model.n_patterns2):
        pattern = detected_patterns[0][p].reshape(detected_patterns.shape[2], detected_patterns.shape[3])
        patern_np = pattern.detach().numpy().reshape(4, 4)
        plt.subplot(4, 4, 1 + p)
        plt.imshow(patern_np, cmap='hot', interpolation='none')
    pre.save_plt_as_image(plt, f'written_number {args.num}_patterns2')