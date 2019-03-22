import os
import torch

class ModelExporter:
    def __init__(self, name):
        self.name = name.lower()
        self.data = {}

        root_dir = os.path.dirname(__file__)
        directory_template = '{root_dir}/../../data/{name}/models/'
        self.directory = directory_template.format(root_dir=root_dir, name=name)

        if not os.path.exists(self.directory):
            print (f'created directory {self.directory}')
            os.makedirs(self.directory)

    def save_nn_model(self, model, optimizer, n_features, n_classes, n_epochs=0):
        the_dict = model.state_dict()
        the_dict['model_class'] = type(model).__name__
        the_dict['optimizer_class']= type(optimizer).__name__
        the_dict['args'] = [model.name, n_features, n_classes]

        file_name = f'{model.name}_{n_features}_{n_classes}_{n_epochs}.pt'
        torch.save(the_dict, self.directory + file_name)
