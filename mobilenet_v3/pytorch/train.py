#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#   Copyright (C) 2022 MSI-FUNTORO
#
#   Licensed under the MSI-FUNTORO License, Version 1.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       https://www.funtoro.com/global/
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import sys
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms, datasets
from torchtoolbox.tools import mixup_criterion, mixup_data

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from mobilenet_v3.pytorch.models.mobilenet_v3 import MobileNetV3Model
from mobilenet_v3.pytorch.models.mobilenet_v3 import mobilenet_v3_small, mobilenet_v3_large

class Train:
    _device = None
    _mobilenet_v3_model = None
    _model = None

    _batch_size = 128
    _lr = 0.0001
    _epochs = 1
    _best_acc = 0

    # Retrain weight
    _use_pretrain_weight = False
    _pretrain_weight_path = None

    # Datasets
    _datasets_path = ''

    # Datasets pytorch transform
    _data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # Save model
    _save_class_json_file_path = ''
    _save_model_path = ''
    _save_model_name = ''
    _save_weight_name = ''


    def __init__(self, mobilenet_v3_model:MobileNetV3Model, batch_size, epochs, pretrain_weight_path, datasets_path, save_class_json_file_path, save_model_path, save_model_name, save_weight_name):
        '''
        Constructor

        Args:
            mobilenet_v3_model:
            epochs:
            pretrain_weight_path:
            datasets_path:
            save_class_json_file_path:
            save_model_path:
            save_model_name:
            save_weight_name:
        '''
        print('\n********** MobileNet v3 PyTorch Train **********')
        print('torch version: ', str(torch.__version__))
        print('')
        print('mobilenet v3 model: ', mobilenet_v3_model)
        print('batch size: ', batch_size)
        print('epochs: ', epochs)
        print('pretrain weight path: ', pretrain_weight_path)
        print('datasets path: ', datasets_path)
        print('save class json file path: ', save_class_json_file_path)
        print('save model path: ', save_model_path)
        print('save model name: ', save_model_name)
        print('save weight name: ', save_weight_name)
        print('************************************************\n')
        self._mobilenet_v3_model = mobilenet_v3_model
        self._batch_size = batch_size
        self._epochs = epochs
        self._pretrain_weight_path = pretrain_weight_path
        self._datasets_path = datasets_path
        self._save_class_json_file_path = save_class_json_file_path
        self._save_model_path = save_model_path
        self._save_model_name = save_model_name
        self._save_weight_name = save_weight_name

        self._init_pytorch_device()
        self._init_number_of_workers()
        self._init_train_datasets()
        self._init_validate_datasets()
        self._init_class()
        self._create_model()


    def train(self):
        '''
        Train
        '''
        loss_function_criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self._model.parameters(), lr=self._lr)
        cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-9)

        for epoch in range(1, self._epochs + 1):
            self._train(loss_function_criterion, optimizer, epoch)
            cosine_schedule.step()
            self._val(loss_function_criterion)

        if self._save_model_name is not None:
            os.rename(os.path.join(self._save_model_path, self._save_model_name),
                      os.path.join(self._save_model_path, str(self._save_model_name).replace('.pth', '') + '_val_acc_' + str(self._best_acc) + '.pth'))
        if self._save_weight_name is not None:
            os.rename(os.path.join(self._save_model_path, self._save_weight_name),
                      os.path.join(self._save_model_path, str(self._save_weight_name).replace('.pth', '') + '_val_acc_' + str(self._best_acc) + '.pth'))


    def _init_pytorch_device(self):
        '''
        Initialize pytorch device
        '''
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Using {} device.'.format(self._device))


    def _init_number_of_workers(self):
        '''
        Initialize number of workers
        '''
        self._nw = min([os.cpu_count(), self._batch_size if self._batch_size > 1 else 0, 8])
        print('Using {} dataloader workers every process'.format(self._nw))


    def _init_train_datasets(self):
        '''
        Initialize train datasets
        '''
        train_datasets_path = os.path.join(self._datasets_path, 'train')
        assert os.path.exists(self._datasets_path), "{} path does not exist.".format(self._datasets_path)
        assert os.path.exists(train_datasets_path), "{} path does not exist.".format(train_datasets_path)
        print('\nTraining datasets image path: ', train_datasets_path)

        self._train_datasets = datasets.ImageFolder(root=train_datasets_path,
                                                    transform=self._data_transform["train"])
        self._train_datasets_number = len(self._train_datasets)

        # Train loader
        self._train_loader = torch.utils.data.DataLoader(self._train_datasets,
                                                         batch_size=self._batch_size, shuffle=True,
                                                         num_workers=self._nw)
        print('Using {} images for training.'.format(self._train_datasets_number))


    def _init_validate_datasets(self):
        '''
        Initialize validate datasets
        '''
        validate_datasets_path = os.path.join(self._datasets_path, 'val')
        assert os.path.exists(validate_datasets_path), "{} path does not exist.".format(validate_datasets_path)
        print('\nValidate datasets image path: ', validate_datasets_path)

        self._validate_datasets = datasets.ImageFolder(root=validate_datasets_path,
                                                       transform=self._data_transform["val"])
        self._validate_datasets_number = len(self._validate_datasets)

        # Validate loader
        self._validate_loader = torch.utils.data.DataLoader(self._validate_datasets,
                                                            batch_size=self._batch_size, shuffle=False,
                                                            num_workers=self._nw)
        print('Using {} images for validation.'.format(self._validate_datasets_number))


    def _init_class(self):
        '''
        Initialize datasets class
        '''
        self._class_list = self._train_datasets.class_to_idx
        print('\nClass list: ', self._class_list)
        print('Save to ', self._save_class_json_file_path)

        # Write to json file
        if self._save_class_json_file_path is None or self._save_class_json_file_path == '':
            return

        class_dict = dict((val, key) for key, val in self._class_list.items())
        json_str = json.dumps(class_dict, indent=4)
        with open(self._save_class_json_file_path, 'w') as json_file:
            json_file.write(json_str)


    def _create_model(self):
        '''
        Create model
        '''
        print('\nCreate model: ', self._mobilenet_v3_model)
        if self._mobilenet_v3_model is MobileNetV3Model.SMALL:
            self._model = mobilenet_v3_small(num_classes=len(self._class_list))

        elif self._mobilenet_v3_model is MobileNetV3Model.LARGE:
            self._model = mobilenet_v3_large(num_classes=len(self._class_list))

        if self._pretrain_weight_path is not None:
            assert os.path.exists(self._pretrain_weight_path), "file {} dose not exist.".format(self._pretrain_weight_path)
            print('Load pretrain weight: ', self._pretrain_weight_path)
            pre_weights = torch.load(self._pretrain_weight_path, map_location='cpu')

            # delete classifier weights
            pre_dict = {k: v for k, v in pre_weights.items() if self._model.state_dict()[k].numel() == v.numel()}
            missing_keys, unexpected_keys = self._model.load_state_dict(pre_dict, strict=False)

            # freeze features weights
            for param in self._model.features.parameters():
                param.requires_grad = False

        self._model.to(self._device)
        # self._model = nn.DataParallel(self._model, device_ids=[0, 1, 2, 3])
        print('')


    def _train(self, loss_function_criterion, optimizer, epoch):
        '''
        Train

        Args:
            loss_function_criterion:
            optimizer:
            epoch:
        '''
        self._model.train()
        sum_loss = 0
        alpha = 0.2
        total_num = len(self._train_loader.dataset)
        # print(total_num, len(self._train_loader))

        for batch_idx, (data, target) in enumerate(self._train_loader):
            data, target = data.to(self._device), target.to(self._device)
            data, labels_a, labels_b, lam = mixup_data(data, target, alpha)
            optimizer.zero_grad()

            output = self._model(data)

            loss = mixup_criterion(criterion=loss_function_criterion, pred=output, y_a=labels_a, y_b=labels_b, lam=lam)
            loss.backward()
            optimizer.step()
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            print_loss = loss.data.item()
            sum_loss += print_loss
            if (batch_idx + 1) % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}\tLR:{:.9f}'.format(
                    epoch, (batch_idx + 1) * len(data), len(self._train_loader.dataset), 100. * (batch_idx + 1) / len(self._train_loader), loss.item(), lr))


    def _val(self, loss_function_criterion):
        '''
        Valuate

        Args:
            loss_function_criterion:
        '''
        self._model.eval()
        test_loss = 0
        correct = 0
        total_num = len(self._validate_loader.dataset)

        with torch.no_grad():
            for data, target in self._validate_loader:
                data, target = Variable(data).to(self._device), Variable(target).to(self._device)

                output = self._model(data)

                loss = loss_function_criterion(output, target)
                _, pred = torch.max(output.data, 1)
                correct += torch.sum(pred == target)
                print_loss = loss.data.item()
                test_loss += print_loss

            correct = correct.data.item()
            acc = correct / total_num
            avgloss = test_loss / len(self._validate_loader)

            if acc > self._best_acc:
                self._best_acc = acc
                if self._save_model_name is not None:
                    torch.save(self._model, os.path.join(self._save_model_path, self._save_model_name))
                if self._save_weight_name is not None:
                    torch.save(self._model.state_dict(), os.path.join(self._save_model_path, self._save_weight_name))

            print('\nVal set: Average loss: {:.4f}, \tAccuracy: {}/{} ({:.2f}%), \tBest accuracy: ({:.2f}%)\n'.format(
                avgloss, correct, len(self._validate_loader.dataset), 100 * acc, 100 * self._best_acc))





'''
=============================
Default
=============================
'''
model_size = 0
batch_size = 128
epochs = 1
pretrain_weight_path = '../../models_zoo/mobilenet_v3_small/pytorch/mobilenet_v3_small-047dcff4.pth'
datasets_path = '../../datasets/flower'
save_class_json_file_path = '../../datasets/flower/flower_class.json'
save_model_path = '../../models_zoo/mobilenet_v3_small/pytorch'
save_model_name = 'flower_classification.pth'
save_weight_name = 'flower_classification_weight.pth'


'''
=============================
Main
=============================
'''
if __name__ == '__main__':
    np.set_printoptions(linewidth=2000)

    parser = argparse.ArgumentParser(description='MobileNet v3 PyTorch Training')
    parser.add_argument('--model_size', type=int, default=model_size, help='0: use small, 1: use large')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='training batch size')
    parser.add_argument('--epochs', type=int, default=epochs, help='training epochs')
    parser.add_argument('--pretrain_weight_path', type=str, default=pretrain_weight_path, help='pre-train weight for training')
    parser.add_argument('--datasets_path', type=str, default=datasets_path, help='datasets path for training')
    parser.add_argument('--save_class_json_file_path', type=str, default=save_class_json_file_path, help='save class json file path')
    parser.add_argument('--save_model_path', type=str, default=save_model_path, help='save model path')
    parser.add_argument('--save_model_name', type=str, default=save_model_name, help='save model name')
    parser.add_argument('--save_weight_name', type=str, default=save_weight_name, help='save model weight name')
    args = parser.parse_args()


    if args.model_size == 0:
        mobilenet_v3_model = MobileNetV3Model.SMALL
    elif args.model_size == 1:
        mobilenet_v3_model = MobileNetV3Model.LARGE
    else:
        mobilenet_v3_model = None

    batch_size = args.batch_size
    epochs = args.epochs
    pretrain_weight_path = args.pretrain_weight_path
    datasets_path = args.datasets_path
    save_class_json_file_path = args.save_class_json_file_path
    save_model_path = args.save_model_path
    save_model_name = args.save_model_name
    save_weight_name = args.save_weight_name



    train = Train(mobilenet_v3_model=mobilenet_v3_model,
                  batch_size=batch_size,
                  epochs=epochs,
                  pretrain_weight_path=pretrain_weight_path,
                  datasets_path=datasets_path,
                  save_class_json_file_path=save_class_json_file_path,
                  save_model_path=save_model_path,
                  save_model_name=save_model_name,
                  save_weight_name=save_weight_name)
    train.train()

