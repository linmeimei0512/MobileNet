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
import time
import numpy as np
import cv2
import torch
import argparse
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from mobilenet_v3.pytorch.models.mobilenet_v3 import MobileNetV3Model, mobilenet_v3_small, mobilenet_v3_large

class MobileNetV3PyTorchModelLoader:
    _device = None
    _mobilenet_v3_model = None
    _model = None
    _model_path = None
    _model_weight_path = None
    _num_classes = 0

    _initialize_image_path = None

    def __init__(self, mobilenet_v3_model:MobileNetV3Model, model_path=None, model_weight_path=None, num_classes=0, initialize_image_path=None):
        '''
        Constructor

        Args:
            mobilenet_v3_model:
            model_path:
            model_weight_path:
            initialize_image_path:
        '''
        print('\n********** MobileNet v3 PyTorch Model Loader **********')
        print('torch version: ', str(torch.__version__))
        print('')
        print('mobilenet v3 model: ', mobilenet_v3_model)
        print('model path: ', model_path)
        print('model weight path: ', model_weight_path)
        print('number of classes: ', num_classes)
        print('initialize image path: ', initialize_image_path)
        print('*******************************************************\n')
        self._mobilenet_v3_model = mobilenet_v3_model
        self._model_path = model_path
        self._model_weight_path = model_weight_path
        self._num_classes = num_classes
        self._initialize_image_path = initialize_image_path

        self._init_pytorch_device()
        self._init_model()
        self._warn_up()


    def predict(self, image=None, image_path=None):
        '''
        Predict

        Args:
            image:
            image_path:

        Returns:
            class
            confidence
        '''
        data_transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        predict_class = None
        predict_confidence = None
        start_time = time.time()

        try:
            if image is not None:
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image = Image.open(image_path)

            image = data_transform(image)
            image = torch.unsqueeze(image, dim=0)

            output = self._model(image.to(self._device))
            output = torch.squeeze(output).cpu()
            output = torch.softmax(output, dim=0)
            predict_class = torch.argmax(output).numpy()
            predict_confidence = output[predict_class].detach().numpy()

        except Exception as ex:
            print('Predict error. ', ex)

        finally:
            print('Predict success. Cost time: ' + str(time.time() - start_time)[0:4] + 's.')
            return predict_class, predict_confidence



    def _init_pytorch_device(self):
        '''
        Initialize pytorch device
        '''
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('\nUsing {} device.'.format(self._device))


    def _init_model(self):
        '''
        Initialize model
        '''
        start_time = time.time()

        try:
            # Load model
            if self._model_path is not None:
                if not os.path.isfile(self._model_path):
                    print('MobileNet v3 model (', self._model_path, ') is not exist.')

                else:
                    print('Starting load mobilenet v3 pytorch model (', self._model_path, ')...')
                    self._model = torch.load(self._model_path)

            # Load weight
            else:
                if not os.path.isfile(self._model_weight_path):
                    print('MobileNet v3 weight (', self._model_weight_path, ') is not exist.')

                else:
                    print('Create mobilenet v3 model ', self._mobilenet_v3_model)
                    if self._mobilenet_v3_model is MobileNetV3Model.SMALL:
                        self._model = mobilenet_v3_small(num_classes=self._num_classes).to(self._device)
                    elif self._mobilenet_v3_model is MobileNetV3Model.LARGE:
                        self._model = mobilenet_v3_large(num_classes=self._num_classes).to(self._device)

                    print('Starting load mobilenet v3 pytorch weight (', self._model_weight_path, ')...')
                    self._model.load_state_dict(torch.load(self._model_weight_path, map_location=self._device))
                    self._model.eval()

        except Exception as ex:
            print('Load model error. ', ex)
            self._model = None

        finally:
            if self._model is None:
                print('Load model fail.')

            else:
                print('Load model success. Cost time: ', str(time.time() - start_time)[0:4], 's.')


    def _warn_up(self):
        '''
        Warn up model
        '''
        if self._initialize_image_path is not None and os.path.isfile(self._initialize_image_path):
            print('Warning up model...')
            start_time = time.time()

            try:
                predict_class, predict_confidence = self.predict(image_path=self._initialize_image_path)

            except Exception as ex:
                print('Warn up error. ', ex)

            finally:
                if predict_class is not None and predict_confidence is not None:
                    print('Warn up finish. Warn up time: ' + str(time.time() - start_time)[0:4] + 's.')



'''
=============================
Default
=============================
'''
model_size = 0
model_path = '../../models_zoo/mobilenet_v3_small/pytorch/flower_classification_val_acc_0.8791208791208791.pth'
model_weight_path = 'None'
num_classes = 5
initialize_image_path = '../../datasets/flower/flower_photos/daisy/54377391_15648e8d18.jpg'
predict_image_path = '../../datasets/flower/flower_photos/daisy/54377391_15648e8d18.jpg'


'''
=============================
Main
=============================
'''
if __name__ == '__main__':
    np.set_printoptions(linewidth=2000)

    parser = argparse.ArgumentParser(description='MobileNet v3 PyTorch Predict')
    parser.add_argument('--model_size', type=int, default=model_size, help='0: use small, 1: use large')
    parser.add_argument('--model_path', type=str, default=model_path, help='model path')
    parser.add_argument('--model_weight_path', type=str, default=model_weight_path, help='model weight path')
    parser.add_argument('--num_classes', type=int, default=num_classes, help='number of class')
    parser.add_argument('--initialize_image_path', type=str, default=initialize_image_path, help='initialize image path')
    parser.add_argument('--predict_image_path', type=str, default=predict_image_path, help='predict image path')
    args = parser.parse_args()


    if args.model_size == 0:
        mobilenet_v3_model = MobileNetV3Model.SMALL
    elif args.model_size == 1:
        mobilenet_v3_model = MobileNetV3Model.LARGE
    else:
        mobilenet_v3_model = None

    if args.model_path == 'None':
        model_path = None
    else:
        model_path = args.model_path

    if args.model_weight_path == 'None':
        model_weight_path = None
    else:
        model_weight_path = args.model_weight_path

    num_classes = args.num_classes
    initialize_image_path = args.initialize_image_path
    predict_image_path = args.predict_image_path


    mobilenet_v3_pytorch_model_loader = MobileNetV3PyTorchModelLoader(mobilenet_v3_model=mobilenet_v3_model,
                                                                      model_path=model_path,
                                                                      model_weight_path=model_weight_path,
                                                                      num_classes=num_classes,
                                                                      initialize_image_path=initialize_image_path)

    # mobilenet_v3_pytorch_model_loader = MobileNetV3PyTorchModelLoader(mobilenet_v3_model=MobileNetV3Model.SMALL,
    #                                                                   model_weight_path='../../models_zoo/mobilenet_v3_small/pytorch/flower_classification_weight_val_acc_0.8791208791208791.pth',
    #                                                                   num_classes=5,
    #                                                                   initialize_image_path='../../datasets/flower/flower_photos/daisy/54377391_15648e8d18.jpg')

    predict_class, predict_confidence = mobilenet_v3_pytorch_model_loader.predict(image_path=predict_image_path)
    print('\nclass: ', predict_class)
    print('confidence: ', predict_confidence)