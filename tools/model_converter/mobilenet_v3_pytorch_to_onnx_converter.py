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

import torch
import time
import numpy as np
from torchsummary import summary

class MobileNetV3PyTorchToONNXConverter:
    def __init__(self):
        self._init_device()

        print('\n*********** MobileNet v3 PyTorch To ONNX Converter ***********')
        print('torch version: ', torch.__version__)
        print('torch opset version: ', torch.onnx.constant_folding_opset_versions)
        print('torch cuda is available: ', torch.cuda.is_available())
        print('torch device: ', self._device)
        print('**************************************************************')


    def _init_device(self):
        '''
        Initialize torch device
        '''
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def convert(self, pytorch_model_path=None, pytorch_weight_path=None, num_classes=None, input_shape=None, onnx_model_output_path=None):
        '''
        Convert

        :param pytorch_model_path:
        :param pytorch_weight_path:
        :param input_shape:
        :param onnx_model_output_path:
        :return:
        '''
        if pytorch_model_path is not None:
            self._load_pytorch_model(pytorch_model_path=pytorch_model_path, input_shape=input_shape)

        elif pytorch_weight_path is not None:
            self._load_pytorch_model_weight(pytorch_weight_path=pytorch_weight_path, num_classes=num_classes, input_shape=input_shape)

        else:
            print('Pytorch model path or weight path is None.')
            return


        if self._pytorch_model is not None:
            print('\nStarting convert to ONNX model ...')
            start_time = time.time()
            try:
                image = torch.empty(size=(1, *input_shape), dtype=torch.float, device=self._device)
                torch.onnx.export(model=self._pytorch_model.module,
                                  args=image,
                                  f=onnx_model_output_path,
                                  verbose=False,
                                  input_names=['input'],
                                  output_names=['output'],
                                  opset_version=11)

            except Exception as ex:
                print('Convert to ONNX model failed. ', ex)
                return

            print('Convert to ONNX model success. Cost time: ', time.time() - start_time, 's.')


    def _load_pytorch_model(self, pytorch_model_path, input_shape):
        print('\nStarting load pytorch model (', pytorch_model_path, ')...')
        start_time = time.time()

        try:
            self._pytorch_model = torch.load(pytorch_model_path)

        except Exception as ex:
            print('Load pytorch model failed. ', ex)
            self._pytorch_model = None
            return

        print('Load pytorch model success. Cost time: ', time.time() - start_time, 's.')


    def _load_pytorch_model_weight(self, pytorch_weight_path, num_classes, input_shape):
        print('\nStarting load pytorch weight (', pytorch_weight_path, ')...')
        start_time = time.time()

        try:
            from mobilenet_v3.pytorch.models.mobilenet_v3 import mobilenet_v3_small
            self._pytorch_model = mobilenet_v3_small(num_classes=num_classes).to(self._device)
            self._pytorch_model.load_state_dict(torch.load(pytorch_weight_path))

            summary(self._pytorch_model, input_size=(3, 224, 224))

        except Exception as ex:
            print('Load pytorch model failed. ', ex)
            self._pytorch_model = None
            return

        print('Load pytorch weight success. Cost time: ', time.time() - start_time, 's.')


'''
=============================
Main
=============================
'''
if __name__ == '__main__':
    np.set_printoptions(linewidth=2000, precision=20)

    mobilenet_v3_pytorch_to_onnx_converter = MobileNetV3PyTorchToONNXConverter()
    # mobilenet_v3_pytorch_to_onnx_converter.convert(pytorch_weight_path='../../models/mobilenet_v3/PyTorch/liveness_detection_mobilenet_v3_small_weight_112.pth',
    #                                                input_shape=(3, 112, 112),
    #                                                onnx_model_output_path='../../models/mobilenet_v3/ONNX/liveness_detection_mobilenet_v3_small.onnx')
    mobilenet_v3_pytorch_to_onnx_converter.convert(pytorch_model_path='../../../Liveness_Detection/models_zoo/mobilenet_v3/PyTorch/0905/01/ir_liveness_detection_mobilenet_v3_small_112_val_acc_0.9996203492786636.pth',
                                                   input_shape=(3, 112, 112),
                                                   onnx_model_output_path='../../../Liveness_Detection/models_zoo/mobilenet_v3/ONNX/0905/01/ir_liveness_detection_mobilenet_v3_small_112_val_acc_0.9996203492786636.onnx')
