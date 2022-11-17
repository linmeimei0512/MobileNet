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
import cv2
import time
import pkg_resources
import numpy as np
from rknn.api import RKNN
import onnx
import tensorflow

from utils.image_utils import Image_transform_RKNN

'''
#################### Python Package Version ####################
conda: mobilenet-v3

onnx            1.11.0
tensorflow-gpu  2.4.0
rknn-toolkit    1.6.1
################################################################
'''

class MobileNetV3RK1808ModleLoader:
    _isInit = False

    def __init__(self, print_log=True):
        print('\n********** MobileNet V3 RK1808 Model Loader **********')
        print('ONNX version: \t\t\t', onnx.__version__)
        print('Tensorflow version: \t', tensorflow.__version__)
        print('RK toolkit version: \t', pkg_resources.get_distribution('rknn-toolkit').version)
        print('******************************************************\n')

        self._init_rknn(print_log)


    def _init_rknn(self, print_log):
        '''
        Initialize RKNN
        '''
        self._rknn = RKNN(verbose=print_log)
        self._rknn.list_devices()


    def load_rknn_model(self, rknn_model_path, device_id=None):
        '''
        Load RKNN model

        Args:
            rknn_model_path:
            device_id:
        '''
        print('\n********************* Load RKNN Model ********************')
        print('Starting load RKNN model \'' + str(rknn_model_path) + '\'...')
        self._isInit = False
        start_time = time.time()

        try:
            # Load rknn model
            result = self._rknn.load_rknn(path=rknn_model_path)
            if result != 0:
                print('Load RKNN model failed. Error result: ', result)
                return

            # Initialize rknn runtime environment
            print('Initialize RKNN runtime environment...')
            if device_id is not None:
                result = self._rknn.init_runtime(target='rk1808', device_id=device_id)
            else:
                result = self._rknn.init_runtime()
            if result != 0:
                print('Initialize RKNN runtime environment failed. Error result: ', result)
                return

            self._isInit = True

        except Exception as e:
            print('Load RKNN model error. ', e)

        finally:
            if self._isInit:
                print('Load RKNN model success. Cost time: ' + str(time.time() - start_time)[0:5] + 's.')

            else:
                print('Load RKNN model fail.')
            print('**********************************************************\n')


    def predict(self, image_path=None, image=None, debug=True):
        '''
        Evaluate rknn model

        Args:
            image_path:
        '''
        if debug:
            print('\n******************* Predict *******************')

        start_time = time.time()
        success = False
        predict_output = None
        softmax_output = None
        output = None
        try:
            if not self._isInit:
                if debug:
                    print('RKNN is not initialize!!')
                return

            if image is None:
                if image_path is None:
                    if debug:
                        print('Image and image path is None.')
                    return

                else:
                    image = cv2.imread(image_path)
            image = Image_transform_RKNN(image)

            # Predict
            predict_output = self._rknn.inference(inputs=[image])
            softmax_output = self._softmax(predict_output[0])
            output = np.argmax(softmax_output[0])
            success = True

        except Exception as e:
            if debug:
                print('Predict rknn model error. ', e)

        finally:
            if debug:
                if success:
                    print('Predict ouptut: ', predict_output)
                    print('Softmax output: ', softmax_output)
                    print('Output: ', output)
                    print('Cost time: ' + str(time.time() - start_time)[0:5] + 's.')
                print('***********************************************\n')

            return predict_output, softmax_output, output


    def _softmax(self, x):
        f_x = np.exp(x) / np.sum(np.exp(x))
        return f_x







'''
=============================
Main
=============================
'''
if __name__ == '__main__':
    np.set_printoptions(linewidth=2000)

    mobilenet_v3_rk1808_model_loader = MobileNetV3RK1808ModleLoader(print_log=True)
    mobilenet_v3_rk1808_model_loader.load_rknn_model(rknn_model_path='../../../Mask_Detection/models_zoo/mobilenet_v3/label_6/rk1808/mask_detection_mobilenet_v3_small_112_label_6_acc_0.984_20220523-recompile-rgb.rknn',
                                                     device_id='TS018083201200011')
    mobilenet_v3_rk1808_model_loader.predict(image_path='../../../Mask_Detection/initialize_image.jpg')