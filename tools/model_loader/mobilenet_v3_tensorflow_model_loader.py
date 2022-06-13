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

import tensorflow as tf
import time
import numpy as np


class MobileNetV3TensorFlowModelLoader:
    _isInit = False

    def __init__(self):
        print('\n********** MobileNet V3 TensorFlow Model Loader **********')
        print('TensorFLow version: \t', tf.__version__)
        print('**********************************************************\n')

    def load_tensorflow_model(self, tensorflow_model_path):
        print('\n*************** Load TensorFlow Model ***************')
        print('Starting load tensorflow model [' + str(tensorflow_model_path) + ']...')
        self._isInit = False
        start_time = time.time()

        try:
            self._tensorflow_model = tf.keras.models.load_model(tensorflow_model_path)
            self._tensorflow_model


        except Exception as e:
            print('Load tensorflow model error. ', e)
            self._tensorflow_model = None


        print('*****************************************************\n')






'''
=============================
Main
=============================
'''
if __name__ == '__main__':
    np.set_printoptions(linewidth=2000, precision=20)

    mobilenet_v3_tensorflow_model_loader = MobileNetV3TensorFlowModelLoader()
    mobilenet_v3_tensorflow_model_loader.load_tensorflow_model(tensorflow_model_path='../../../Mask_Detection/models_zoo/mobilenet_v3/label_6/tensorflow/mask_detection_mobilenet_v3_small_112_label_6_acc_0.984_20220523')