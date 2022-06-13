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

import time
import pkg_resources
import onnx
import tensorflow as tf
from onnx_tf.backend import prepare


'''
************************
tensorflow-gpu==2.4.0
tensorflow-addons==0.14.0
onnx==1.11.0
onnx-tf==1.9.0
************************
'''


class MobileNetV3ONNXToTensorFlowConverter:
    def __init__(self):
        print('\n*********** MobileNet v3 ONNX To TensorFlow Converter ***********')
        print('onnx version: ', onnx.__version__)
        print('onnx_tf version: ', pkg_resources.get_distribution('onnx_tf').version)
        print('tensorflow version: ', tf.__version__)
        print('*****************************************************************')


    def convert(self, onnx_model_path, tensorflow_model_output_path):
        '''
        Convert

        :param onnx_model_path:
        :param tensorflow_model_output_path:
        :return:
        '''
        self._load_onnx_model(onnx_model_path=onnx_model_path)

        start_time = time.time()
        print('\nStarting convert to tensorflow pb model ...')

        try:
            self._onnx_tf_exporter = prepare(self._onnx_model)
            print(self._onnx_tf_exporter)
            self._onnx_tf_exporter.export_graph(tensorflow_model_output_path)

        except Exception as ex:
            print('Convert to tensorflow model failed.', ex)
            return

        print('Convert to tensorflow model success. Cost time: ', time.time() - start_time, 's.')


    def _load_onnx_model(self, onnx_model_path):
        print('\nStarting load onnx model (', onnx_model_path, ')...')
        start_time = time.time()

        try:
            self._onnx_model = onnx.load(onnx_model_path)

        except Exception as ex:
            print('Load onnx model failed. ', ex)
            self._onnx_tf_exporter = None
            return

        print('Load onnx model success. Cost time: ', time.time() - start_time, 's.')







'''
=============================
Main
=============================
'''
if __name__ == '__main__':
    mobilenet_v3_onnx_to_tensorflow_converter = MobileNetV3ONNXToTensorFlowConverter()
    mobilenet_v3_onnx_to_tensorflow_converter.convert(onnx_model_path='../../../Mask_Detection/models_zoo/mobilenet_v3/onnx/label_6/mask_detection_mobilenet_v3_small_112_label_6_weight_val_acc_0.9840162374730432.onnx',
                                                      tensorflow_model_output_path='../../../Mask_Detection/models_zoo/mobilenet_v3/tensorflow/label_6/mask_detection_mobilenet_v3_small_112_label_6_weight_val_acc_0.9840162374730432')
