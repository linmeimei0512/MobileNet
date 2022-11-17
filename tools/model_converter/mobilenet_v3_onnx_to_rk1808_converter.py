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
import time
import pkg_resources
import numpy as np
from rknn.api import RKNN
import onnx
import tensorflow
import cv2

from utils.image_utils import Image_transform_RKNN

'''
#################### Python Package Version ####################
conda: mobilenet-v3

onnx            1.11.0
tensorflow-gpu  2.4.0
rknn-toolkit    1.6.1
################################################################
'''

class ONNXToRK1808Converter:
    _convert_success = False
    _onnx_model_path = ''
    _rknn_model_output_path = ''

    def __init__(self, print_log=True):
        print('\n********** ONNX To RK1808 Converter **********')
        print('ONNX version: \t\t\t', onnx.__version__)
        print('Tensorflow version: \t', tensorflow.__version__)
        print('RK toolkit version: \t', pkg_resources.get_distribution('rknn-toolkit').version)
        print('**********************************************\n')

        self._init_rknn(print_log)


    def _init_rknn(self, print_log):
        '''
        Initialize RKNN
        '''
        self._rknn = RKNN(verbose=print_log)
        self._rknn.list_devices()


    def release(self):
        '''
        Release RKNN
        '''
        self._rknn.release()
        self._rknn = None


    def create_quantized_dataset_text(self, quantized_dataset_path):
        '''
        Create quantized dataset text

        Args:
            quantized_dataset_path:
        '''
        print('\n')
        print('Create quantized dataset...')

        image_list = os.listdir(quantized_dataset_path)
        with open(os.path.join(quantized_dataset_path, 'dataset.txt'), 'w') as file:
            for image_name in image_list:
                image = cv2.imread(os.path.join(quantized_dataset_path, image_name))
                image = Image_transform_RKNN(image)
                print(image.shape)
                print(type(image))
                np.save(os.path.join(quantized_dataset_path, image_name.replace('.jpg', '.npy')), image)
                file.write(os.path.join(quantized_dataset_path, image_name.replace('.jpg', '.npy')) + '\n')

        print('Create quantized dataset text success. Save as dataset.txt.')


    def convert(self, onnx_model_path, rknn_model_output_path, do_quantization=False, quantized_dataset_path=None, quantized_dtype='dynamic_fixed_point-16', optimization_level=3, batch_size=200, pre_compile=False):
        '''
        Convert

        Args:
            onnx_model_path:
            rknn_model_output_path:
            do_quantization:
            quantized_dataset_path:
            quantized_dtype:
            optimization_level:
            batch_size:
            pre_compile:
        '''
        print('\n******************* Convert ONNX To RKNN *******************')
        print('ONNX model path: ', onnx_model_path)
        print('RKNN model output path: ', rknn_model_output_path)
        print('Do quantization: ', do_quantization)
        print('Quantized dataset path: ', quantized_dataset_path)
        print('Quantized dtype: ', quantized_dtype)
        print('Optimization level: ', optimization_level)
        print('Batch size: ', batch_size)
        print('Pre compile: ', pre_compile)
        self._onnx_model_path = onnx_model_path
        self._rknn_model_output_path = rknn_model_output_path


        start_time = time.time()
        print('\nStarting convert to rknn model...')

        try:
            # Set config
            print('Set rknn config ...')
            self._rknn.config(quantized_dtype=quantized_dtype, optimization_level=optimization_level, batch_size=batch_size, reorder_channel='0 1 2')
            print('Set rknn config success.')

            # Load onnx
            print('Load onnx model...')
            result = self._rknn.load_onnx(model=onnx_model_path)
            if result != 0:
                print('Load onnx model failed! Error result: ', result)
                return
            print('Load onnx model success.')

            # Build rknn model
            print('Build rknn model...')
            result = self._rknn.build(do_quantization=do_quantization, dataset=quantized_dataset_path, pre_compile=pre_compile)
            if result != 0:
                print('Build rknn model failed! Error result: ', result)
                return
            print('Build rknn model success.')

            # Export rknn model
            print('Export rknn model...')
            result = self._rknn.export_rknn(export_path=rknn_model_output_path)
            if result != 0:
                print('Export rknn model failed!. Error result: ', result)
                return
            print('Export rknn model success.')
            self._convert_success = True

        except Exception as e:
            print('Convert error. ', e)

        finally:
            if self._convert_success:
                print('Convert to rknn model success, saved as ' + str(rknn_model_output_path) + '. Cost time: ' + str(time.time() - start_time))
            else:
                print('Convert to rknn model fail.')
            print('************************************************************\n')


    def evaluate(self, device_id=None):
        '''
        Evaluate rknn model
        '''
        print('\n******************* Evaluate RKNN Model *******************')
        try:
            if not self._convert_success:
                print('RKNN is not convert success!!')
                return

            if device_id is not None:
                self._rknn.init_runtime(target='rk1808', device_id=device_id)
            else:
                self._rknn.init_runtime()
            self._rknn.eval_perf()

        except Exception as e:
            print('Evaluate rknn model error. ', e)

        finally:
            print('***********************************************************\n')


    def verify_accuracy(self, image_path, device_id=None):
        # ONNX
        from tools.model_loader.mobilenet_v3_onnx_model_loader import MobileNetV3ONNXModelLoader
        mobilenet_v3_onnx_model_loader = MobileNetV3ONNXModelLoader()
        mobilenet_v3_onnx_model_loader.load_onnx_model(onnx_model_path=self._onnx_model_path)
        # mobilenet_v3_onnx_model_loader.load_onnx_model(onnx_model_path='../../../Mask_Detection/models_zoo/mobilenet_v3/label_6/onnx/mask_detection_mobilenet_v3_small_112_label_6_acc_0.984_20220523.onnx')
        onnx_predict_output, onnx_softmax_output, onnx_output = mobilenet_v3_onnx_model_loader.predict(image_path=image_path, debug=False)

        # RKNN
        from tools.model_loader.mobilenet_v3_rk1808_model_loader import MobileNetV3RK1808ModleLoader
        mobilenet_v3_rk1808_model_loader = MobileNetV3RK1808ModleLoader(print_log=True)
        mobilenet_v3_rk1808_model_loader.load_rknn_model(rknn_model_path=self._rknn_model_output_path,
                                                         device_id=device_id)
        # mobilenet_v3_rk1808_model_loader.load_rknn_model(rknn_model_path='../../../Mask_Detection/models_zoo/mobilenet_v3/label_6/rk1808/mask_detection_mobilenet_v3_small_112_label_6_acc_0.984_20220523-quant-16-recompile-rgb.rknn',
        #                                                  device_id=device_id)
        rknn_predict_output, rknn_softmax_output, rknn_output = mobilenet_v3_rk1808_model_loader.predict(image_path=image_path, debug=False)


        print('\n********** Verify Accuracy **********')
        from scipy.spatial.distance import cosine

        acc = 1 - cosine(onnx_predict_output, rknn_predict_output)
        print('Accuracy: ' + str(acc * 100)[0:5] + '%')
        print('*************************************\n')








'''
=============================
Main
=============================
'''
if __name__ == '__main__':
    np.set_printoptions(linewidth=2000)

    onnx_to_rk1808_converter = ONNXToRK1808Converter(print_log=True)
    # onnx_to_rk1808_converter.create_quantized_dataset_text(quantized_dataset_path='./rknn_convert_dataset')

    onnx_to_rk1808_converter.convert(onnx_model_path='../../../Mask_Detection/models_zoo/mobilenet_v3/label_6/onnx/mask_detection_mobilenet_v3_small_112_label_6_acc_0.984_20220523.onnx',
                                     rknn_model_output_path='../../../Mask_Detection/models_zoo/mobilenet_v3/label_6/rk1808/mask_detection_mobilenet_v3_small_112_label_6_acc_0.984_20220523-recompile-rgb.rknn',
                                     do_quantization=False,
                                     quantized_dataset_path='./rknn_convert_dataset/dataset.txt',
                                     quantized_dtype='dynamic_fixed_point-16',
                                     optimization_level=3,
                                     batch_size=200,
                                     pre_compile=True)
    onnx_to_rk1808_converter.evaluate(device_id='TS018083201200011')
    onnx_to_rk1808_converter.release()

    time.sleep(1)
    onnx_to_rk1808_converter.verify_accuracy(image_path='../../../Mask_Detection/initialize_image.jpg', device_id='TS018083201200011')
