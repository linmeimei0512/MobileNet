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

import onnxruntime
import time
import numpy as np
import cv2

'''
#################### Python Package Version ####################
conda: mobilenet-v3

onnxruntime-gpu     1.9.0
################################################################
'''

class MobileNetV3ONNXModelLoader:
    _isInit = False

    def __init__(self):
        print('\n********** MobileNet V3 ONNX Model Loader **********')
        print('onnxruntime version: \t', onnxruntime.__version__)
        print('****************************************************\n')


    def load_onnx_model(self, onnx_model_path):
        '''
        Load ONNX model

        Args:
            onnx_model_path:
        '''
        print('\n****************** Load ONNX Model ******************')
        print('Starting load onnx model [' + str(onnx_model_path) + ']...')
        self._isInit = False
        start_time = time.time()

        try:
            self._onnx_model = onnxruntime.InferenceSession(onnx_model_path)
            self._input_name = self._onnx_model.get_inputs()[0].name
            self._output_name = self._onnx_model.get_outputs()[0].name

            self._isInit = True

        except Exception as e:
            print('Load onnx model error. ', e)
            self._onnx_model = None

        finally:
            if self._isInit:
                print('Input name: ', self._input_name)
                print('Output name: ', self._output_name)
                print('Load onnx model success. Cost time: ' + str(time.time() - start_time)[0:5] + 's.')

            else:
                print('Load onnx model fail.')
        print('*****************************************************\n')


    def predict(self, image_path=None, image=None, debug=True):
        if debug:
            print('******************** Predict ********************')

        start_time = time.time()
        success = False
        predict_output = None
        softmax_output = None
        output = None
        try:
            if not self._isInit:
                if debug:
                    print('ONNX model is not init.')
                return

            if image is None:
                if image_path is None:
                    if debug:
                        print('Image and image path is None.')
                    return

                else:
                    image = cv2.imread(image_path)
            image = self._image_transform(image)

            # Predict
            predict_output = self._onnx_model.run(None, {self._input_name: image})
            softmax_output = self._softmax(predict_output[0])
            output = np.argmax(softmax_output[0])
            success = True

        except Exception as e:
            if debug:
                print('Predict error. ', e)

        finally:
            if debug:
                if success:
                    print('Predict ouptut: ', predict_output)
                    print('Softmax output: ', softmax_output)
                    print('Output: ', output)
                    print('Cost time: ' + str(time.time() - start_time)[0:5] + 's.')
                print('*************************************************')

            return predict_output, softmax_output, output


    def _image_transform(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1))
        image = np.array(image, dtype=np.float32)
        image[0] = (image[0] / 255 - 0.485) / 0.229
        image[1] = (image[1] / 255 - 0.456) / 0.224
        image[2] = (image[2] / 255 - 0.406) / 0.225
        image = np.array([image])
        return image


    def _softmax(self, x):
        f_x = np.exp(x) / np.sum(np.exp(x))
        return f_x








'''
=============================
Main
=============================
'''
if __name__ == '__main__':
    np.set_printoptions(linewidth=2000, precision=20)

    mobilenet_v3_onnx_model_loader = MobileNetV3ONNXModelLoader()
    mobilenet_v3_onnx_model_loader.load_onnx_model(onnx_model_path='../../../Liveness_Detection/models_zoo/mobilenet_v3/ONNX/0826/02/ir_liveness_detection_mobilenet_v3_small_112_val_acc_0.999287072243346.onnx')
    # mobilenet_v3_onnx_model_loader.predict(image_path='../../../Mask_Detection/initialize_image.jpg', debug=True)