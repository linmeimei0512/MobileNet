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
import cv2
import tensorflow as tf
import numpy as np

class MobileNetV3TensorFLowLiteModelLoader:
    _isInit = False

    def __init__(self):
        print('\n********** MobileNet V3 TensorFlow Lite Model Loader **********')
        print('tensorflow version: \t', tf.__version__)
        print('***************************************************************\n')


    def load_tensorflow_lite_model(self, tensorflow_lite_model_path):
        print('\n************** Load TensorFlow Lite Model ***************')
        print('Starting load tensorflow lite model [' + str(tensorflow_lite_model_path) + ']...')
        self._isInit = False
        start_time = time.time()

        try:
            self._tensorflow_lite_model = tf.lite.Interpreter(model_path=tensorflow_lite_model_path)
            self._tensorflow_lite_model.allocate_tensors()

            self._input = self._tensorflow_lite_model.get_input_details()
            self._output = self._tensorflow_lite_model.get_output_details()

            self._isInit = True


        except Exception as e:
            print('Load tensorflow lite model error. ', e)
            self._tensorflow_lite_model = None

        finally:
            if self._isInit:
                print('Input: ', self._input)
                print('Output: ', self._output)
                print('Load tensorflow lite model success. Cost time: ' + str(time.time() - start_time)[0:5] + 's.')

            print('*********************************************************\n')


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
            self._tensorflow_lite_model.set_tensor(self._input[0]['index'], image)
            self._tensorflow_lite_model.invoke()
            predict_output = self._tensorflow_lite_model.get_tensor(self._output[0]['index'])
            softmax_output = self._softmax(predict_output[0])
            output = np.argmax(softmax_output)
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

    mobilenet_v3_tensorflow_lite_model_loader = MobileNetV3TensorFLowLiteModelLoader()
    mobilenet_v3_tensorflow_lite_model_loader.load_tensorflow_lite_model(tensorflow_lite_model_path='../../../Mask_Detection/models_zoo/mobilenet_v3/label_6/tensorflow_lite/mask_detection_mobilenet_v3_small_112_label_6_acc_0.984_20220523-quant-int8.tflite')
    mobilenet_v3_tensorflow_lite_model_loader.predict(image_path='../../../Insightface/images/0_1633071405551.jpg')