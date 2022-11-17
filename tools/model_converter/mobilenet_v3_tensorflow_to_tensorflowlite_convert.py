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
import tensorflow as tf
import numpy as np
from enum import Enum


'''
************************
tensorflow-gpu==2.5.0
tensorflow-addons==0.14.0
************************
'''


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class TensorFlowLiteQuantType(Enum):
    EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8 = 1
    TFLITE_BUILTINS_INT8 = 2



class MobileNetV3TensorFlowToTensorFlowLiteConvert:
    quant_dataset_path = ''

    def __init__(self):
        print('\n*********** Tensorflow To Tensorflow Lite Converter ***********')
        print('tensorflow version: ', tf.__version__)
        print('***************************************************************')


    def convert(self, tensorflow_model_path, tensorflow_lite_model_output_path, input_names, output_names, quant, quant_type:TensorFlowLiteQuantType, quant_dataset_path):
        print('\nStarting convert to tensorflow lite ...')
        self.quant_dataset_path = quant_dataset_path
        start_time = time.time()

        try:
            converter = tf.lite.TFLiteConverter.from_saved_model(tensorflow_model_path)

            #### Quant ####
            if quant:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]

                # EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
                if quant_type == TensorFlowLiteQuantType.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8:
                    print('Quant by EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8')
                    converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]

                # TFLITE_BUILTINS_INT8
                elif quant_type == TensorFlowLiteQuantType.TFLITE_BUILTINS_INT8:
                    print('Quant by TFLITE_BUILTINS_INT8')
                    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                    converter.representative_dataset = self.representative_dataset_gen

                # converter.optimizations = [tf.lite.Optimize.DEFAULT]
                # converter.target_spec.supported_types = [tf.float16]
                # converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
                # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                # converter.representative_dataset = self.representative_dataset_gen

            tflite_model = converter.convert()

            with open(tensorflow_lite_model_output_path, 'wb') as f:
                f.write(tflite_model)

        except Exception as ex:
            print('Convert to tensorflow lite failed.', ex)
            return

        print('Convert to tensorflow lite model success. Cost time: ', time.time() - start_time, 's.')



    def representative_dataset_gen(self):
        import cv2
        # dataset_path = '../../../Insightface/images/test/Aaron Lee/fail/mask'
        for image in os.listdir(self.quant_dataset_path):
            image = cv2.imread(os.path.join(self.quant_dataset_path, image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.transpose(image, (2, 0, 1))
            image = np.array(image, dtype=np.float32)
            image[0] = (image[0] / 255 - 0.485) / 0.229
            image[1] = (image[1] / 255 - 0.456) / 0.224
            image[2] = (image[2] / 255 - 0.406) / 0.225
            image = np.array([image])
            yield [image]


    def representative_dataset(self):
        for _ in range(100):
            data = np.random.rand(1, 112, 112, 3)
            yield [data.astype(np.float32)]







'''
=============================
Config
=============================
'''
onnx_model_path = '../../../Liveness_Detection/models_zoo/mobilenet_v3/ONNX/0905/01/ir_liveness_detection_mobilenet_v3_small_112_val_acc_0.9996203492786636.onnx'
tensorflow_model_path = '../../../Liveness_Detection/models_zoo/mobilenet_v3/TensorFlow/0905/01/ir_liveness_detection_mobilenet_v3_small_112_val_acc_0.9996203492786636'
tensorflow_lite_model_output_path = '../../../Liveness_Detection/models_zoo/mobilenet_v3/TensorFlowLite/0905/01/ir_liveness_detection_mobilenet_v3_small_112_val_acc_0.99962_20220905_01.tflite'
input_names = ['input']
output_names = ['output']

quant = True
quant_type = TensorFlowLiteQuantType.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
quant_dataset_path = '/media/funtoro/SSD Disk/Liveness_Detection/Datasets/IR/val/00_Living'

predict_image_path = '../../../Mask_Detection/initialize_image.jpg'



'''
=============================
Main
=============================
'''
if __name__ == '__main__':
    # mobilenet_v3_tensorflow_to_tensorflowlite_convert = MobileNetV3TensorFlowToTensorFlowLiteConvert()
    # mobilenet_v3_tensorflow_to_tensorflowlite_convert.convert(tensorflow_model_path=tensorflow_model_path,
    #                                                           tensorflow_lite_model_output_path=tensorflow_lite_model_output_path,
    #                                                           input_names=input_names,
    #                                                           output_names=output_names,
    #                                                           quant=quant,
    #                                                           quant_type=quant_type,
    #                                                           quant_dataset_path=quant_dataset_path)



    from tools.model_loader.mobilenet_v3_onnx_model_loader import MobileNetV3ONNXModelLoader
    mobilenet_v3_onnx_model_loader = MobileNetV3ONNXModelLoader()
    mobilenet_v3_onnx_model_loader.load_onnx_model(onnx_model_path=onnx_model_path)
    onnx_predict_output, onnx_softmax_output, onnx_output = mobilenet_v3_onnx_model_loader.predict(image_path=predict_image_path, debug=False)
    print(onnx_predict_output)

    from tools.model_loader.mobilenet_v3_tensorflow_lite_model_loader import MobileNetV3TensorFLowLiteModelLoader
    mobilenet_v3_tensorflow_lite_model_loader = MobileNetV3TensorFLowLiteModelLoader()
    mobilenet_v3_tensorflow_lite_model_loader.load_tensorflow_lite_model(tensorflow_lite_model_path=tensorflow_lite_model_output_path)
    tensorflow_lite_predict_output, tensorflow_lite_softmax_output, tensorflow_lite_output = mobilenet_v3_tensorflow_lite_model_loader.predict(image_path=predict_image_path, debug=False)
    print(tensorflow_lite_predict_output)

    print('\n********** Verify Accuracy **********')
    from scipy.spatial.distance import cosine

    acc = 1 - cosine(onnx_predict_output, tensorflow_lite_predict_output)
    print('Accuracy: ' + str(acc * 100)[0:5] + '%')
    print('*************************************\n')