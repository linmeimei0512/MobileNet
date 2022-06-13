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

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class MobileNetV3TensorFlowToTensorFlowLiteConvert:
    def __init__(self):
        print('\n*********** Tensorflow To Tensorflow Lite Converter ***********')
        print('tensorflow version: ', tf.__version__)
        print('***************************************************************')


    def convert(self, tensorflow_model_path, tensorflow_lite_model_output_path, input_names, output_names, quant):
        print('\nStarting convert to tensorflow lite ...')
        start_time = time.time()

        try:
            converter = tf.lite.TFLiteConverter.from_saved_model(tensorflow_model_path)
            if quant:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                # converter.target_spec.supported_types = [tf.float16]
                converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
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
        dataset_path = '../../../Insightface/images/test/Aaron Lee/fail/mask'
        for image in os.listdir(dataset_path):
            image = cv2.imread(os.path.join(dataset_path, image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.transpose(image, (2, 0, 1))
            image = np.array(image, dtype=np.float32)
            image[0] = (image[0] / 255 - 0.485) / 0.229
            image[1] = (image[1] / 255 - 0.456) / 0.224
            image[2] = (image[2] / 255 - 0.406) / 0.225
            image = np.array([image])
            yield [image]









'''
=============================
Main
=============================
'''
if __name__ == '__main__':
    mobilenet_v3_tensorflow_to_tensorflowlite_convert = MobileNetV3TensorFlowToTensorFlowLiteConvert()
    mobilenet_v3_tensorflow_to_tensorflowlite_convert.convert(tensorflow_model_path='../../../Mask_Detection/models_zoo/mobilenet_v3/label_6/tensorflow/mask_detection_mobilenet_v3_small_112_label_6_acc_0.984_20220523',
                                                              tensorflow_lite_model_output_path='../../../Mask_Detection/models_zoo/mobilenet_v3/label_6/tensorflow_lite/mask_detection_mobilenet_v3_small_112_label_6_acc_0.984_20220523-quant-16x8.tflite',
                                                              input_names=['input'],
                                                              output_names=['output'],
                                                              quant=True)



    from tools.model_loader.mobilenet_v3_onnx_model_loader import MobileNetV3ONNXModelLoader
    mobilenet_v3_onnx_model_loader = MobileNetV3ONNXModelLoader()
    mobilenet_v3_onnx_model_loader.load_onnx_model(onnx_model_path='../../../Mask_Detection/models_zoo/mobilenet_v3/label_6/onnx/mask_detection_mobilenet_v3_small_112_label_6_acc_0.984_20220523.onnx')
    onnx_predict_output, onnx_softmax_output, onnx_output = mobilenet_v3_onnx_model_loader.predict(image_path='../../../Mask_Detection/initialize_image.jpg', debug=False)
    print(onnx_predict_output)

    from tools.model_loader.mobilenet_v3_tensorflow_lite_model_loader import MobileNetV3TensorFLowLiteModelLoader
    mobilenet_v3_tensorflow_lite_model_loader = MobileNetV3TensorFLowLiteModelLoader()
    mobilenet_v3_tensorflow_lite_model_loader.load_tensorflow_lite_model(tensorflow_lite_model_path='../../../Mask_Detection/models_zoo/mobilenet_v3/label_6/tensorflow_lite/mask_detection_mobilenet_v3_small_112_label_6_acc_0.984_20220523-quant-16x8.tflite')
    tensorflow_lite_predict_output, tensorflow_lite_softmax_output, tensorflow_lite_output = mobilenet_v3_tensorflow_lite_model_loader.predict(image_path='../../../Mask_Detection/initialize_image.jpg', debug=False)
    print(tensorflow_lite_predict_output)

    print('\n********** Verify Accuracy **********')
    from scipy.spatial.distance import cosine

    acc = 1 - cosine(onnx_predict_output, tensorflow_lite_predict_output)
    print('Accuracy: ' + str(acc * 100)[0:5] + '%')
    print('*************************************\n')