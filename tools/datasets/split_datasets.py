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
from shutil import copy, rmtree
import random

class SplitDatasets:
    _datasets_root_path = ''
    _datasets_path = ''
    _datasets_train_path = ''
    _datasets_val_path = ''
    _val_rate = 0.1

    def __init__(self, datasets_root_path, datasets_folder_name, val_rate=0.1):
        self._datasets_root_path = datasets_root_path
        self._datasets_path = os.path.join(datasets_root_path, datasets_folder_name)
        self._datasets_train_path = os.path.join(datasets_root_path, 'train')
        self._datasets_val_path = os.path.join(datasets_root_path, 'val')
        self._val_rate = val_rate

        self._create_folder()


    def _create_folder(self):
        '''
        Create folder
        '''
        # train folder
        if os.path.isdir(self._datasets_train_path):
            rmtree(self._datasets_train_path)
        os.mkdir(self._datasets_train_path)

        # val folder
        if os.path.isdir(self._datasets_val_path):
            rmtree(self._datasets_val_path)
        os.mkdir(self._datasets_val_path)


    def split(self):
        '''
        Split datasets
        '''
        random.seed(0)
        assert os.path.exists(self._datasets_path), "path '{}' does not exist.".format(self._datasets_path)

        # Get datasets class
        datasets_class = [cla for cla in os.listdir(self._datasets_path)
                          if os.path.isdir(os.path.join(self._datasets_path, cla))]

        for cla in datasets_class:
            os.mkdir(os.path.join(self._datasets_train_path, cla))
            os.mkdir(os.path.join(self._datasets_val_path, cla))

        for cla in datasets_class:
            cla_path = os.path.join(self._datasets_path, cla)
            images = os.listdir(cla_path)
            num = len(images)

            eval_index = random.sample(images, k=int(num * self._val_rate))
            for index, image in enumerate(images):
                if image in eval_index:
                    image_path = os.path.join(cla_path, image)
                    new_path = os.path.join(self._datasets_val_path, cla)
                    copy(image_path, new_path)

                else:
                    image_path = os.path.join(cla_path, image)
                    new_path = os.path.join(self._datasets_train_path, cla)
                    copy(image_path, new_path)
                print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")
            print()

        print("processing done!")



'''
=============================
Main
=============================
'''
if __name__ == '__main__':
    split_datasets = SplitDatasets(datasets_root_path='/media/funtoro/SSD Disk/Flower/Datasets',
                                   datasets_folder_name='flower_photos',
                                   val_rate=0.1)
    split_datasets.split()