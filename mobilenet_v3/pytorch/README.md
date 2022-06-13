
# MobileNet v3 - PyTorch

## Recent Update

**`2022-06-10`** : First commit.



## Set Up
#### CPU Version
| Package                   | Version       | 
| :---                      | :---          |
| torch                     | 1.8.0         |
| opencv-python             | latest        |
| Pillow                    | latest        |

###
#### GPU Version (CUDA 10.1)
| Package                   | Version       | 
| :---                      | :---          |
| torch                     | 1.8.0+cu111   |
| opencv-python             | latest        |
| Pillow                    | latest        |



##
## Create Python Environment
Python 3.6
```shell
conda create -n insightface python=3.6
```

##
## Install PyTorch
#### [PyTorch](https://pytorch.org/get-started/previous-versions/)
### v1.9.0
#### OSX
```shell
pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0
```
#### Linux and Windows
```shell
# ROCM 4.2 (Linux only)
pip install torch==1.9.0+rocm4.2 torchvision==0.10.0+rocm4.2 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# ROCM 4.1 (Linux only)
pip install torch==1.9.0+rocm4.1 torchvision==0.10.0+rocm4.1 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# ROCM 4.0.1 (Linux only)
pip install torch==1.9.0+rocm4.0.1 torchvision==0.10.0+rocm4.0.1 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 11.1
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 10.2
pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# CPU only
pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

### v1.8.0
#### OSX
```shell
pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0
```
#### Linux and Windows
```shell
# RocM 4.0.1 (Linux only)
pip install torch -f https://download.pytorch.org/whl/rocm4.0.1/torch_stable.html
pip install ninja
pip install 'git+https://github.com/pytorch/vision.git@v0.9.0'

# CUDA 11.1
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 10.2
pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0

# CPU only
pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```


##
## Install Python Package
[requirements.txt](requirements.txt)
```shell
pip install -r requirements.txt
```


##
## Training
### [train.py](train.py)
```shell
usage: train.py [-h] [--model_size MODEL_SIZE] [--batch_size BATCH_SIZE] [--epochs EPOCHS]
                [--pretrain_weight_path PRETRAIN_WEIGHT_PATH]
                [--datasets_path DATASETS_PATH]
                [--save_class_json_file_path SAVE_CLASS_JSON_FILE_PATH]
                [--save_model_path SAVE_MODEL_PATH]
                [--save_model_name SAVE_MODEL_NAME]
                [--save_weight_name SAVE_WEIGHT_NAME]

MobileNet v3 PyTorch Training

optional arguments:
  -h, --help            show this help message and exit
  --model_size MODEL_SIZE
                        0: use small, 1: use large
  --batch_size BATCH_SIZE
                        training batch size
  --epochs EPOCHS       training epochs
  --pretrain_weight_path PRETRAIN_WEIGHT_PATH
                        pre-train weight for training
  --datasets_path DATASETS_PATH
                        datasets path for training
  --save_class_json_file_path SAVE_CLASS_JSON_FILE_PATH
                        save class json file path
  --save_model_path SAVE_MODEL_PATH
                        save model path
  --save_model_name SAVE_MODEL_NAME
                        save model name
  --save_weight_name SAVE_WEIGHT_NAME
                        save model weight name
```
```shell
python train.py \
      --model_size 0 \
      --batch_size 128 \
      --epochs 20 \
      --pretrain_weight_path ../../models_zoo/mobilenet_v3_small/pytorch/mobilenet_v3_small-047dcff4.pth \
      --datasets ../../datasets/flower/ \
      --save_class_json_file_path ../../datasets/flower/flower_class.json \
      --save_model_path ../../models_zoo/mobilenet_v3_small/pytorch/ \
      --save_model_name flower_classification.pth \
      --save_weight_name flower_classification_weight.pth
```


##
## Predict
### [mobilenet_v3_pytorch_model_loader.py](../../tools/model_loader/mobilenet_v3_pytorch_model_loader.py)
```shell
usage: mobilenet_v3_pytorch_model_loader.py [-h] [--model_size MODEL_SIZE]
                                            [--model_path MODEL_PATH]
                                            [--model_weight_path MODEL_WEIGHT_PATH]
                                            [--num_classes NUM_CLASSES]
                                            [--initialize_image_path INITIALIZE_IMAGE_PATH]

MobileNet v3 PyTorch Predict

optional arguments:
  -h, --help            show this help message and exit
  --model_size MODEL_SIZE
                        0: use small, 1: use large
  --model_path MODEL_PATH
                        model path
  --model_weight_path MODEL_WEIGHT_PATH
                        model weight path
  --num_classes NUM_CLASSES
                        number of class
  --initialize_image_path INITIALIZE_IMAGE_PATH
                        initialize image path
```
```shell
python mobilenet_v3_pytorch_model_loader.py \
      --model_size 0 \
      --model_path ../../models_zoo/mobilenet_v3_small/pytorch/flower_classification_val_acc_0.8791208791208791.pth \
      --num_classes 5 \
      --initialize_image_path ../../datasets/flower/flower_photos/daisy/54377391_15648e8d18.jpg \
      --predict_image_path ../../datasets/flower/flower_photos/daisy/54377391_15648e8d18.jpg
```