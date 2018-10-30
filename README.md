# keras-yolo3

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

A Keras implementation of YOLOv3 (Tensorflow backend) inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K).


---

## Quick Start

1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model to a Keras model.
3. Run YOLO detection.

```
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
python yolo_video.py [OPTIONS...] --image, for image detection mode, OR
python yolo_video.py [video_path] [output_path (optional)]
```

For Tiny YOLOv3, just do in a similar way, just specify model path and anchor path with `--model model_file` and `--anchors anchor_file`.

### Usage
Use --help to see usage of yolo_video.py:
```
usage: yolo_video.py [-h] [--model MODEL] [--anchors ANCHORS]
                     [--classes CLASSES] [--gpu_num GPU_NUM] [--image]
                     [--input] [--output]

positional arguments:
  --input        Video input path
  --output       Video output path

optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      path to model weight file, default model_data/yolo.h5
  --anchors ANCHORS  path to anchor definitions, default
                     model_data/yolo_anchors.txt
  --classes CLASSES  path to class definitions, default
                     model_data/coco_classes.txt
  --gpu_num GPU_NUM  Number of GPU to use, default 1
  --image            Image detection mode, will ignore all positional arguments
```
---

4. MultiGPU usage: use `--gpu_num N` to use N GPUs. It is passed to the [Keras multi_gpu_model()](https://keras.io/utils/#multi_gpu_model).

## Training

1. Generate your own annotation file and class names file.  
    One row for one image;  
    Row format: `image_file_path box1 box2 ... boxN`;  
    Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  
    For VOC dataset, try `python voc_annotation.py`  
    Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```

2. Make sure you have run `python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5`  
    The file model_data/yolo_weights.h5 is used to load pretrained weights.

3. Modify train.py and start training.  
    `python train.py`  
    Use your trained weights or checkpoint weights with command line option `--model model_file` when using yolo_video.py
    Remember to modify class path or anchor path, with `--classes class_file` and `--anchors anchor_file`.

If you want to use original pretrained weights for YOLOv3:  
    1. `wget https://pjreddie.com/media/files/darknet53.conv.74`  
    2. rename it as darknet53.weights  
    3. `python convert.py -w darknet53.cfg darknet53.weights model_data/darknet53_weights.h5`  
    4. use model_data/darknet53_weights.h5 in train.py

---

## Some issues to know

1. The test environment is
    - Python 3.5.2
    - Keras 2.1.5
    - tensorflow 1.6.0

2. Default anchors are used. If you use your own anchors, probably some changes are needed.

3. The inference result is not totally the same as Darknet but the difference is small.

4. The speed is slower than Darknet. Replacing PIL with opencv may help a little.

5. Always load pretrained weights and freeze layers in the first stage of training. Or try Darknet training. It's OK if there is a mismatch warning.

6. The training strategy is for reference only. Adjust it according to your dataset and your goal. And add further strategy if needed.

7. For speeding up the training process with frozen layers train_bottleneck.py can be used. It will compute the bottleneck features of the frozen model first and then only trains the last layers. This makes training on CPU possible in a reasonable time. See [this](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) for more information on bottleneck features.


# Keras YOLOv3 on self Panda dataset

## Modify points base on master branch

### train.py
```python
    annotation_path = '2007_train.txt'
    log_dir = 'logs/000/'
    classes_path = 'model_data/my_classes.txt'
```

`train.py` line 70
```
    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if False:  # My computer(E5-2660/16G/GTX1060) fails run this configuration
```

### voc_annotation.py
```python
# sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
sets=[('2007', 'train')]

# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes = ["panda"]
```

### yolo.py
```python
class YOLO(object):
    _defaults = {
        "model_path": 'model_data/panda_trained_weights_stage_1.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/my_classes.txt',
```

## How to use

1. Download [VOCdevkit_Panda60_1030.zip](https://github.com/LichenZeng/keras-yolo3/releases) and extract to this root directory, such as '/home/tensorflow01/workspace/python_study/keras-yolo3/VOCdevkit'
2. Download [panda_trained_weights_stage_1.h5.zip](https://github.com/LichenZeng/keras-yolo3/releases) and extract to model_data, such as '/home/tensorflow01/workspace/python_study/keras-yolo3/model_data/panda_trained_weights_stage_1.h5'

**Note** : 'VOCdevkit_Panda60_1030.zip' is the dataset create by self, 'panda_trained_weights_stage_1.h5.zip' is the pre-trained weights base on self Panda dataset.

3. Run `voc_annotation.py` ($ python voc_annotation.py) to create train file "2007_train.txt"

4. Run `yolo_video.py` ($ python yolo_video.py --image) to detect picture base on pre-trained weights 'panda_trained_weights_stage_1.h5'

5. Run `train.py` ($ python train.py) to re-train YOLO v3 base on `yolo_weights.h5` (as above `python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5`) and 'VOCdevkit_Panda60_1030.zip' dataset.  
If you want to re-train base on 'panda_trained_weights_stage_1.h5', you can modify code as follows:
```python
    else:
        model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/yolo_weights.h5') # make sure you know what you freeze
```


**Reference**:
1. [【AI实战】动手训练自己的目标检测模型(YOLO篇)](https://www.liangzl.com/get-article-detail-12753.html)
2. [【AI实战】手把手教你训练自己的目标检测模型（SSD篇）](https://my.oschina.net/u/876354/blog/1927351)
