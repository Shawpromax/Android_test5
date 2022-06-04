# Android-实验5（TensorFlow Lite 模型生成）

# 一、准备工作，安装一些必备的库（安装的时候遇到ERROR: Cannot uninstall 'llvmlite'.的问题。卸载llvmlite包即可解决）
![image1](https://github.com/Shawpromax/images/blob/main/test5_1.png)
![image2](https://github.com/Shawpromax/images/blob/main/test5_2.png)
![image3](https://github.com/Shawpromax/images/blob/main/test5_3.png)

# 二、导入相关库
```
import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt
```

# 三、模型训练
## 1.获取数据
```
image_path = tf.keras.utils.get_file(
      'flower_photos.tgz',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      extract=True)
image_path = os.path.join(os.path.dirname(image_path), 'flower_photos')
```
## 2.运行实例（运行结果在下方）
### 第一步：加载数据集，并将数据集分为训练数据和测试数据
```
data = DataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)
```
### 第二步：训练Tensorflow模型
```
inception_v3_spec = image_classifier.ModelSpec(uri='https://storage.googleapis.com/tfhub-modules/tensorflow/efficientnet/lite0/feature-vector/2.tar.gz')

inception_v3_spec.input_image_shape = [240, 240]
model = image_classifier.create(train_data, model_spec=inception_v3_spec)
```
### 第三步：评估模型
```
loss, accuracy = model.evaluate(test_data)
```

### 第四步：导出Tensorflow Lite模型
```
model.export(export_dir='.')
```

# 四、使用实验三的应用验证导出的Tensorflow Lite模型
![image4](https://github.com/Shawpromax/images/blob/main/test5_4.png)
![image5](https://github.com/Shawpromax/images/blob/main/test5_5.png)
![image6](https://github.com/Shawpromax/images/blob/main/test5_6.png)
![image7](https://github.com/Shawpromax/images/blob/main/test5_7.png)




```python
import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt
```


```python
image_path = tf.keras.utils.get_file(
      'flower_photos.tgz',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      extract=True)
image_path = os.path.join(os.path.dirname(image_path), 'flower_photos')
```

    Downloading data from https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    228813984/228813984 [==============================] - 31s 0us/step
    


```python
data = DataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)
```

    INFO:tensorflow:Load image with size: 3670, num_label: 5, labels: daisy, dandelion, roses, sunflowers, tulips.
    


```python
inception_v3_spec = image_classifier.ModelSpec(uri='https://storage.googleapis.com/tfhub-modules/tensorflow/efficientnet/lite0/feature-vector/2.tar.gz')

inception_v3_spec.input_image_shape = [240, 240]
model = image_classifier.create(train_data, model_spec=inception_v3_spec)
```

    INFO:tensorflow:Retraining the models...
    

    INFO:tensorflow:Retraining the models...
    

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     hub_keras_layer_v1v2 (HubKe  (None, 1280)             3413024   
     rasLayerV1V2)                                                   
                                                                     
     dropout (Dropout)           (None, 1280)              0         
                                                                     
     dense (Dense)               (None, 5)                 6405      
                                                                     
    =================================================================
    Total params: 3,419,429
    Trainable params: 6,405
    Non-trainable params: 3,413,024
    _________________________________________________________________
    None
    Epoch 1/5
    

    D:\anaconda3\lib\site-packages\keras\optimizers\optimizer_v2\gradient_descent.py:108: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
      super(SGD, self).__init__(name, **kwargs)
    

    103/103 [==============================] - 73s 688ms/step - loss: 0.8784 - accuracy: 0.7621
    Epoch 2/5
    103/103 [==============================] - 69s 667ms/step - loss: 0.6583 - accuracy: 0.8938
    Epoch 3/5
    103/103 [==============================] - 70s 676ms/step - loss: 0.6246 - accuracy: 0.9169
    Epoch 4/5
    103/103 [==============================] - 69s 665ms/step - loss: 0.6058 - accuracy: 0.9281
    Epoch 5/5
    103/103 [==============================] - 69s 668ms/step - loss: 0.5942 - accuracy: 0.9266
    


```python
loss, accuracy = model.evaluate(test_data)
```

    12/12 [==============================] - 10s 687ms/step - loss: 0.5785 - accuracy: 0.9401
    


```python
model.export(export_dir='.')
```

    INFO:tensorflow:Assets written to: C:\Users\53261\AppData\Local\Temp\tmpy8n6j8ur\assets
    

    INFO:tensorflow:Assets written to: C:\Users\53261\AppData\Local\Temp\tmpy8n6j8ur\assets
    D:\anaconda3\lib\site-packages\tensorflow\lite\python\convert.py:766: UserWarning: Statistics for quantized inputs were expected, but not specified; continuing anyway.
      warnings.warn("Statistics for quantized inputs were expected, but not "
    

    INFO:tensorflow:Label file is inside the TFLite model with metadata.
    

    INFO:tensorflow:Label file is inside the TFLite model with metadata.
    

    INFO:tensorflow:Saving labels in C:\Users\53261\AppData\Local\Temp\tmpk98hjdlx\labels.txt
    

    INFO:tensorflow:Saving labels in C:\Users\53261\AppData\Local\Temp\tmpk98hjdlx\labels.txt
    

    INFO:tensorflow:TensorFlow Lite model exported successfully: .\model.tflite
    

    INFO:tensorflow:TensorFlow Lite model exported successfully: .\model.tflite
    


```python

```
