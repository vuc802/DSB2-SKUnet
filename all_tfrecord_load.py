import tensorflow as tf
import os
import glob

image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string)
    }

@tf.autograph.experimental.do_not_convert
def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    feature_dict = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.io.decode_png(feature_dict['image'], channels=1) 
    label = tf.io.decode_png(feature_dict['label'], channels=1) 
    image = tf.cast(image, dtype='float32') / 255.
    label = tf.cast(label, dtype='float32') / 255.
    return image, label

def load_train_tfrecord(filenames,batch):
    raw_dataset = tf.data.TFRecordDataset(filenames)
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string)
    }
    #raw_dataset = raw_dataset.repeat (1)
    raw_dataset = raw_dataset.map(_parse_image_function)  # 解析數據
    raw_dataset = raw_dataset.shuffle (buffer_size = 1024) # 在緩衝區中隨機打亂數據
    raw_dataset  = raw_dataset.batch (batch_size = batch, drop_remainder=True) # 每32條數據爲一個batch，生成一個新的Dataset
    
    return raw_dataset





       