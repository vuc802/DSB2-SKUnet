import tensorflow as tf
import os
import numpy as np

data_dir = 'D:\\shu\\project\\data\\'
train_img_dir = data_dir + 'train\\img\\'
train_gt_dir = data_dir + 'train\\gt\\'
val_img_dir = data_dir + 'validate\\img\\'
val_gt_dir = data_dir + 'validate\\gt\\'
test_img_dir = data_dir + 'test\\img\\'
test_gt_dir = data_dir + 'test\\gt\\'
foldertype=['train',"validate", "test"]

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() 
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


#寫入func.
def write_tfrecord(tfr_img,tfr_gt,folder):
    tfrecord_file = data_dir + folder +'.tfrecords'

    # filenames = [filename for filename in os.listdir(tfr_img)]
    # labels = [filename_gt for filename_gt in os.listdir(tfr_gt)]#順序對不上,改別種用法
    filenames=[]
    labels=[]
    img_list=os.listdir(tfr_img)
    for i in np.arange(len(img_list)):
        img=img_list[i]
        img_dir=tfr_img + img
        gt=(img_list[i]).replace('.png','_gt.png')
        gt_dir=tfr_gt + gt
        filenames.append(img_dir)
        labels.append(gt_dir)

    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for filename, label in zip(filenames, labels):
            image = open(filename, 'rb').read()  
            image_shape = tf.image.decode_png(image).shape
            gt = open(label, 'rb').read()     
            feature = {                             
                'image': _bytes_feature(image),  
                'label': _bytes_feature(gt)   
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature)) 
            writer.write(example.SerializeToString())   


if __name__=='__main__':
    write_tfrecord(train_img_dir,train_gt_dir,'train')
    write_tfrecord(val_img_dir,val_gt_dir,'val')
    write_tfrecord(test_img_dir,test_gt_dir,'test') 