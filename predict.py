import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from all_tfrecord_load import load_train_tfrecord
from loss import binary_bce_dice_loss ,jaccard_distance_loss
from metrics import dice_coefficient ,iou
from model import Axpby
from keras import backend as K

def solve_cudnn_error():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

solve_cudnn_error()


if __name__ == "__main__":
    ## Dataset
    BASE_DIR = 'D:\\shu\\project\\data\\'
    batch = 16
    test_path = BASE_DIR + 'test_normal.tfrecords'
    test_dataset = load_train_tfrecord(test_path,batch=batch)

    with CustomObjectScope({ 'Axpby': Axpby , 'dice_coefficient': dice_coefficient , 'iou' : iou }):
         #if you use custom layer ,loss,or metrics , please add it here
         model = tf.keras.models.load_model(BASE_DIR + "calc\\skunet_model.h5")

    acc=[]
    for image, mask in iter(test_dataset):     
        cost = model.evaluate(image, mask, batch_size=16)
        acc.append(cost)

    mean_acc=np.mean(acc,axis=0)
    print("test cost: {}".format(mean_acc))
    K.clear_session()