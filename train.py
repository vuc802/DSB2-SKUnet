import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from all_tfrecord_load import load_train_tfrecord
from model import build_model
from loss import binary_bce_dice_loss ,jaccard_distance_loss
from metrics import dice_coefficient ,iou
from keras import backend as K

###如果GPU內存不夠,就會跳出>>cuda dnn is wrong
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
     train_path = BASE_DIR + 'train_normal.tfrecords'
     val_path = BASE_DIR +'val_normal.tfrecords'
     
     ## Hyperparameters
     batch =16
     lr = 1*1e-4
     epochs = 75
     train_dataset = load_train_tfrecord(train_path,batch=batch)
     val_dataset = load_train_tfrecord(val_path,batch=batch)

     model = build_model()
     
     opt = tf.keras.optimizers.Adam(lr)
     metrics = [dice_coefficient, iou]
     model.compile(loss="binary_crossentropy", optimizer=opt, metrics=metrics)
     model.summary()
     #"binary_crossentropy"

     callbacks = [
          ModelCheckpoint(BASE_DIR + "calc//skunet_model.h5"),
          ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10),
          CSVLogger(BASE_DIR + "calc//skunet.csv"),
          TensorBoard(),
          EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=False)
     ]

     model.fit(train_dataset,
          validation_data=val_dataset,
          epochs=epochs,
          callbacks=callbacks)

     model.save(BASE_DIR +"calc//skunet_model.h5")
     K.clear_session()