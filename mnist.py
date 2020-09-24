import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam, SGD
import argparse
import os
import re
import time

NUM_CLASSES = 10

def get_dataset(filenames, batch_size):
    """Read the images and labels from 'filenames'."""
    # Repeat infinitely.
    dataset = tf.data.TFRecordDataset(filenames).repeat().shuffle(10000)

    # Parse records.

    # Batch it up.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset



def preprocess(ds):
    ds = ds.batch(batch_size)
    return ds

def get_model(learning_rate, weight_decay, optimizer, momentum):
    model = tf.keras.models.Sequential([tf.keras.layers.Lambda(lambda x: x/255.0,input_shape=(28,28,1)),
                                    tf.keras.layers.Conv2D(16,3),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128,activation='relu'),
                                    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')])
    return model


def main(args):
    # Hyper-parameters
    epochs       = args.epochs
    lr           = args.learning_rate
    batch_size   = args.batch_size
    momentum     = args.momentum
    weight_decay = args.weight_decay
    optimizer    = args.optimizer
    


    # SageMaker options
    training_dir   = args.training
    validation_dir = args.validation
    eval_dir       = args.eval
    
    train_dataset = get_dataset(training_dir+'/mnist-train.tfrecord-00000-of-00001',  batch_size)
    eval_dataset  = get_dataset(eval_dir+'/mnist-test.tfrecord-00000-of-00001', batch_size)
    
    input_shape = (HEIGHT, WIDTH, DEPTH)
    model = get_model(input_shape, lr, weight_decay, optimizer, momentum)
    
    # Optimizer
    if optimizer.lower() == 'sgd':
        opt = SGD(lr=lr, decay=weight_decay, momentum=momentum)
    else:
        opt = Adam(lr=lr, decay=weight_decay)

    # Compile model
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train model
    history = model.fit(train_dataset, steps_per_epoch=40000 // batch_size,epochs=epochs)
                        
    
    # Evaluate model performance
    score = model.evaluate(eval_dataset, steps=10000 // batch_size, verbose=1)
    print('Test loss    :', score[0])
    print('Test accuracy:', score[1])
    
    # Save model to model directory
    model.save(f'{os.environ["SM_MODEL_DIR"]}/{time.strftime("%m%d%H%M%S", time.gmtime())}', save_format='tf')