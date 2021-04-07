import tensorflow as tf
import argparse
from matplotlib import pyplot as plt
import pandas as pd
import os
import numpy as np
import logging
import datetime
import csv
from sklearn.utils import class_weight

print('tensorflow version:{}'.format(tf.__version__ ))

# ["0: Amblyomma, 1: Dermacentor, 2: Ixodes"] Class nums

img_shape = (224, 224, 3)


# Input arguments as a single comma separated line
def read_args():
    parser = argparse.ArgumentParser(description="TickNet NN")

    parser.add_argument("input",
                        type=str,
                        help="name of input file")

    args = parser.parse_args()
    return args


args = read_args()


def read_input(args):
    inputs = [line.rstrip() for line in open(args.input)]

    return inputs


inputs = read_input(args)

save_model = bool(inputs[9])
log = bool(inputs[10])

if log:
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
else:
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')


def config(inputs):
    logging.info("FUNCTION: config() -- initializing")
    config_dict = {
        "datetime": str(datetime.datetime.now()),
        "test_name": inputs[0],
        "train_dir": inputs[2],
        "val_dir": inputs[3],
        "num_classes": len(inputs[1].strip('][').split(', ')),
        "classes": inputs[1].strip('][').split(', '),
        "img_shape": img_shape,
        "model": "ResNet50V2",
        "optimizer": "Adam",
        "learning_rate": float(inputs[4]),
        "momentum": float(inputs[5]),
        "epochs": int(inputs[6]),
        "mbs": int(inputs[7]),
        "loss": "CategoricalCrossentropy",
        "weight_init": "imagenet",
        "class_weights": bool(inputs[8])
    }

    # logging.info("FUNCTION: config() -- writing config_dict to config.txt")
    # with open('config.txt', 'w') as f:
    #     print(config_dict, file=f)

    with open('config_{}.csv'.format(lr), 'w') as f:
        writer = csv.writer(f)
        for key, value in config_dict.items():
            writer.writerow([key, value])

    logging.info("FUNCTION: config() -- exiting")

    return config_dict


configs = config(inputs)


def define_model():
    logging.info("FUNCTION: define_model() -- initializing")
    base_model = tf.keras.applications.InceptionV3(
        input_shape=img_shape,
        include_top=False,
        weights="imagenet")

    # # Alternative model construction
    # maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
    # prediction_layer = tf.keras.layers.Dense(len(configs['classes']), activation='softmax')

    # model = tf.keras.Sequential([
    #     base_model,
    #     maxpool_layer,
    #     prediction_layer
    # ])

    x = base_model.output
    x1 = tf.keras.layers.GlobalMaxPooling2D()(x)
    x2 = tf.keras.layers.Dense(len(configs['classes']), activation='softmax')(x1)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=x2)

    model.summary()     # print network architecture

    opt = tf.keras.optimizers.SGD(lr=configs['lr'], momentum=configs['momentum'])
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    logging.info("FUNCTION: define_model() -- exiting (model compiled)")
    return model


def get_class_weights(train_nums):
    logging.info("FUNCTION: get_class_weights() -- initializing")
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(train_nums).tolist(),
                                                      train_nums)
    class_weight_dict = dict(enumerate(class_weights))
    logging.info("FUNCTION: get_class_weights() -- exiting")
    return class_weight_dict


def model_results(model, val_generator):
    logging.info("FUNCTION: model_results() -- initializing")
    pred = model.predict(val_generator, steps=len(val_generator), verbose=0)

    results_dict = {
        "fname2": [os.path.basename(file) for file in val_generator.filenames],
        "class_num": val_generator.classes,
        "label": [list(val_generator.class_indices)[idx] for idx in val_generator.classes],
        "pred_num": np.argmax(pred, 1),
        "pred_label": [list(val_generator.class_indices)[idx] for idx in np.argmax(pred, 1)],
        "prob": [round(np.max(idx) * 100, 2) for idx in pred],
        "prob_all": [idx for idx in pred]
    }

    correct = []
    for idx in range(len(results_dict["fname2"])):
        if results_dict['class_num'][idx] == results_dict['pred_num'][idx]:
            correct.append(1)
        else:
            correct.append(0)
    results_dict["correct"] = correct

    df = pd.DataFrame.from_dict(results_dict)
    logging.info("FUNCTION: model_results() -- exiting (results_dict returned as pd.Dataframe)")
    return df


# Save model as json file (optional)
def model2json(model):
    model_json = model.to_json()
    with open("model_{}.json".format(lr), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_{}.h5".format(lr))
    print("Saved model to disk")


def run_test_harness():
    logging.info("FUNCTION: run_test_harness() -- initializing")
    # define model
    model = define_model()
    # create data generator
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(featurewise_std_normalization=True,
                                                              featurewise_center=True)
    # specify mean values for centering (ImageNet)
    datagen.mean = [0.485, 0.456, 0.406]
    datagen.std = 0.225  # [0.229, 0.224, 0.225]

    # prepare iterator
    train_it = datagen.flow_from_directory(configs['train_dir'],
                                           class_mode='categorical',
                                           batch_size=configs['mbs'],
                                           target_size=(img_shape[0], img_shape[1]))
    val_it = datagen.flow_from_directory(configs['val_dir'],
                                         class_mode='categorical',
                                         batch_size=configs['mbs'],
                                         target_size=(img_shape[0], img_shape[1]),
                                         shuffle=False)

    if class_weights:
        weights = get_class_weights(train_it.classes)
        history = model.fit(train_it,
                            steps_per_epoch=len(train_it),
                            validation_data=val_it,
                            validation_steps=int(len(val_it)/2),
                            epochs=configs['epochs'],
                            class_weight=weights,
                            verbose=1)
    else:
        # fit model
        history = model.fit(train_it,
                            steps_per_epoch=len(train_it),
                            validation_data=val_it,
                            validation_steps=int(len(val_it)/2),
                            epochs=configs['epochs'],
                            verbose=1)


    results = model_results(model, val_it)
    logging.info("FUNCTION: run_test_harness() -- writing results to results.csv")

    results.to_csv("results.csv")

    model.save('my_model') # Model directory. See Tensorflow method https://www.tensorflow.org/guide/keras/save_and_serialize

    logging.info("FUNCTION: run_test_harness() -- exiting")


run_test_harness()
