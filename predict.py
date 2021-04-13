import tensorflow as tf
import pandas as pd
from PIL import Image
import numpy as np
import os
import argparse

def read_args():  # Get user arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--source",
                        type=str,
                        default=None,
                        help="Path to directory of cropped 224x224 images to predict")

    parser.add_argument("--dest",
                        type=str,
                        default=None,
                        help="Save destination for results csv")
    
    parser.add_argument("--model",
                        type=str,
                        default=None,
                        help="Path to save model directory")
     return args

args = read_args()


path = args.source # path to folder of cropped images
img_shape = (224,224,3)

species = ['Amblyomma americanum', 'Dermacentor variabilis', 'Ixodes scapularis']
mbs =32

model = tf.keras.models.load_model(args.model)


datagen = tf.keras.preprocessing.image.ImageDataGenerator(featurewise_std_normalization=True,
                                                              featurewise_center=True)
datagen.mean = [0.485, 0.456, 0.406]
datagen.std = 0.225  # [0.229, 0.224, 0.225]

test_it = datagen.flow_from_directory(path,
                                         class_mode='categorical',
                                         batch_size=mbs,
                                         target_size=(img_shape[0], img_shape[1]),
                                         shuffle=False)


def model_results(model, test_generator):
    pred = model.predict(test_generator, steps=len(test_generator), verbose=1)

    results_dict = {
        "fname": [os.path.basename(file) for file in test_generator.filenames],
        "class_num": test_generator.classes,
        "label": [list(test_generator.class_indices)[idx] for idx in test_generator.classes],
        "pred_num": np.argmax(pred, 1),
        "pred_label": [list(test_generator.class_indices)[idx] for idx in np.argmax(pred, 1)],
        "prob": [round(np.max(idx) * 100, 2) for idx in pred],
        "prob_all": [idx for idx in pred]
    }

df = model_results(model, test_it)

df.to_csv(os.path.join(args.destination,'results.csv'))    # save results to csv file
