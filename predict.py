import tensorflow as tf
import pandas as pd
from PIL import Image
import numpy as np
from math import floor
import os

path = '/ImageFolder' # path to folder of cropped images
img_shape = (224,224,3)

species = ['Amblyomma', 'Dermacentor', 'Ixodes']
mbs =32

model = tf.keras.models.load_model('Model_Final')


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

    correct = []
    for idx in range(len(results_dict["fname"])):
        if results_dict['class_num'][idx] == results_dict['pred_num'][idx]:
            correct.append(1)
        else:
            correct.append(0)
    results_dict["correct"] = correct

    df = pd.DataFrame.from_dict(results_dict)
    return df


df = model_results(model, test_it)

df.to_csv('results.csv')    # save results to csv file
