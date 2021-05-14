import tensorflow as tf
import pandas as pd
from PIL import Image
import numpy as np
import os
import argparse
from math import floor

img_shape = (224, 224, 3)

classes = ['Amblyomma americanum', 'Dermacentor variabilis', 'Ixodes scapularis']


def read_args():  # Get user arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("source",
                        type=str,
                        default=None,
                        help="Path to directory of cropped 224x224 images to predict")

    parser.add_argument("dest",
                        type=str,
                        default=None,
                        help="Save destination for results csv")

    parser.add_argument("model",
                        type=str,
                        default=None,
                        help="Path to save model directory")

    args = parser.parse_args()

    return args


args = read_args()


def get_input(path):  # Load image
    img = Image.open(path)
    return img


def im_crop(image):  # Crop images into a square along their shortest side
    dim = image.size
    shortest = min(dim[0:2])
    longest = max(dim[0:2])

    if shortest != longest:
        lv = np.array(range(0, shortest)) + floor((longest - shortest) / 2)
        if dim[0] == shortest:
            im_cropped = np.asarray(image)[lv, :, :]
        else:
            im_cropped = np.asarray(image)[:, lv, :]

        im_cropped = Image.fromarray(im_cropped)
    else:
        im_cropped = image

    return im_cropped


images = []
fname = []
for f in os.listdir(args.source):
    try:
        img = get_input(os.path.join(args.source, f))
        img_cropped = im_crop(img)
        img_resized = img_cropped.resize((img_shape[0], img_shape[1]))
        pixels = np.asarray(img_resized)  # convert image to array
        pixels = pixels.astype('float32')
        input = np.expand_dims(pixels, axis=0)  # adds batch dimension
        images.append(input)
        fname.append(f)
    except:
        print("Image {} could not be opened".format(f))
        pass

# stack up images list to pass for prediction
images = np.vstack(images)

model = tf.keras.models.load_model(args.model)

results = model.predict(images, batch_size=32, verbose=1)

class_prob = np.amax(results, 1).tolist()
rounded_class_prob = [round(100 * x, 2) for x in class_prob]
class_ind = np.argmax(results, 1)
preds = [classes[i] for i in class_ind]

NN_dict = {
    "fname": fname,
    "prediction": preds,
    "class": class_ind.tolist(),
    "prob": rounded_class_prob,
    "results": results.tolist()
}

df = pd.DataFrame(NN_dict)

df.to_csv(os.path.join(args.dest, 'results.csv'))  # save results to csv file

