import xml.etree.ElementTree as ET
import os
import numpy as np
import keras
import math
import tensorflow as tf
import cv2
from os import listdir, walk
from os.path import join
from keras_retinanet.bin.train import create_generators,create_models,create_callbacks
from keras_retinanet.models import backbone,load_model,convert_model
from keras_retinanet.utils.config import read_config_file,parse_anchor_parameters
from keras_retinanet.utils.visualization import draw_boxes
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa

tf.set_random_seed(31) # SEEDS MAKE RESULTS MORE REPRODUCABLE
np.random.seed(17)

import csv 
import pandas as pd 
with open('training_data/images/deduped_annotations.csv') as f : 
    ls = list(csv.reader(f))[1:]
    
def to_annotation(l) : 
    # Note swap of 1st and 2nd Y
    return ("training_data/images/" + l[0][11:],l[1],l[4],l[3],l[2],1)

x_train,x_val = train_test_split([to_annotation(l) for l in ls], test_size=0.2)
pd.DataFrame(x_train).to_csv("annotations.csv",header=None,index=None)
pd.DataFrame(x_val).to_csv("val_annotations.csv",header=None,index=None)

b = backbone('resnet50')
training_len = 5408
testing_len = 1353

class args:
    batch_size = 64
    config = read_config_file('config.ini')
    random_transform = True # Image augmentation
    annotations = 'annotations.csv'
    val_annotations = 'val_annotations.csv'
    classes = 'classes.csv'
    image_min_side = 672
    image_max_side = 672
    dataset_type = 'csv'
    tensorboard_dir = ''
    evaluation = False
    snapshots = True
    snapshot_path = "saved/"
    backbone = 'resnet50'
    epochs = 5
    steps = training_len//(batch_size)
    weighted_average = True

train_gen,valid_gen = create_generators(args,b.preprocess_image)

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
# Define our sequence of augmentation steps that will be applied to every image.
seq = iaa.Sequential(
    [
        #
        # Execute 1 to 9 of the following (less important) augmenters per
        # image. Don't execute all of them, as that would often be way too
        # strong.
        #
        iaa.SomeOf((1, 9),
            [

                        # Blur each image with varying strength using
                        # gaussian blur (sigma between 0 and .5),
                        # average/uniform blur (kernel size 1x1)
                        # median blur (kernel size 1x1).
                        iaa.OneOf([
                            iaa.GaussianBlur((0,0.5)),
                            iaa.AverageBlur(k=(1)),
                            iaa.MedianBlur(k=(1)),
                        ]),

                        # Sharpen each image, overlay the result with the original
                        # image using an alpha between 0 (no sharpening) and 1
                        # (full sharpening effect).
                        iaa.Sharpen(alpha=(0, 0.25), lightness=(0.75, 1.5)),

                        # Add gaussian noise to some images.
                        # In 50% of these cases, the noise is randomly sampled per
                        # channel and pixel.
                        # In the other 50% of all cases it is sampled once per
                        # pixel (i.e. brightness change).
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.01*255), per_channel=0.5
                        ),

                        # Either drop randomly 1 to 10% of all pixels (i.e. set
                        # them to black) or drop them on an image with 2-5% percent
                        # of the original size, leading to large dropped
                        # rectangles.
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.1), per_channel=0.5),
                            iaa.CoarseDropout(
                                (0.03, 0.15), size_percent=(0.02, 0.05),
                                per_channel=0.2
                            ),
                        ]),

                        # Add a value of -5 to 5 to each pixel.
                        iaa.Add((-5, 5), per_channel=0.5),

                        # Change brightness of images (85-115% of original value).
                        iaa.Multiply((0.85, 1.15), per_channel=0.5),

                        # Improve or worsen the contrast of images.
                        iaa.ContrastNormalization((0.75, 1.25), per_channel=0.5),

                        # Convert each image to grayscale and then overlay the
                        # result with the original with random alpha. I.e. remove
                        # colors with varying strengths.
                        iaa.Grayscale(alpha=(0.0, 0.25)),

                        # In some images distort local areas with varying strength.
                        sometimes(iaa.PiecewiseAffine(scale=(0.001, 0.01)))
                    ],
            # do all of the above augmentations in random order
            random_order=True
        )
    ],
    # do all of the above augmentations in random order
    random_order=True
)

def augment_train_gen(train_gen,visualize=False):
    '''
    Creates a generator using another generator with applied image augmentation.
    Args
        train_gen  : keras-retinanet generator object.
        visualize  : Boolean; False will convert bounding boxes to their anchor box targets for the model.
    '''
    imgs = []
    boxes = []
    targets = []
    size = train_gen.size()
    idx = 0
    while True:
        while len(imgs) < args.batch_size:
            image       = train_gen.load_image(idx % size)
            annotations = train_gen.load_annotations(idx % size)
            image,annotations = train_gen.random_transform_group_entry(image,annotations)
            imgs.append(image)            
            boxes.append(annotations['bboxes'])
            targets.append(annotations)
            idx += 1
        if visualize:
            imgs = seq.augment_images(imgs)
            im2 = np.array(imgs)
            boxes = np.array(boxes)
            yield im2,boxes
        else:
            imgs = seq.augment_images(imgs)
            imgs,targets = train_gen.preprocess_group(imgs,targets)
            imgs = train_gen.compute_inputs(imgs)
            targets = train_gen.compute_targets(imgs,targets)
            im2 = np.array(imgs)
            yield im2,targets
        imgs = []
        boxes = []
        targets = []
        

model, training_model, prediction_model = create_models(
            backbone_retinanet=b.retinanet,
            num_classes=train_gen.num_classes(),
            weights=None,
            multi_gpu=False,
            freeze_backbone=True,
            lr=1e-3,
            config=args.config
        )

callbacks = create_callbacks(
    model,
    training_model,
    prediction_model,
    valid_gen,
    args,
)

training_model.load_weights('resnet50_coco_best_v2.1.0.h5',skip_mismatch=True,by_name=True)

import datetime 
print("Here we go... " + str(datetime.datetime.now()))
training_model.fit_generator(generator=augment_train_gen(train_gen),
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,)
print("That was fun. " + str(datetime.datetime.now()))