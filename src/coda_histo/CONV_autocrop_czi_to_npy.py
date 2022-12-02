"""
This is a program that takes in czi files with multiple tissue samples in one image and breaks them into individual
images and saves as npy files. This is only tuned for spinal cord tissue at the moment. Should be refactored for more
flexibility.

I think moving forward histo team is just going to put single tissue samples in single images, so coordinate with them
before diving too hard into this.

"""
import javabridge
import bioformats as bf
import numpy as np
import pandas as pd
import time, os, sys
from urllib.parse import urlparse
import skimage.io
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage import color
import seaborn as sns
from pathlib import Path
import cv2
import timeit
import random as rng
from tqdm import tqdm
from tqdm import trange
import sys
import argparse
from coda_histo import histo_analysis as ha

javabridge.start_vm(class_path=bf.JARS)

series_num = 5

path = r'/media/tom/Rapid/2020-0006 Ventral horn/Naive spinal cord/'

files = [file.as_posix() for file in Path(path).glob('*.czi')]

def bound_box_area(boundbox):
    areas = []
    for i in range(len(boundbox)):
        areas.append(boundbox[i][2] * boundbox[i][3])
    return areas


df = pd.read_csv(r'/media/tom/Rapid/2020-0006 Ventral horn/Naive spinal cord/xml_metadata.csv')

df = df[(df['image_name'] != 'label image') & (df['image_name'] != 'macro image')]

lastx = 0

for image_i in trange(len(df)):

    sizex = df['sizex'].iloc[image_i]

    if sizex > lastx:

        img = bf.load_image(df['file'].iloc[image_i + series_num], series=df['index_'].iloc[image_i + series_num],
                            rescale=False)  # load the image
        tissue_mask = ha.tissue_mask_from_morph(img, 2, 8, thresh=80, plot=False)

        # img_ = np.uint8(img[:,:,4])
        img_ = np.uint8(tissue_mask)
        canny_output = cv2.Canny(img_, 1, 1 * 2)

        contours, heirarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours_poly = [None] * len(contours)
        boundRect = [None] * len(contours)
        centers = [None] * len(contours)
        radius = [None] * len(contours)
        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])
            centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

        areas = bound_box_area(boundRect)

        boxes = []
        contours_ = []

        last = [0, 0]

        for i in range(len(contours)):

            if areas[i] > 5000:

                rect = [boundRect[i][0], boundRect[i][0] + boundRect[i][2]], [boundRect[i][1],
                                                                              boundRect[i][1] + boundRect[i][3]]

                if np.abs(np.sum(np.subtract(last, rect))) > 100:
                    boxes.append(rect)
                    contours_.append(contours[i])

                last = rect

        img_large = bf.load_image(df['file'].iloc[image_i + 1], series=df['index_'].iloc[image_i + 1], rescale=False)

        for i, box in enumerate(boxes):
            box_ = np.multiply(box, 16)

            img_save = img_large[box_[1][0]:box_[1][1], box_[0][0]:box_[0][1], :]

            np.save(path + 'npy/' + df['image_name'].iloc[
                image_i] + '_slice' + str(i), img_save)

            plt.imshow(img_save[:,:,4])
            plt.savefig(path + 'npy/' + df['image_name'].iloc[image_i] + '_slice' + str(i) + '.png')
            plt.close()

    lastx = sizex
