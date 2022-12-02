import javabridge
import bioformats as bf
import numpy as np

javabridge.start_vm(class_path=bf.JARS)

import numpy as np
import pandas as pd
import pickle
import time, os, sys
from urllib.parse import urlparse
import skimage.io
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage import color
import seaborn as sns
from pathlib import Path
import cv2
from tqdm import tqdm
from tqdm import trange

from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import morphology
import sys
import re
from coda_histo import histo_analysis as ha
from scipy import ndimage

from skimage.transform import resize
from skimage import img_as_bool

mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams.update({'font.size': 22})


def csv_to_mask(df_, sizex_, sizey_):
    csv_ = df_.copy()
    xys = np.flip(csv_.values[:, 0:2], axis=1)
    # xys = csv_.values[:,0:2]
    mask_ = np.zeros((sizey_, sizex_))

    for xy in zip(xys):
        # print(xy)
        mask_[xy[0][0] - 1, xy[0][1] - 1] = 1
    return mask_


def match_csv_to_czi(csv_filename=str, czi_path_=str):
    return czi_path_ + csv_filename.split('/')[-1].split(' ')[0]


def get_image_name_in_csv_title(file_name=str):
    return file_name.split('/')[-1].split(' ')[-2] + ' ' + file_name.split('/')[-1].split(' ')[-1][:-6]


def get_image_index(csv_filename, image_object):
    ''' Returns the index of the image that the csv was created from '''
    image_ = get_image_name_in_csv_title(csv_filename)
    series_ = image_object.image_count
    for i in range(series_):
        # print(image_object.image(i).Name, '    ', image_)
        if image_object.image(i).Name == image_:
            return i


def get_sizex(image_object, index_=int):
    return image_object.image(index_).Pixels.SizeX


def get_sizey(image_object, index_=int):
    return image_object.image(index_).Pixels.SizeY


def get_image_name(image_object, index_=int):
    return image_object.image(index_).Name


def index_of_high_res_image(image_object, index_=int):
    '''
    Works backwards to find the index of the high resolution image.
    returns the index and scale factor btw current image and high res image
    '''
    last = get_sizex(image_object, index_)

    if index_ > 0:
        for i_ in reversed(range(index_ - 1)):
            if last > get_sizex(image_object, i_):
                return i_ + 1
            else:
                last = get_sizex(image_object, i_)


thresh = 1418
kernel1 = 30
kernel2 = 15
kernel1 = disk(kernel1)
kernel2 = disk(kernel2)

czi_path = r'/media/tom/Rapid/2020_0010_DAPI_GFP_GFAP_NeuN_axio_11.1.20_CD/'
csv_path = r'/media/tom/Rapid/2020_0010_DAPI_GFP_GFAP_NeuN_axio_11.1.20_CD/to_do/'

df = pd.read_csv(czi_path + 'xml_metadata')
df = df.rename(columns={'index': 'index_'})
df['index_'] = df['index_'].astype(int)

df_csv = pd.DataFrame()

files = [file.as_posix() for file in Path(csv_path).glob('*.csv')]

#############Start for loop

for j, file in tqdm(enumerate(files)):

    img_name = get_image_name_in_csv_title(file)

    rat = ha.get_rat_number(file)

    slide_number = file.split(' ')[-2][-8:-6]

    #     if slide_number == '42' or slide_number == '43':

    df_ind = df[df['image_name'] == img_name].index[0]
    sizex = df[df['image_name'] == img_name]['sizex'].values[0]
    sizey = df[df['image_name'] == img_name]['sizey'].values[0]

    csv = pd.read_csv(file)

    mask = csv_to_mask(csv, int(sizex), int(sizey))

    lastx = sizex

    for i in reversed(range(df_ind - 1)):
        if df['index_'].iloc[i] == 0:
            index = i + 3
            break
        if lastx > df['sizex'].iloc[i]:
            index = i + 1 + 3
            break
        else:
            lastx = df['sizex'].iloc[i]

    img = bf.load_image(df['file'].iloc[index], series=df['index_'].iloc[index], rescale=False)

    mask_resize = img_as_bool(resize(mask, (np.shape(img)[0], np.shape(img)[1])))

    cropped = np.multiply(mask_resize, img[:, :, 2])
    threshed = np.clip(cropped, thresh, np.percentile(img[:, :, 2], 99.9))

    scaled = threshed - thresh
    img_thresh = (scaled > 1) * 1

    img2 = cv2.morphologyEx(np.uint8(img_thresh), cv2.MORPH_CLOSE, kernel2)
    img3 = cv2.morphologyEx(np.uint8(img2), cv2.MORPH_OPEN, kernel2)
    img4 = cv2.morphologyEx(np.uint8(img3), cv2.MORPH_CLOSE, kernel1)
    # img4 = cv2.dilate(np.uint8(img3),kernel2,iterations = 3)
    img5 = morphology.remove_small_objects(img4.astype(bool), 200 ** 2)

    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    ax = axs.flatten()
    imgsc = ha.scale_pixels(img[:, :, 2])
    ax[0].imshow(imgsc)
    ax[1].imshow(img_thresh)
    ax[0].set_title('Scaled (Deceptive)')
    ax[1].set_title('Thresholded')
    ax[2].set_title('Calc Area')
    ax[2].imshow(img5, alpha=1, cmap='spring')
    ax[2].imshow(imgsc, alpha=0.7, cmap='cividis')
    ax[3].imshow(mask)
    plt.savefig(czi_path + '/imgs/' + img_name + '.png')
    plt.close()

    masked = np.multiply(img5, img[:, :, 2])

    masks = masked[masked > 0]

    pixels = img[img[:, :, 2] > 0]

    #         fig, ax = plt.subplots(1,1,figsize=(10,10))
    #         ax.imshow(img_cropped)
    #         ax.imshow(mask_resize, alpha=0.5)

    try:
        ninetieth = np.percentile(masks, 90)
    except:
        ninetieth = 0

    for_thresh = np.percentile(cropped[cropped > 0], 99.99)

    if np.sum(img5)/np.sum(mask) > 1:
        percent_coverage = 100
    else: percent_coverage = 100*np.sum(img5)/np.sum(mask)

    dic = {'file': df['file'].iloc[index],
           'slide_number': slide_number,
           'image_name': img_name,
           'image_num': df['image_name'].iloc[index][-3:],
           'image_index': df['index_'].iloc[index],
           'rat': rat,
           'infected_area': np.sum(img5),
           'percent_coverage': percent_coverage,
           'pixel_volume': np.sum(masked),
           'norm_pixel_volume': np.sum(masked) / len(pixels),
           '90th_percentile': ninetieth,
           # '99.99': for_thresh,
           }

    df_csv = df_csv.append(dic, ignore_index=True)

df_csv.to_csv(r'/media/tom/Rapid/2020_0010_DAPI_GFP_GFAP_NeuN_axio_11.1.20_CD/summary.csv')

quit()