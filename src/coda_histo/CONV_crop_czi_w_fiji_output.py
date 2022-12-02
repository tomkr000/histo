'''

This program uses the output of manual cropping of images in fiji and converts the czi file to a cropped npy version.

Important note: It doesn't matter which zoom level you perform cropping at - the program selects the image you cropped
and scales to the largest zoom level for the image. This allows faster cropping as you don't have to wait for larger zoom
levels to load.


########################################################
How to crop an image for use with this program using Fiji:
1) open the image in Fiji (the image that is ~1000 to 2000 pixels on a side)
2) duplicate the image so you don't modify it (ctrl + shift + d)
3) select the polygon tool
4) draw a polygon around the hippocampus (or whatever structure you're cropping)
5) select "edit" -> "clear outside"
6) select "analyze" -> "tools" -> "Save XY Coordinates"
7) that will bring up a dialog box to save a csv of XY coordinates
8) don't modify the automatic file name, just save directly as is
9) close the image
#########################################################
'''

import numpy as np
import pandas as pd
import pickle
import time, os, sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage import img_as_bool
from skimage.transform import resize
import seaborn as sns
from pathlib import Path
import cv2
from tqdm import tqdm
from tqdm import trange
import sys
import re
import bioformats as bf
import javabridge
import argparse
from coda_histo import histo_analysis as ha


def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('directory', type=str,
                        help='''name of directory to process''')

    parser.add_argument('croppings_directory', type=str,
                        help='''name of directory in which cropping csvs from fiji are saved process''')

    args = parser.parse_args()
    return args


def get_image_name_fiji(file_):
    '''
    Get the file
    '''
    split = file.split('/')[-1].split('-')
    image_name_ = split[2] + '-' + split[3]
    return image_name_[1:]

def create_metadata_csv(path, csv_name='all_metadata.csv', filetype='czi'):
    '''
    This is used elsewhere and can be put into histo_analysis
    Makes dataframe with file, image_name (unique identifier), index and size of all czi files and images within
    '''
    files = Path(path).glob('*' + filetype)
    df_ = pd.DataFrame()
    files = [file.as_posix() for file in files]

    for file in tqdm(files):
        o = bf.OMEXML(bf.get_omexml_metadata(file))
        series = o.image_count
        for i in range(series):
            dic = {'file': file,
                   'image_name': ha.get_image_name(o, i),
                   'index_': int(i),
                   'sizex': ha.get_sizex(o, i),
                   'sizey': ha.get_sizey(o, i)}
            df_ = df_.append(dic, ignore_index=True)

    df_ = df_.query('image_name != "label image"').query('image_name != "macro image"')
    print(df_)
    df_.reset_index(inplace=True)
    df_.drop(columns='index', inplace=True)
    # df_.to_csv(path + csv_name)
    return df_


if __name__ == '__main__':
    javabridge.start_vm(class_path=bf.JARS, max_heap_size='32G')

    args = parseArguments()

    folder = args.directory

    crop_folder = args.croppings_directory

    df = create_metadata_csv(folder)

    files = [file.as_posix() for file in Path(crop_folder).glob('*.csv')]

    Path(folder + '/cropped/').mkdir(parents=True, exist_ok=True)

    for j, file in tqdm(enumerate(files)):
        img_name = get_image_name_fiji(file)

        slide_number = file.split(' ')[-2][-8:-6]

        #     if slide_number == '42' or slide_number == '43':

        df_ind = df[df['image_name'] == img_name].index[0]
        sizex = df[df['image_name'] == img_name]['sizex'].values[0]
        sizey = df[df['image_name'] == img_name]['sizey'].values[0]

        csv = pd.read_csv(file)

        mask = ha.csv_to_mask(csv, int(sizex), int(sizey))

        # Fine the largest version of the image that the cropping was made from in the czi file
        lastx = sizex
        for i in reversed(range(df_ind - 1)):
            if df['index_'].iloc[i] == 0:
                index = i
                break
            if lastx > df['sizex'].iloc[i]:
                index = i + 1
                break
            else:
                lastx = df['sizex'].iloc[i]

        # load the largest image
        img = bf.load_image(df['file'].iloc[index], series=df['index_'].iloc[index], rescale=False)

        # resize the mask created in fiji to the largest image size from the czi file
        mask_resize = img_as_bool(resize(mask, (np.shape(img)[0], np.shape(img)[1])))

        x_max = np.where(mask_resize == 1)[0].max()
        x_min = np.where(mask_resize == 1)[0].min()
        y_max = np.where(mask_resize == 1)[1].max()
        y_min = np.where(mask_resize == 1)[1].min()

        img_save = img[x_min:x_max, y_min:y_max, :]

        np.save(folder + 'cropped/' + img_name + file[-6:-4], img_save)






