'''
FOR RUNNING ON EC2 INSTANCE WITHOUT METAFLOW. THIS JUST GETS THE JOB DONE.

NOTE: YOU NEED TO
'''
from coda_histo import histo_analysis as ha
import numpy as np
from coda_discovery.utils import codaAirtable
from pathlib import Path
import bioformats as bf
from tqdm import tqdm
import pickle
import pandas as pd
import time
import argparse
import cv2
from cellpose import models
import javabridge
from tqdm import trange


pkl_path = Path('/histo/pkl_files')
s3_pkl_prefix = r'/histo/pkl_files/'

airtable = codaAirtable('Histology')
df_airtable = airtable.to_df()

df_airtable = df_airtable[df_airtable['analysis_params'] == 'vhorn_params_ctrl']
files = df_airtable.s3_pkl_path.to_list()

for file in tqdm(files):

    print(file)
    data = pickle.load( open( file, 'rb'))

    img = data['img']
    cp_masks = data['cp_mask']
    tissue = data['tissue_mask']
    params = data['params']

    if np.max(cp_masks) != 0:

        start = time.time()
        cp_masks = cp_masks.astype(int)
        df_pixels = ha.make_aligned_pix_df(img, cp_masks, params)
        #df_pixels = ha.get_slide_and_tissue_background_means(df_pixels, params)
        # print('df_pixel dataframe took >>>>>>   ' + str(time.time() - start))

        start = time.time()
        df_cells = ha.get_pixel_stats_by_cell(df_pixels, params)

        df_cells = df_cells.drop([0])

        # get x y values for each cell
        df_xy = pd.DataFrame(columns=['cp_masks', 'X', 'Y'])

        for i in range(1, int(df_pixels.cp_masks.max())):
            y, x = (df_pixels.cp_masks >= i).idxmax()[0], (df_pixels.cp_masks >= i).idxmax()[1]
            df_xy = df_xy.append({'cp_masks': i, 'X': x, 'Y': y}, ignore_index=True)

        df_cells = pd.merge(df_cells, df_xy, on='cp_masks')
        df_cells.set_index('cp_masks')

        print('df_cells dataframe took >>>>>>     ' + str(time.time() - start))

    else:
        df_pixels = 'None'
        df_cells = 'None'
        print(file, ' <<<<< HAD NO CELLS')

    data_dic = {'df_pixels': df_pixels,
            'df_cells': df_cells,
            'img': img,
            'tissue_mask': tissue,
            'cp_mask': cp_masks,
            'params': params}

    with open(file, 'wb') as f:
        pickle.dump(data_dic, f)