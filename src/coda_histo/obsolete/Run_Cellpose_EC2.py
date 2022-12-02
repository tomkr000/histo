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
javabridge.start_vm(class_path=bf.JARS, max_heap_size='32G')


parser = argparse.ArgumentParser(description='Run Cellpose')


parser.add_argument('directory', type=str,
                    help='''name of directory to process, must be from mounted S3 dir,
                            ex: /histo/''')
parser.add_argument('params_name', type=str,
                    help = 'name of parameters in airtable row to use')

parser.add_argument('-csv', '--load_metadata_csv',
                    action = 'store_true',
                    help='load csv from previous run')
parser.add_argument('-o', '--overwrite_old_files',
                    action = 'store_true',
                    help='overwrite pkl and airtable records')

args = parser.parse_args()

path = Path(args.directory)

params_name = args.params_name

meta_file = r'/metadata.csv'

metadata_filepath = path.as_posix() + meta_file

if args.load_metadata_csv:
    df_to_process = pd.read_csv(metadata_filepath)
else:
    df_to_process = ha.create_czi_master_df(path, 'all', params_name) # pass path object and number of files to process, "all" if all files
    df_to_process.to_csv(metadata_filepath)

print(df_to_process)

####################################################################################
## FILL IN EXTRA COLUMNS

params = ha.CellposeParams(params_name) # initiate a params object to fill the df from

print('DONE WITH DATAFRAME')

pkl_path = Path('/histo/pkl_files')

if args.overwrite_old_files:
    pkl_files = []
else:
    pkl_files = [file.as_posix() for file in pkl_path.glob('*.pkl')]

###################################################################################
## START OF MASSIVE FOR LOOP

for iii in tqdm(range(len(df_to_process))):

    if df_to_process.iloc[iii]['s3_pkl_path'] not in pkl_files:

        print('processing ' + df_to_process.iloc[iii]['s3_pkl_path'])
        params.designate_within_slide_params(df_to_process.iloc[iii]['s3_file_path'])

        # used to load from czi or lif file, now transform into .npy first and go from there.
        # img = bf.load_image(df_to_process.iloc[iii]['s3_file_path'],
        #                 series=df_to_process.iloc[iii]['series_num'],
        #                 rescale=False)  # load the image



    # IMAGE PROCESSING ##############################################################################
        img2 = cv2.resize(img,
                          dsize=(int(round(np.shape(img)[1]/params.scale_factor)),
                                 int(round(np.shape(img)[0]/params.scale_factor))),
                          interpolation=cv2.INTER_CUBIC)

        img1 = ha.image_preprocessing(img2, [params.cyto_ch, params.entropy_ch])

        # save the image if it hasn't already been created
        np.save(df_to_process.iloc[iii]['s3_image_path'],
            img1, allow_pickle=False)

        start = time.time()
        entr_mask = ha.tissue_mask_from_morph(img1,
                                              params.entropy_ch,
                                              params.morph_kernel)
        print('tissue finding took  >>>> ' +str(time.time()-start))

        img_cell = ha.create_tif_stack(img1, [params.cyto_ch])

        start = time.time()

        model = models.Cellpose(gpu=True, model_type=params.model_type)

        imgs = [img_cell]
        cp_masks, flows, styles, diams = model.eval(imgs,
                                             diameter=params.diameter,
                                             channels=params.cellpose_ch,
                                             do_3D=False,
                                             flow_threshold=params.flow_threshold,
                                             cellprob_threshold=params.cellprob_threshold)
        print('Cells???? >>>>   ' + str(np.max(cp_masks) != 0))

        print('cellpose took >>>>> ' + str(time.time() - start))

        if np.max(cp_masks) != 0:

            start = time.time()
            df_pixels = ha.make_aligned_pix_df(img1, entr_mask, cp_masks[0], params)
            df_pixels = ha.get_slide_and_tissue_background_means(df_pixels, params)
            print('df_pixel dataframe took >>>>>>   ' + str(time.time() - start))

            start = time.time()
            df_cells = ha.get_pixel_stats_by_cell(df_pixels, params)

            df_cells = df_cells.drop([0])

            # get x y values for each cell
            df_xy = pd.DataFrame(columns=['cp_masks', 'X', 'Y'])

            for i in range(1, df_pixels.cp_masks.max()):
                y, x = (df_pixels.cp_masks >= i).idxmax()[0], (df_pixels.cp_masks >= i).idxmax()[1]
                df_xy = df_xy.append({'cp_masks': i, 'X': x, 'Y': y}, ignore_index=True)

            df_cells = pd.merge(df_cells, df_xy, on='cp_masks')
            df_cells.set_index('cp_masks')

            print('df_cells dataframe took >>>>>>     ' + str(time.time() - start))

        else:
            df_pixels = 'None'
            df_cells = 'None'
            print(unique_id, ' <<<<< HAD NO CELLS')


        data_dic = {'df_pixels': df_pixels,
                'df_cells': df_cells,
                'img': img1,
                'entr_mask': entr_mask,
                'cp_mask': cp_masks,
                'params': params}

        with open(df_to_process.iloc[iii].s3_pkl_path, 'wb') as f:
            pickle.dump(data_dic, f)

        print('DONE WITH ' + str(iii) + ' of ' + str(len(df_to_process)))

if 'Unnamed: 0' in df_to_process.columns:
    df_to_process.drop(columns=['Unnamed: 0'], inplace=True)
airtable = codaAirtable('Histology')
airtable.upload_df_to_airtable(df_to_process, primary_key = 'unique_id', overwrite=True)

javabridge.kill_vm()
