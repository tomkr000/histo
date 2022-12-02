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

parser = argparse.ArgumentParser(description='Run Cellpose')

parser.add_argument('directory', type=str,
                    help='''name of directory to process, must be from mounted S3 dir,
                            ex: /histo/''')
parser.add_argument('params_name', type=str,
                    help='name of parameters in airtable row to use')

parser.add_argument('-o', '--overwrite_old_files',
                    action='store_true',
                    help='overwrite pkl and airtable records')

args = parser.parse_args()

path = Path(args.directory)

params_name = args.params_name

files = [file.as_posix() for file in path.glob('*.npy')]

params = ha.CellposeParams(params_name) # initiate a params object to fill the df from

# need a second set of params for 2nd run of cellpose.... lots of doubles here.....
params2 = ha.CellposeParams('vhorn_params_2') # hardcoded this time... should fix

pkl_path = Path('/histo/pkl_files')
s3_pkl_prefix = r'/histo/pkl_files/'

airtable = codaAirtable('Histology')

if args.overwrite_old_files:
    pkl_files = []
else:
    pkl_files = [file.as_posix() for file in pkl_path.glob('*.pkl')]

###################################################################################
## START OF MASSIVE FOR LOOP

for iii in trange(len(files)):

    pkl_file = s3_pkl_prefix + files[iii].split('/')[-1] + '_' + params_name + '.pkl'

    if pkl_file not in pkl_files:

        print('processing >>> ' + files[iii])
        params.designate_within_slide_params(files[iii])

        airtable_dic = ha.create_histo_airtable_df(files[iii], params_name)

        img = np.load(files[iii])

        # Designate the tissue area to process
        tissue = ha.designate_calc_area_sc(img,
                                   params.cyto_ch,
                                   1000000,
                                   70,
                                   outlier_percent=99.999999,
                                   bottom_percent=99,
                                   bottom_thresh=18000)

        # multiply tissue area for cal by original cyto image to get only calc area
        img_proc = np.multiply(tissue, img[:,:,params.cyto_ch])

        # IMAGE PROCESSING ##############################################################################
        start = time.time()

        height = params.tiley
        width = params.tilex
        xs = int(np.ceil(np.shape(img)[0] / width))
        ys = int(np.ceil(np.shape(img)[1] / height))

        cp_masks = np.zeros((np.shape(img_proc)[0], np.shape(img_proc)[1]))

        model = models.Cellpose(gpu=True, model_type=params.model_type)

        #break the image into tiles and process each individually through cellpose
        #stitch together at the end (this makes the process O*N instead of O^N in space and time)
        for i in range(xs):
            for j in range(ys):

                if i < xs - 1 and j < ys - 1:
                    tile = img_proc[i * width:(i + 1) * width, j * height:(j + 1) * height]
                elif i < xs - 1 and j == ys - 1:
                    tile = img_proc[i * width:(i + 1) * width, j * height:]
                elif i == xs - 1 and j < ys - 1:
                    tile = img_proc[i * width:, j * height:(j + 1) * height]
                else:
                    tile = img_proc[i * width:, j * height:]

                if np.max(tile) != 0:

                    img1 = ha.single_image_preprocessing(tile,
                                                          outlier_percent=params.outlier_percent,
                                                          bottom_percent=params.bottom_percent,
                                                          bottom_thresh=params.bottom_thresh)

                    img2 = ha.create_tif_stack(img1, [params.cyto_ch, params.cyto_ch, params.cyto_ch])
                    imgs = [img2]
                    cp_masks1, flows, styles, diams = model.eval(imgs,
                                                                diameter=params.diameter,
                                                                channels=params.cellpose_ch,
                                                                do_3D=False,
                                                                flow_threshold=params.flow_threshold,
                                                                cellprob_threshold=params.cellprob_threshold)

                    cp_masks2, flows, styles, diams = model.eval(imgs,
                                                                 diameter=params2.diameter,
                                                                 channels=params2.cellpose_ch,
                                                                 do_3D=False,
                                                                 flow_threshold=params2.flow_threshold,
                                                                 cellprob_threshold=params2.cellprob_threshold)

                    tile_masks = cp_masks1 + cp_masks2

                    if i < xs - 1 and j < ys - 1:
                        cp_masks[i * width:(i + 1) * width, j * height:(j + 1) * height] = tile_masks[0]
                    elif i < xs - 1 and j == ys - 1:
                        cp_masks[i * width:(i + 1) * width, j * height:] = tile_masks[0]
                    elif i == xs - 1 and j < ys - 1:
                        cp_masks[i * width:, j * height:(j + 1) * height] = tile_masks[0]
                    else:
                        cp_masks[i * width:, j * height:] = tile_masks[0]

        cp_masks_ = (cp_masks > 0)*1
        num_labels, cp_masks = cv2.connectedComponents(np.uint8(cp_masks_))

        print('Cells???? >>>>   ' + str(np.max(cp_masks) != 0))

        print('cellpose took >>>>> ' + str(time.time() - start))

        if np.max(cp_masks) != 0:

            start = time.time()
            cp_masks = cp_masks.astype(int)
            df_pixels = ha.make_aligned_pix_df(img, cp_masks, params)
            #df_pixels = ha.get_slide_and_tissue_background_means(df_pixels, params)
            print('df_pixel dataframe took >>>>>>   ' + str(time.time() - start))

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
            print(files[iii], ' <<<<< HAD NO CELLS')

        data_dic = {'df_pixels': df_pixels,
                'df_cells': df_cells,
                'img': img,
                'tissue_mask': tissue,
                'cp_mask': cp_masks,
                'params': params}

        with open(pkl_file, 'wb') as f:
            pickle.dump(data_dic, f)

        df = pd.DataFrame()
        df = df.append(airtable_dic, ignore_index=True)
        airtable = codaAirtable('Histology')
        airtable.upload_df_to_airtable(df, primary_key='unique_id', overwrite=True)
    # airtable = codaAirtable('Histology')
    # airtable.upload_df_to_airtable(df_to_process, primary_key = 'unique_id', overwrite=True)

javabridge.kill_vm()
