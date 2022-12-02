import bioformats as bf
import numpy as np
from pathlib import Path
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
import re
from skimage.filters.rank import entropy
from skimage.morphology import disk
import sys
import boto3
from coda_discovery.utils import codaAirtable
import time

def parse_filename(filepath):
    ''' break filename into component parts for matching string patterns'''
    filename = filepath.split('/')[-1]
    namespace = re.split(' |_|-', filename)
    return namespace


def get_rat_number(filepath):
    '''Return rat number from filename'''
    namespace = parse_filename(filepath)
    for name in namespace:
        try: 
            if len(name) > 0 and name[0] == 'R':
                count = 0
                for i in name[1:]:
                    if i.isdigit:
                        count += 1
                    if count == 4:
                        return name.split('.')[0]
        except: return 'none'


def designate_controls(Rat):
    '''
    Deprecated, using either 'y' in controls column in Histology airtable, or 'Naive control' in Animal Log
    '''
    if Rat == 'R0000' or Rat =='R0212':
        return 'y'
    else:
        return 'n'


def get_filename(filepath):
    return os.path.basename(filepath)


def return_drg_list():
    ''' create list of the order the drg's are in on the slide '''
    r = ['R3', 'R4', 'R5', 'R6']
    l = ['L3', 'L4', 'L5', 'L6']
    rl = 3*r + 3*l
    return rl


def create_histo_airtable_df(filepath_, params_name):
    """
    Hack to do spinal cord with - REFACTOR!!!!!
    Example filepath_ : '/histo/2020_0006_vhorn/Transverse VH/npy/20201111__R0396 VHorn Profile7 Slide11.czi #01_slice0.npy',
    """
    s3_pkl_prefix = r'/histo/pkl_files/'
    pkl_file = s3_pkl_prefix + filepath_.split('/')[-1] + '_' + params_name + '.pkl'
    params_ = CellposeParams(params_name)
    params_.designate_within_slide_params(filepath_)
    rat = get_rat_number(filepath_)
    filename = filepath_.split('/')[-1].split('.')[0] + '.czi'
    series = float(''.join(c for c in filepath_.split('#')[1][0:2] if c.isdigit())) - 1
    unique_id = filepath_.split('/')[-1] + '_' + params_name
    image_name = filepath_.split('/')[-1]

    dic = {'imagename': image_name,
                              'filename': filename,
                            's3_file_path': filepath_,
                              's3_pkl_path': pkl_file,
                              's3_image_path': filepath_,
                              'unique_id': unique_id,
                              'analysis_params': params_name,
                            'series_num': series,
                              'Rat_': rat,
                              'microscope_profile_': params_.profile,
                                'tissue': params_.tissue}
    return dic


def create_czi_file_df(filepath_, params_name):
    '''
    Create a dataframe of which images from a czi file should be processed
    Used to sort extraneous images... soooo dumb.
    When opening images, use SizeX to check that it is the correct image using image dims
    filepath_ : path object including file
    '''
    s3_pkl_prefix = r'/histo/pkl_files/'
    s3_image_prefix = '/histo/processed_images/'

    params_ = CellposeParams('histo_params', params_name)
    params_.designate_within_slide_params(filepath_)

    o = bf.OMEXML(bf.get_omexml_metadata(filepath_))
    series = o.image_count
    df_ = pd.DataFrame()

    rat = get_rat_number(filepath_)
    naive_control = designate_controls(rat)
    previous_size = 0
    filename = get_filename(filepath_)

    for i in range(series-2):

        if o.image(i).Pixels.get_SizeX() * o.image(i).Pixels.get_SizeY() > previous_size:

            df_ = df_.append({'imagename': o.image(i).Name,
                              'filename': get_filename(filepath_),
                            's3_file_path': filepath_,
                              's3_pkl_path': s3_pkl_prefix + o.image(i).Name + '_num' + str(i)+'_'+ params_name + '.pkl',
                              's3_image_path': s3_image_prefix + o.image(i).Name + '_num'+str(i)+'.npy',
                              'unique_id': o.image(i).Name + '_num'+str(i)+'_'+params_name,
                              'analysis_params': params_name,
                            'series_num': i,
                              'Rat_': rat,
                              'microscope_profile_': params_.profile,
                                'naive_control': naive_control,
                            'X': o.image(i).Pixels.Plane().get_PositionX(),
                            'Y': o.image(i).Pixels.Plane().get_PositionY()},
                            ignore_index=True)
            previous_size = o.image(i).Pixels.get_SizeX() * o.image(i).Pixels.get_SizeY()

        else:
            previous_size = o.image(i).Pixels.get_SizeX() * o.image(i).Pixels.get_SizeY()

    print('METADATA DONE FOR ' + filename)
    return df_


def create_czi_master_df(dirpath_, n_files, params_):
    '''
    dirpath_ : path object to directory
    n_files : int, number of files to run on, 'all' process all files
    '''

    files_ = dirpath_.glob('*.czi')
    files = [file.as_posix() for file in files_]

    if n_files == 'all':
        n_files = len(files)

    df_master = pd.DataFrame()

    for file in files[0:n_files]:

        df_temp = create_czi_file_df(file, params_)
        df_master=pd.concat([df_master, df_temp])

    return df_master



################## PARAMS OBJECT ##########################################################################

class CellposeParams:
    '''
    Class for storing experimental/processing parameters in. Initializes by pulling values from the "Name" row in the
    histo_params table. Then you can use that to taylor to specific slides/channels/etc.

    airtable_tablename: str , name of airtable table to pull from (default "histo_params")
    params_name: str , params (first column) to pull from airtable

    '''
    def __init__(self, params_name, airtable_tablename="histo_params"):

        airtable = codaAirtable(airtable_tablename)
        params = params_name
        record = airtable.match('Name', params)
        params_dic = airtable.get(record.get('id')).get('fields')

        self.exp_name = params_dic.get('exp_name')
        self.tissue = params_dic.get('tissue')

        # Cellpose params:
        self.model_type =  params_dic.get('model_type')
        # set self.cellpose_ch to [0,0] to sort on just cytoplasm using 'cyto' model
        # set to [0,1] to sort with 2 channels, cyto and nuclei, using 'cyto' model
        # set to [1,0] to sort with 2 channels, nuclei and cyto, using 'nuclei' model
        # set to [1,1] to sort with 1 channel, nuclei, using 'nuclei' model
        ch = params_dic.get('cellpose_ch')
        self.cellpose_ch = [int(ch[1]), int(ch[3])]
        self.diameter = params_dic.get('diameter')
        self.flow_threshold = params_dic.get('flow_threshold')
        self.cellprob_threshold = params_dic.get('cellprob_threshold')

        # Image processing params:
        self.morph_kernel = params_dic.get('morph_kernel')
        self.entropy_kernel = params_dic.get('entropy_kernel')
        self.entropy_thresh = params_dic.get('entropy_thresh')
        self.entropy_disk = params_dic.get('entropy_disk')
        self.outlier_percent = params_dic.get('outlier_percent')
        self.scale_factor = params_dic.get('scale_factor')
        self.tilex = params_dic.get('tilex')
        self.tiley = params_dic.get('tiley')
        self.bottom_percent = params_dic.get('bottom_percent')
        self.bottom_thresh = int(params_dic.get('bottom_thresh'))

        # Channel params:
        self.entropy_fluor = params_dic.get('entropy_fluor')
        self.cyto_fluor = params_dic.get('cyto_fluor')
        self.nuclei_fluor = params_dic.get('nuclei_fluor')

        # Channels to calc fluorescence from !!!!!!! IMPORTANT !!!!!!
        self.pix_fluor = params_dic.get('pix_fluor')

        self.channel_df = False


    def designate_within_slide_params(self, filepath_):
        '''
        Execute once you know which slide you are working with
        Designates which channel is used for cellpose and tissue finding
        '''
        self.filepath = filepath_
        self.microscope_profile_name(filepath_)
        self.get_channel_df()
        self.entropy_ch = self.get_channel_from_fluor(self.entropy_fluor)
        self.cyto_ch = self.get_channel_from_fluor(self.cyto_fluor)
        self.nuclei_ch = self.get_channel_from_fluor(self.nuclei_fluor)


    def get_channel_df(self):
        '''
        Return a df of the channels corresponding to their fluorophores
        Reads which profile from "Histology_Profiles" table in airtable and pulls in appropriate values
        '''
        profile_airtable = codaAirtable('Histology_Profiles')
        channel_df_ = profile_airtable.to_df()
        df__ = channel_df_[channel_df_['Name'] == self.profile]
        df__.reset_index(drop=True, inplace=True)
        self.channel_df = df__


    def microscope_profile_name(self, filepath_):
        '''
        Return the profile name to match with airtable Histology_Profiles table
        '''
        namespace = parse_filename(filepath_)
        profile = [x for x in namespace if 'Profile' in x][0]
        profile_ = profile.split('.')[0]
        self.profile = profile_


    def get_channel_from_fluor(self, fluor):
        '''
        obtain which channel to use for sorting given the fluorophore used
        Must match fluorophore listed in airtable
        '''

        # print('channel_df  >>>>  ',self.channel_df)
        channel_dic = {'Channel_1': 0,
                       'Channel_2': 1,
                       'Channel_3': 2,
                       'Channel_4': 3,
                       'Channel_5': 4,
                       'Channel_6': 5,
                       'Channel_7': 6}

        for column in self.channel_df.columns:
            if self.channel_df[column][0] == fluor:
                return channel_dic.get(column)



##################### IMAGE PROCESSING ####################################################


def clip_outlier_pixels(img_, outlier_percent):
    '''
    There are always a few pixels that are orders of magnitude brighter than all the others - use this function to
    clip them to a percentile of the population
    img_ : 2D array , usually the image
    outlier_percent : float (usually 99.999 or so), percentile above which pixel values are clipped
    '''
    img_out = np.clip(img_, 0, np.percentile(img_, outlier_percent))
    return img_out


def create_tif_stack(img_, channels_):
    '''
    make nd array to be passed to cellpose model for processing
    or any other thing you might need to convert to tif [m:n:3] dimension
    remember channel order!!!
    '''
    if len(np.shape(img_)) == 2:
        dim = np.zeros(np.shape(img_))
        img_out = img_
        img_out = np.dstack((img_out, dim))
        img_out = np.dstack((img_out, dim))
    else:
        dim = np.zeros(np.shape(img_[:,:,0]))
        img_out = img_[:,:,channels_]

        for i in range(3):
            if np.shape(img_out)[2] < 3:
                img_out = np.dstack((img_out, dim))

    return img_out


def scale_pixels(img, lower=.05, upper=.995, bitdepth=255):
    '''
    Rescale the pixel values so image is viewable.
    Use just for plotting!!!!!
    '''

    df = pd.DataFrame(img).stack()
    df = df.replace(0, np.NaN)
    lo = df.quantile(lower)
    up = df.quantile(upper)
    img1 = img - lo
    img1 = bitdepth*(img1/up)
    img1 = np.clip(img1, 0, bitdepth)
    img1 = img1.astype(int)
    return img1


def images_preprocessing(img_,
                         channels_,
                         outlier_percent_=[99.9],
                         bottom_percent_=[80],
                         bottom_thresh_=[300]):
    """
    First thing to do to the image upon loading.
    Resize, clip outlier pixels, threshold out background
    Return ND array of new size

    img_ : 3D array, X:Y:Channel
    channels_ : list-like of ints, channels to perform preprocessing on
    outlier_percent_ : list-like of floats, top percentile of each channel that designates outlier pixels
    bottom_percent : list-like of floats, bottom percentile of each channel below which pixel are set to 0
    bottom_thresh : list-like of floats, bottom threshold of each channel to set pixels below to 0
    """
    img_this = img_.copy()

    for i, channel_ in enumerate(channels_):
        img_this[:,:,channel_] = single_image_preprocessing(img_[:,:,channel_],
                                                            outlier_percent_[i],
                                                            bottom_percent_[i],
                                                            bottom_thresh_[i])
    return img_this


def single_image_preprocessing(img_,
                                outlier_percent,
                                bottom_percent,
                                bottom_thresh):
    """
    First thing to do to the image upon loading.
    clip outlier pixels, threshold out background
    Return ND array of new size

    img_ : 2D array, X:Y of pixels
    outlier_percent : float, top percentile that designates outlier pixels
    bottom_percent : float, bottom percentile of each channel below which pixel are set to 0
    bottom_thresh : float, bottom threshold of each channel to set pixels below to 0
    """
    img_this = img_.copy()
    img1 = clip_outlier_pixels(img_this, outlier_percent)
    img2 = subtract_background(img1, bottom_thresh)
    bottom = np.percentile(img2, bottom_percent)

    img3 = np.clip(img2,
                   bottom,
                   np.max(img2))
        # set lowest pixels to 0
    img1_ = img3 - bottom


    return img1_


def subtract_background(img, thresh=int):
    """
    Non-negative subtraction of a threshold value from all pixels

    img : 2D array, [X:Y] of pixels
    thresh : int, value to subtract from pixels
    """
    img1_ = img
    img1_[img1_ > thresh] -= thresh
    img1_[img1_ < thresh] = 0
    return img1_


def tissue_mask_from_entropy(img_, channel_, thresh_=7.2, kernel_=30, disk_=15, plot=False):
    '''
    Create a mask of where the tissue is in the image.

    Uses:
        from skimage.filters.rank import entropy
        from skimage.morphology import disk

    img_ : numpy array, multi-channel image
    channel_ : int, which channel of the image to make the mask from
    thresh : float64, the threshold of the entropy value to make the mask from
    kernel : int, size of the kernel, in pixels, used to enlarge and contract entropy mask to
                get rid of artifacts
    disk : int, size of the disk to use to calculate local entropy
    '''
    entr_img1 = entropy(img_[:,:,channel_], disk(disk_)) # calc local entropy of each pixel
    entr_img = (entr_img1 > thresh_) * 1 # threshold to get raw mask

    # this step enlarges the entropy mask then shrinks it
    kernel2D = np.ones((kernel_, kernel_),np.uint8) # create the kernels for morphing entropy mask
    entr_img2 = cv2.morphologyEx(np.uint8(entr_img), cv2.MORPH_OPEN, kernel2D) # enlarge ent mask
    mask_ = cv2.morphologyEx(entr_img2, cv2.MORPH_CLOSE, kernel2D) # shrink ent mask
    if plot:
        plot_entropy(img_, channel_, mask_, entr_img1)
    return mask_, entr_img1


def tissue_mask_from_morph(img_, channel_, kernel, thresh=50, plot=False):
    '''
    Create mask from where tissue is in the image just using pixel values
    and morphology.
    Best practice to use DAPI channel.

    Uses:
        from skimage.morphology import disk
    img_ : numpy array, multi-channel image
    channel_ : int, which channel of the image to make the mask from
    kernel : int, size of the kernel, in pixels, used to enlarge and contract entropy mask to
                get rid of artifacts
    thresh : int, threshold in either percentile or absolute pixel value to use to threshold the initial image
                if above 100, it's absolute; if below, it's percentile
    '''
    img1_ = img_[:,:,channel_]
    kernel1_ = kernel
    kernel2_ = kernel*2

    if thresh < 100:
        mask = img1_>np.percentile(img1_, thresh)
    else:
        mask = img1_>thresh
    mask = mask*1
    kernel1 = disk(kernel1_)
    kernel2 = disk(kernel2_)
    start=time.time()
    expand1 = cv2.morphologyEx(np.uint8(mask), cv2.MORPH_CLOSE, kernel1)
    #print('expansion1 took >>>>>>> ', time.time()-start)
    start=time.time()
    contract1 = cv2.morphologyEx(np.uint8(expand1), cv2.MORPH_OPEN, kernel1)
    #print('contract1 took >>>>>>>>  ', time.time()-start)
    start=time.time()
    expand2 = cv2.morphologyEx(np.uint8(contract1), cv2.MORPH_CLOSE, kernel2)
    #print('expansion2 took >>>>>>>   ', time.time() - start)
    if plot:
        fig, ax = plt.subplots(1,1,figsize=(8,8))
        ax.imshow(expand2)
        ax.imshow(img1_, cmap='gray', alpha=0.5)
    return expand2


def designate_calc_area_sc(img_,
                           channel_,
                           size_,
                            kernel,
                            outlier_percent=99.999999,
                            bottom_percent=99,
                            bottom_thresh=18000):
    """
    Specific function to designate where to run cellpose in the image area for transverse spinal sections.
    img_ : ND numpy array, image to be used. In this case, NeuN channel
    channel_ : int, channel to perform operation on 
    size_ : int, max area of objects to reject
    kernel : int, diameter of disk kernel for morph dilations
    outlier_percent : float, percentile above which pixels are scaled down to
    bottom_percent : float/int, percentile below which pixels => 0
    bottom_thresh : int, lowest abs pixel value below which pixels => 0
    """
    img1 = img_.copy()
    img1 = images_preprocessing(img1,
                                   [channel_],
                                   outlier_percent_=[outlier_percent],
                                   bottom_percent_=[bottom_percent],
                                   bottom_thresh_=[bottom_thresh])
    kernel1_ = kernel
    kernel2_ = kernel/2
    kernel1 = disk(kernel1_)
    kernel2 = disk(kernel2_)
    start=time.time()
    img2 = img1[:,:,0]>0
    img3 = cv2.dilate(np.uint8(img2), kernel1,iterations=1)
    print(time.time() - start)
    img4 = cv2.morphologyEx(np.uint8(img3), cv2.MORPH_CLOSE, kernel1)
    print(time.time()-start)
    img5 = remove_small_objects(img4, size_)
    return img5


def remove_small_objects(mask_, size_):
    """
    Remove objects (binary) smaller than size_ from a binary mask (mask_)
    img_ : 2D numpy array, binary mask to remove objects from
    size : int, size in pixels below which objects are removed
    """
    num_labels, labels_im = cv2.connectedComponents(np.uint8(mask_), connectivity=4)

    for label in range(1, num_labels + 1):
        if sum(sum(labels_im == label)) < size_:
            labels_im[labels_im == label] = 0

    return (labels_im > 0) * 1



def make_aligned_pix_df(img_, cp_masks_, params_):
    '''
    Create dataframe of stacked pixel masks from microscope image, entropy image, and
    cellpose masks. Must also pass the params object in.
    '''
    df_combined = pd.DataFrame()
    cp_mask_df = pd.DataFrame(cp_masks_)
    df_combined['cp_masks'] = cp_mask_df.stack()

    for pix_f in params_.pix_fluor:
        pix_ch = params_.get_channel_from_fluor(pix_f)
        img_df = pd.DataFrame(img_[:,:,pix_ch])
        df_combined[pix_f] = img_df.stack()

    return df_combined


def get_slide_and_tissue_background_means(df_combined_, params_):
    '''
    Take the DF created above and get the tissue and slide background means
    Return df with columns reflecting these.. kind of dumb, but useful for reanalysis
    '''
    for pix_f in params_.pix_fluor:

        tissue = []
        slide = []
        if 1 in df_combined_.tissue.to_list():
            tissue = df_combined_.groupby(['tissue']).get_group(1)[pix_f].tolist()
            df_combined_[pix_f + '_tissue'] = np.mean(tissue)
        else:
            df_combined_[pix_f + '_tissue'] = 'NaN'
        if 0 in df_combined_.tissue.to_list():
            slide = df_combined_.groupby(['tissue']).get_group(0)[pix_f].tolist()
            df_combined_[pix_f + '_slide'] = np.mean(slide)
        else:
            df_combined_[pix_f + '_slide'] = 'NaN'

        return df_combined_


def get_pixel_stats_by_cell(df_combined_, params_):
    '''
    Return df with 90th percentile and mean pixel values for each cell in an image
    and the number of pixels in each cell
    '''
    df_cell = pd.DataFrame()

    for pix_f in params_.pix_fluor:

        df_cell[pix_f + '_ninetieth'] = df_combined_.groupby(['cp_masks']).quantile()[pix_f]
        df_cell[pix_f + '_mean'] = df_combined_.groupby(['cp_masks']).mean()[pix_f]

    df_cell['pixels'] = df_combined_['cp_masks'].value_counts()

    return df_cell


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


def csv_to_mask(df_, sizex_, sizey_):
    """
    Create a binary image mask using a csv created in fiji

    df_ : dataframe of floats, created by "pd.read_csv()"
    sizex_ : int, the horizontal size of the output image
    sizey_ : int, the vertical size of the output image
    """
    csv_ = df_.copy()
    xys = np.flip(csv_.values[:,0:2], axis=1)
    # xys = csv_.values[:,0:2]
    mask_ = np.zeros((sizey_, sizex_))

    for xy in zip(xys):
            #print(xy)
            mask_[xy[0][0]-1, xy[0][1]-1] = 1
    return mask_


###########################################################
### PLOTTING CODE #########################################

def plot_cellpose_results(img_, mask_, channel_, file_, color = 'green'):
    ''' Quick function to plot results of cellpose, not used in final
    img_ : 2d array of image. make sure to pass this in as 2d (select channel)
    mask_ : 2d array of masks created from cellpose
    color : str denoting what color user wants masks to be (default = 'green')'''
    # uses cellpose.plot
    color_ = make_color_list(mask_, color)
    overlay = plot.mask_overlay(img_[:,:,channel_], mask_, np.array(color_))
    fig, ax = plt.subplots(1,2, figsize=(40,20))

    ax[0].imshow(overlay)
    ax[0].set_title(file_, fontsize=40)
    ax[0].axis('off')

    # Raw Plot:
    ax[1].axis('off')
    ax[1].imshow(img_[:,:,channel_])
    ax[1].set_title('Raw', fontsize=40)


def plot_entropy(img_, channel_, mask_, entr_image_):
     # plot triple pane entropy fig
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
    ax[0].imshow(scale_pixels(img_[:,:,channel_]), cmap='gray')
    ax[0].set_xlabel("Noisy image")
    ax[0].axis("off")
    ax[1].imshow(mask_, cmap='viridis')
    ax[1].imshow(scale_pixels(img_[:,:,channel_]), cmap='gray', alpha=0.5)
    ax[1].set_xlabel("Local entropy")
    sns.distplot(entr_image_.ravel(), bins=100, ax=ax[2], color='b')
    sns.despine()
    ax[1].axis("off")
    ax[2].set_title('Entropy hist')
    #ax[2].set_xlim(0,7)
    fig.tight_layout()


def plot_img_w_pix_hist(img_, channel):
    # Show the cyto channel and histogram
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].imshow(img_[:,:,channel])
    sns.distplot(img_[:,:,channel].ravel(), bins=np.linspace(0,np.max(img_[:,:,channel]), 100), ax=ax[1], color='b', kde=False)
    sns.despine()


def make_color_list(masks_, color):
    '''
    create list to be used in plotting mask overlays
    '''
    color_dic = {'red': [1, 0, 0],
                 'green': [0,1,0],
                 'blue': [0,0,1],
                'cyan': [0,1,1]}
    cl = []
    for i in range(np.max(masks_)):
        cl.append(color_dic.get(color))
    return cl


def plot_pix_hist(df_, file_save_path):
    '''
    Need to refactor all of this
        - replace all_dic with df
        - test
    '''
    # plot pixel histogram and save
    # plot green
    fig, ax = plt.subplots(2,2,figsize=(16,5))
    sns.distplot(green_tissue, bins=100, ax=ax[0,0], color='b')
    ax[0,0].text(all_dic['green_tissue_mean'] + all_dic['green_tissue_std'],
            ax[0,0].get_ylim()[1]/2,
            'Red, mean: ' + str(round(all_dic['green_tissue_mean'], 2)),
            fontsize=15)
    ax[0,0].set_title(filename, fontsize=15)
    ax[0,0].set_xlim(0, np.percentile(green_tissue, outlier_percent))

    # plot red
    sns.distplot(red_tissue, bins=100, ax=ax[0,1], color='b')
    ax[0,1].text(all_dic['red_tissue_mean'] + all_dic['red_tissue_std'],
            ax[0,1].get_ylim()[1]/2,
            'Red, mean: ' + str(round(all_dic['red_tissue_mean'], 2)),
            fontsize=15)
    ax[0,1].set_title(filename, fontsize=15)
    ax[0,1].set_xlim(0, np.percentile(red_tissue, outlier_percent))

    sns.despine()

    try:
        sns.distplot([p for p in img[:,:,2].ravel() if p > neun_background_threshold],
                     bins=100,
                     ax=ax[1,1],
                     color='b')
        ax[1,1].text(np.mean(img[:,:,2]) + np.std(img[:,:,2]),
                ax[1,1].get_ylim()[1,1]/2,
                'NeuN, mean: ' + str(round(np.mean(img[:,:,2]), 2)),
                fontsize=15)
        ax.set_xlim(0, np.percentile(img[:,:,2], outlier_percent))
        sns.despine()

    except:
        ax[1,1].set_xlim(0, np.percentile(img[:,:,2], outlier_percent))

    fig.savefig(file_save_path + ' tissue_pix_hist.png')
    plt.close()
