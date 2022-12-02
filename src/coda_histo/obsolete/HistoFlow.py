from metaflow import FlowSpec, step, IncludeFile, Parameter
from coda_histo import histo_analysis as ha
import numpy as np
from coda_discovery.utils import codaAirtable
from pathlib import Path
import bioformats as bf
import seaborn as sns

from cellpose import utils
from cellpose import models
from cellpose import plot

import javabridge



class HistoFlow(FlowSpec):
    """
    Histology Pipeline Flow
    """

    sns.set_style('darkgrid')

    directory = Parameter('directory',
                             help='folder to get new images from',
                             default='/histo/czi_files')

    process_new_files_only = Parameter(
        'process_new_files_only',
        help='process new files only, if False, will rerun all data',
        default=True)

    n_files_to_process = Parameter(
        'n_files',
        help='number of files within directory to process',
        default='all')

    @step
    def start(self):
        self.dir_path = Path(self.directory)
        self.next(self.get_files_to_process)

    @step
    def get_files_to_process(self):
        '''
        dirpath_ : path object to directory
        n_files : int, number of files to run on
        '''
        #files_ = [self.dir_path.glob(e) for e in ['*.czi', '*.lif']]

        files_ = self.dir_path.glob('*.czi')
        files = [file.as_posix() for file in files_]

        histo_airtable = codaAirtable('Histology')
        histo_df = histo_airtable.to_df()
        print(histo_df.columns)
        existing_files = set(histo_df['filename'])
        new_files = set(files)

        if self.process_new_files_only:
            self.files_to_process = list(new_files - existing_files)
        else:
            self.files_to_process = list(new_files)
            overwrite = True

        self.next(self.get_metadata_for_each, foreach='files_to_process')


    @step
    def get_metadata_for_each(self):
        '''
        compile a dataframe of metadata for each file
        '''
        

        if self.input[-4:] == '.czi':
            javabridge.start_vm(class_path=bf.JARS)
            self.df_thisfile = ha.create_czi_file_df(self.input)
            javabridge.kill_vm()

        elif self.input[-4:] == '.lif':
            pass # need to create 'create_lif_file_df'

        self.next(self.join_metadata)

    @step
    def join_metadata(self, inputs):
        '''
        make a list of metadata df's from inputs and concat
        '''
        dfs = []
        for input in inputs:
            dfs.append(input.df_thisfile)
        self.df_to_process = pd.concat(dfs)
        print('JOINED METADATA DATAFRAME')
        self.next(self.finish_df_prep)

    @step
    def finish_df_prep(self):

        params = ha.CellposeParams() # initiate a params object to fill the df from

        self.df_to_process[params.cyto_fluor] = 0

        for pix_f in params.pix_fluor:

            self.df_to_process[pix_f] = 0
            self.df_to_process['max_brightness_' + pix_f] = 0

        self.images = list(zip(files_to_process, df_to_process['series_num'].tolist()))


        print('DONE WITH DATAFRAME')

        self.next(self.run_cellpose, foreach='images')

    @step
    def run_cellpose(self):

        params = ha.CellposeParams()

        params.designate_within_slide_params(self.input[0])

        javabridge.start_vm(class_path=bf.JARS)

        img = bf.load_image(self.input[0],
                            self.input[1],
                            rescale=False)  # load the image

        javabridge.kill_vm()

        img1 = ha.image_preprocessing(img, params)

        entr_mask, entr_img = ha.tissue_mask_from_entropy(img1,
                                                       params.entropy_ch,
                                                        params.entropy_thresh,
                                                       params.entropy_kernel,
                                                        params.entropy_disk)

        img_cell = ha.create_tif_stack(img1, [params.cyto_ch])

        model = models.Cellpose(gpu=False, model_type=params.model_type)

        imgs = [img_cell]
        cp_masks, flows, styles, diams = model.eval(imgs,
                                                 diameter=params.diameter,
                                                 channels=params.cellpose_ch,
                                                 do_3D=False,
                                                 flow_threshold=params.flow_threshold,
                                                 cellprob_threshold=params.cellprob_threshold)

        if np.max(cp_masks) != 0:

            df_pixels = ha.make_aligned_pix_df(img1, entr_mask, cp_masks[0], self.params)

            df_pixels = ha.get_slide_and_tissue_background_means(df_pixels, self.params)

            df_cells = ha.get_pixel_stats_by_cell(df_pixels, self.params)

            df_cells = df_cells.drop([0])

        else:
            df_pixels = 'None'
            df_cells = 'None'


        self.df_out = pd.DataFrame()
        self.df_out['filename'] = input[0]
        self.df_out['series_num'] = input[2]
        self.df_out[params.cyto_fluor] = len(df_cells)

        data_dic = {'df_pixels': df_pixels,
                    'df_cells': df_cells,
                    'img': img1,
                    'cp_mask': cp_masks,
                    'params': params}

        with open('~/histo/pkl_files/' + input[0] + ' ' + input[1], 'wb') as f:
            pickle.dump(data_dic, f)

        print('DONE WITH '
                + str(self.df_to_process['filename'][file_ind])
                + ' of '
                + str(len(self.df_to_process)))


        self.next(self.join_df_for_airtable)


    @step
    def join_df_for_airtable(self, inputs):
        dfs = []
        for input in inputs:
            dfs.append(input.df_out)
        dfs_all = pd.concat(dfs)

        self.df_to_process = pd.merge(self.df_to_process,
                                    dfs_all,
                                    on=['filename', 'series_num'],
                                    how='outer')

        self.next(self.upload_to_airtable)


    @step
    def upload_to_airtable(self):

        self.df_to_process = self.df_to_process.drop(['SizeX'], axis=1)

        histo_airtable.upload_df_to_airtable(self.df_to_process,
                                            primary_key='imagename',
                                            overwrite=overwrite)

        self.next(self.end)

    @step
    def end(self):
       # javabridge.kill_vm()
       pass

if __name__ == '__main__':
    HistoFlow()
