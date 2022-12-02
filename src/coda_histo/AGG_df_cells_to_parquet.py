'''
Takes all pickle files generated with a specified parameter name (listed in "histo_params" airtable "Name" column)
and aggregates the df_cells (cell-by-cell stats) into tidy-form dataframe and saves as parquet in the "df_cells" key within
the histology bucket on S3.

The next step (not done) would be to aggregate these into one large database.

Program needs to be run in an EC2 instance or mounted directories will not work.

mounting should be of histo bucket as /histo

to mount the histology bucket onto the EC2 use:
sudo s3fs histology-0b171b86-25d4-40c1-847b-ab4f82275e01 -o use_cache=/tmp -o allow_other -o uid=1000 -o mp_umask=0277 -o multireq_max=5 /histo -o nonempty
'''


import numpy as np
import pandas as pd
import time, os, sys
import pickle
from scipy import stats
from datetime import date
from tqdm import tqdm
import boto3
from io import StringIO
import itertools
import pyarrow
import argparse

from coda_discovery.utils import codaAirtable
from coda_histo import histo_analysis as ha

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser(description=
                                     'Put cell counts from an exp into single large dataframe and save to parquet file')

    parser.add_argument('params', type=str,
                        help='''name of experimental run to process, pulls everything else from airtable''')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArguments()

    params = args.params

    save_path = '/histo/df_cells/'

    params_ = ha.CellposeParams(params)

    airtable=codaAirtable('Histology')
    df_airtable = airtable.to_df()
    df_airtable= df_airtable.applymap(lambda x: x[0] if isinstance(x, list) else x)

    df_airtable = df_airtable[df_airtable['analysis_params'] == params]

    df = pd.DataFrame()
    files = df_airtable.s3_pkl_path.to_list()

    for i, file in enumerate(tqdm(files)):

        data = pickle.load( open( file, 'rb'))

        df_ = data['df_cells']
        if type(df_) != str:
            for column in df_airtable.columns:
                #print(column, ' >>>>> ', df_airtable.iloc[i][column])
                fill = df_airtable.iloc[i][column]
                try:
                    df_[column] = fill
                    if column == 'naive_control':
                        print(fill)
                except:
                    pass #print('meh')

            df = pd.concat([df, df_])

    df.reset_index(inplace=True)

    df.to_parquet(save_path + params_.exp_name + '_' + params + '_no_ctrls')
    
    try:
        df_control = df.groupby('naive_control').get_group('y')
        for fluor in params_.pix_fluor:
            df[fluor] = df[fluor + '_ninetieth'] > np.percentile(df_control[fluor + '_ninetieth'], 99.9)

        df.to_parquet(save_path + params_.exp_name + '_' + params)
    except:
        print('NO CONTROLS DESIGNATED')
