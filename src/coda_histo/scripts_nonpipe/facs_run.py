

"""
Files can be found here.  S:\Discovery\Corey

10152020 Was the 24 hour analysis. Rows B and C are HA stained. D and E are a-bungarotoxin stained.

10222020 Is the 48 Hour analysis:  Rows B and C are HA stained. D and E are a-bungarotoxin stained.

Channels are as follows: GFP: Channel 1: External: Channel 2 (AF555) Internal: Channel 3 (AF647)

We do not need to look at GFP expression as it was a different plasmid co-transfected in these experiments.

Sample s are as follows: 2: CODA153  3: CODA823 (Glyr1a) 4: CODA1282 (Glyrb negative control) 5: CODA801, 6: CODA1278 7: CODA1279.

Thanks again and let me know if you have further questions.

"TileScan 2/B/2" would be B: HA, 2: CODA0153
"""

import javabridge
import bioformats as bf
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path
from tqdm import trange
import pyarrow.parquet as pq
from coda_histo import histo_analysis as ha
javabridge.start_vm(class_path=bf.JARS)


mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams.update({'font.size': 22})

thresh1 = 400
thresh2 = 400
date_map = {'2020-10-15': '24hr', '2020-10-22': '48hr'}
stain_map = {'B': 'HA', 'C': 'HA', 'D': 'a-bungarotoxin', 'E': 'a-bungarotoxin'}
# coda_map = {'2': 'CODA0153', '3': 'CODA0823', '4': 'CODA1282', '5': 'CODA0801', '6': 'CODA1278', '7': 'CODA1279'}

path = r'/home/tom/data/facs/'

files = Path(path).glob('*.lif')

df = pd.DataFrame()

# files = ['/home/tom/data/facs/corey_2020-10-22.lif',
#         '/home/tom/data/facs/corey_2020-10-15.lif']

files = ['/media/tom/Rapid/corey_2020-11-12.lif']

# try:
#     table_done = pq.read_table('/home/tom/data/facs/all.pq')
#     df = table_done.to_pandas()
#     images_done = df.image_id.to_list()
# except:
#     images_done = []
images_done = []

for file in files:

    print(file)

    cond = date_map.get(file.split('/')[-1].split('_')[-1][:-4])

    o = bf.OMEXML(bf.get_omexml_metadata(file))

    series_ = o.image_count

    for i in trange(series_):
        # print(i)
        # print(o.image(i).Name)

        name = o.image(i).Name

        image_id = file + ' ' + name + ' ' + str(i)

        if image_id not in images_done:

            img = bf.load_image(file, series=i, rescale=False)

            #             name = o.image(i).Name
            #             stain = stain_map.get(name.split('/')[-2])
            #             coda = coda_map.get(name.split('/')[-1][0])

            # for External (channel 2)
            img1 = ha.clip_outlier_pixels(img, 99.9)[:, :, 1]
            img_mask = (img1 > thresh2) * 1
            img_multiply = np.multiply(img_mask, img1)
            masked_pix = img_multiply[img_multiply > 0]

            try:
                external_90 = np.percentile(masked_pix, 90)
                external_95 = np.percentile(masked_pix, 95)
                external_mean = np.mean(masked_pix)
                external_median = np.median(masked_pix)
                external_99 = np.percentile(masked_pix, 99)
            except:
                external_90 = np.nan
                external_95 = np.nan
                external_mean = np.nan
                external_median = np.nan
                external_99 = np.nan

            fig, ax = plt.subplots(2, 2, figsize=(10, 8))
            ax[0, 0].imshow(ha.scale_pixels(img1))
            ax[0, 0].set_title('Scaled (deceptive)')
            ax[1, 0].imshow(img[:, :, 1])
            ax[1, 0].set_title('Unscaled')
            ax[0, 1].imshow(img_mask)
            ax[0, 1].set_title('Masked')
            #         ax[0,1].text(50,200, 'Thresh: ' + str(thresh), color='r')
            try:
                sns.distplot(masked_pix, ax=ax[1, 1])
            except:
                ax[1, 1].text(0, 0, 'Bad KDE')
            ax[1, 1].set_title('Masked Pixel Intensities')

            if len(masked_pix) > 0:
                ax[1, 1].set_xlim(0, np.max(masked_pix))

            plt.suptitle(file.split('/')[-1] + name + '\n ' + ' external')

            sns.despine()
            fig.tight_layout(pad=1.2)
            plt.savefig('/home/tom/data/facs/figs/' + file.split('/')[-1] + name.replace('/', '') + ' external ' + str(
                i) + '.png')
            plt.close()
            plt.close()

            # for internal (channel 3)
            img1 = ha.clip_outlier_pixels(img, 99.999)[:, :, 2]
            img_mask = (img1 > thresh2) * 1
            img_multiply = np.multiply(img_mask, img1)
            masked_pix = img_multiply[img_multiply > 0]

            try:
                internal_90 = np.percentile(masked_pix, 90)
                internal_95 = np.percentile(masked_pix, 95)
                internal_mean = np.mean(masked_pix)
                internal_median = np.median(masked_pix)
                internal_99 = np.percentile(masked_pix, 99)

            except:
                internal_90 = np.nan
                internal_95 = np.nan
                internal_99 = np.nan
                internal_mean = np.nan
                internal_median = np.nan

            fig, ax = plt.subplots(2, 2, figsize=(10, 8))
            ax[0, 0].imshow(ha.scale_pixels(img1))
            ax[0, 0].set_title('Scaled (deceptive)')
            ax[1, 0].imshow(img[:, :, 2])
            ax[1, 0].set_title('Unscaled')
            ax[0, 1].imshow(img_mask)
            ax[0, 1].set_title('Masked')
            #         ax[0,1].text(50,200, 'Thresh: ' + str(thresh), color='r')
            try:
                sns.distplot(masked_pix, ax=ax[1, 1])
            except:
                ax[1, 1].text(0, 0, 'Bad KDE')
            ax[1, 1].set_title('Masked Pixel Intensities')

            if len(masked_pix) > 0:
                ax[1, 1].set_xlim(0, np.max(masked_pix))

            plt.suptitle(file.split('/')[-1] + name + '\n ' + ' internal')

            sns.despine()
            fig.tight_layout(pad=1.2)
            plt.savefig('/home/tom/data/facs/figs/' + file.split('/')[-1] + name.replace('/', '') + ' internal ' + str(
                i) + '.png')
            plt.close()
            plt.close()

            dic = {'file': file,
                   'image_name': name,
                   'external_99': external_99,
                   'external_90': external_90,
                   'external_95': external_95,
                   'external_mean': external_mean,
                   'external_median': external_median,
                   'internal_99': internal_99,
                   'internal_90': internal_90,
                   'internal_95': internal_95,
                   'internal_mean': internal_mean,
                   'internal_median': internal_median,
                   'image_n': i,
                   'image_id': image_id}

            df = df.append(dic, ignore_index=True)

df.to_parquet('/home/tom/data/facs/20201119_all.pq')
df.to_csv('/home/tom/data/facs/20201119_all.csv')