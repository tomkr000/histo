### Disclaimer
This was an ETL pipeline for histology analysis that I built for a startup a few years ago interfacing with Zeiss, Airtable and AWS. 

==========
histo
==========
#### Installation (within an EC2 with a GPU):

1. `nano . ~/.bashrc`

2. add to the bottom `export JAVA_HOME=${JAVA_HOME}:/usr/lib/jvm/default-java`
3. `git clone <repo>
4. `cd histo`
5. `conda env create -f environment.yml`
6. `conda activate histo`
7. cd to histo and type `python setup.py develop`
8. to mount the histology bucket as `/histo` on your EC2:
sudo s3fs histology-0b171b86-25d4-40c1-847b-ab4f82275e01 -o use_cache=/tmp -o allow_other -o uid=1000 -o mp_umask=0277 -o multireq_max=5 /histo -o nonempty
9. Make sure you are running CUDA10 or CUDA10.1 or CUDA10.2 ($nvidia-smi) and replace the installed mxnet version with the appropriate cuda version:

pip uninstall mxnet-mkl
pip uninstall mxnet
pip install mxnet-cu101


#### Basic flow

High level, this pipeline takes raw image files in Leica (.lif) or Zeiss (.czi) format, transforms them into numpy
arrays, runs the cellpose algorithm on one of the channels in the array ( [:,:,channel] ) to designate cell masks,
then transposes those masks into the other channels to get statistics about fluorescence from each cell.

Cellpose docs: https://cellpose.readthedocs.io/en/latest/

I highly recommend reading the entire thing including API ref as there are a lot of tricks that can be helpful.

There are 4 main steps to get to the point where you have pixel statistics for each cell:

Upload -> Conversion -> Cellpose -> Aggregation

Upload:

This is straightforward using the AWS cli which is currently installed on the zeiss comp with Tom's credentials.
Windows uses the same syntax as linux so <aws s3 cp . s3://xxxxxxx> will do. Easiest to leave all czi files in
one directory and label the directory with the experiment name (ex: "EXP2020-0011_drg_hotplate")

Conversion:

Scripts designated with the "CONV" prefix. There are 3 of them right now:
CONV_czi_to_npy.py : uses the image as is, with the user specifying the resolution level (0 is full resolution, 1 is 0.5 resolution, 2 is 0.25 etc...)
CONV_autocrop_czi_to_npy.py : this is currently tuned for spinal cord and will have to be refactored for drg or hippocampus.
It takes an image with multiple tissue sections in it, finds eah section and crops them into individual images, saving to npy format.
CONV_crop_czi_w_fiji_output.py : this uses csv files created using the polygon tool in fiji (instructions in the script
itself) to crop out specific sections of an image and save them. Very important for SNr and hippocampus sections and also creates a
training dataset for possibly training a CNN to do this automatically in the future.

Each of these programs ends up outputting npy files of each image with the image name into the same directory as the
original files and is then ready for cellpose.

Cellpose:

Scripts designated with "EC2" prefix. There are 2, one for ventralhorn images, and one for drgs. They use slightly different
methods to extract cells. These programs interface with 2 different tables in Airtable: "Histology_Profiles" and
"histo_params". "histo_params" is where you designate the parameters used by the program, and "Histology_Profiles"
is where the microscopist enters which channel belongs to which fluorophore.

If you do multiple runs of the cellpose algorithm with different parameters, the histo_params name is the unique identifier
in the filename of the output pickle file ('image_name_params_name.pkl' would be the output file name)that says the file
was created with a specific set of params. This allows you to run cellpose multiple times with different
params and keep track of each run.

The CellposeParams (defined in "histo_analysis.py") object automatically selects the correct channel numbers given the fluorophores chosen to run cellpose
and extract stats from.

Current parameter options:
exp_name : quick designator for the experiment number and descriptor.
profile : microscope profile from the "Histology_Profiles" table
tissue : designates where the tissue came from
cellpose_ch : which channels to run cellpose on. see cellpose readthedocs for formatting, but basically [0,0] means run on the cyto_fluor channel
cyto_fluor : which ihc stain to run cellpose on (so far only NeuN)
pix_fluor : which ihc stains to get pixel/cell stats from
model_type : "cyto" or "nuclei", whether to use cytoplasm or nuclei to find cells
entropy_fluor : deprecated, but which ihc stain to use for entropic tissue finding
diameter : expected cellular diameter
flow_thresh : threshold to set that determines if something could be cell (see cellpose readthedocs, + makes it harder)
cellprob_thresh : threshold to set that determines if something that could be a cell is indeed a cell (see cellpose readthe docs, + makes it harder)
morph_kernel : size of initial kernel disk to perform tissue finding with
entropy_kernel : deprecated in favor of morphing
entropy_thresh : deprecated in favor of morphing
entropy_disk : deprecated in favor of morphing
(Note: used to use entropy to find tissue, now just threshold and use morphing, which is much faster)
outlier_percent : in every image there are a few pixels that are orders of magnitude brighter than others so a first
pre-processing step is to get rid of them. this sets the percentile above which pixels are clipped.
bottom_thresh : essentially background subtraction - turns every pixel below the value into 0
bottom_percent : second, percentile-based background subtraction, percentile below which pixels are assigned 0
scale_factor : factor to scale the image down by upon loading
nuclei_fluor : designates which ihc stain is the nucleus. haven't used.
tilex : how large the tiling window (for increased speed) in x direction should be
tiley : how large the tiling window (for increased speed) in y direction should be

The output is a pickle file saved in "pkl_files" within the /histo bucket. The pickle file contains:

data_dic = {'df_pixels': df_pixels,     # df of unraveled pixels, essentially just img in 2D
                'df_cells': df_cells,   # cell by cell stats
                'img': img,             # original image
                'tissue_mask': tissue,  # if there is a tissue mask from tissue finding, it's here
                'cp_mask': cp_masks,    # numbered masks designating each cell's footprint
                'params': params}       # the parameters used for this run

So you can repeat pixel stats for each cell with just "img" and "cp_mask" and quickly check quality with quick plotting like:

fig, ax = plt.subplots(1,1,figsize=(10,10))
ax.imshow(img[:,:,params.cyto_ch)
ax.imshow(cp_masks>1, alpha=0.5)

Each processed file gets an entry in the "Histology" table in airtable with a link to the pickle file and its original s3 link (which is somewhat useless).
In addition the script figures out which rat the image is from and populates metadata for the rat from the "Animal Log" table.
Designating a rat as a control is, for right now, best done manually by changing the value in the "naive_control" column of the "Histology"
table to "y". I've tried to get Transx team to be consistent with group names so I could just pull it out of that, but it hasn't happened.

POSSIBLE REFACTOR: Right now there isn't a great way to designate which drg is which in the Histology table, but there is a column for it (Location).
I should have a script for this by the time I leave, which should be run at this point in the flow.

There is a ton of extraneous info in the Histology table and most could be deleted. All you really need is the pkl file path, params_name, Rat number,
rat metadata, location, and naive_control y/n. The rest was aspirational and is best left to post-hoc analysis.


Aggregation:

This is done using AGG_df_cells_to_parquet.py. This takes every pickle file created with a set of parameters (params_name) and
aggregates all the df_cells (cell by cell stats in tidy format, each cell being a row). It also sets the threshold to determine
if a cell is positive for a protein using all animals
designated controls ("y" in naive_control). So a cell that is positive for HA_tag has a "True" in the "HA_tag" column. It does
this based on the ninetieth percentile of the fluorescence for the pixels in the cell's footprint in the fluorophore channel. Mean
was misleading as there is usually a hole in the middle of the cell with no fluorophore. So the column with the cell's fluorescence
in the HA_tag channel is "HA_tag_ninetieth". The program automatically makes this column for each fluorophore.

The output is a parquet file to the "df_cells" folder in the /histo bucket with every cell from the run.
From there you can download it and import into a notebook to run analyses





NOTES:

I highly recommend getting a large, fast external SSD drive to download the images to and run on your local comp.
You will probably want to use Fiji to have an easy way to just look at images as a quick qc and make guesstimates about
initial pixel thresholds to try.
