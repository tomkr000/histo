# -*- coding: utf-8 -*-

import pytest
from coda_histo.skeleton import fib

__author__ = "Tom Roseberry"
__copyright__ = "Tom Roseberry"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)

def lif_import():
    import javabridge
    javabridge.start_vm(class_path=bf.JARS)
    o = bf.OMEXML(bf.get_omexml_metadata(r'/mnt/c/code/coda_histo/data/)

def test_cellpose():
    import cellpose
    urls = ['http://www.cellpose.org/static/images/img02.png']
    files = []
    for url in urls:
        parts = urlparse(url)
        filename = os.path.basename(parts.path)
        if not os.path.exists(filename):
            sys.stderr.write('Downloading: "{}" to {}\n'.format(url, filename))
            utils.download_url_to_file(url, filename)
        files.append(filename)
    imgs = [skimage.io.imread(f) for f in files]
    nimg = len(imgs)
    from cellpose import models
    model = models.Cellpose(gpu=False, model_type='cyto')
    channels = [[2,3], [0,0], [0,0]]
    masks, flows, styles, diams = model.eval(imgs, diameter=None, channels=channels)


def test_s3_airtable_function():
    import boto3
    from botocore.exceptions import NoCredentialsError
    from coda_discovery.utils import codaAirtable
