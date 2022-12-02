"""
Quick script to upload a directory to S3. Usually cli is faster, but this is a little lst cumbersome sometimes.
"""

import boto3
import sys
import threading
from pathlib import Path
import os


class ProgressPercentage(object):

    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify, assume this is hooked up to a single filename
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)" % (
                    self._filename, self._seen_so_far, self._size,
                    percentage))
            sys.stdout.flush()


s3 = boto3.client('s3', region_name='us-west-2')
bucket_name = 'histology-0b171b86-25d4-40c1-847b-ab4f82275e01'

path = Path(r'/mnt/r/Research/Translation/Sunny/EXP2020-0001 PDGF')

filepaths = path.glob('*.czi')

filelist = [filepath.as_posix() for filepath in filepaths]

filelist = filelist[5:]

for file in filelist:

    print('Uploading ' + file)

    s3_key = 'czi_files/' + file.split('/')[-1]

    s3.upload_file(file, bucket_name, s3_key, Callback=ProgressPercentage(file))

    print(file + ' Successful')
