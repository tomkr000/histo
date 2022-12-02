from metaflow.metaflow_config import get_authenticated_boto3_client
from metaflow import FlowSpec, step, IncludeFile, Parameter, conda_base, S3, batch
#
import keras
from keras import layers, models, callbacks
import numpy as np

class TestDockerCellpose(FlowSpec):

    @step
    def start(self):

        self.next(self.prep_data)

    @batch(image='alexcoda1/kerasenv:latest')
    @step
    def prep_data(self):
        import bioformats as bf
        from cellpose import models

        with S3() as s3:
            bucket = 'histology-0b171b86-25d4-40c1-847b-ab4f82275e01'
            key = 'czi_files/R0000_DRG_Slide6_Profile1.czi'


        o = bf.OMEXML(bf.get_omexml_metadata(filepath_))




        with S3() as s3:
            bucket = 'plate-reader-screens-5b139e0f-9545-4285-a204-e999788adb86'
            key = 'protvec_PR_pec50.csv'
            obj = s3.get('s3://'+bucket+'/'+key)
            self.orig_Z = pd.read_csv(io.BytesIO(obj.blob), index_col=0)

            bucket = 'coda-receptor-modeling'
            key = '20200324_23_all_protvec.csv'
            obj = s3.get(f's3://{bucket}/{key}')
            self.ns = pd.read_csv(io.BytesIO(obj.blob), index_col=0)

        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    TestDockerFlow()
