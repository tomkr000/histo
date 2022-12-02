from metaflow import FlowSpec, step, resources, conda, batch

class BigSum(FlowSpec):

    #@conda(libraries={"numpy": "1.18.1"})
    @batch(image='alexcoda1/kerasenv:latest', memory=50000, cpu=4)
    #@resources(memory=6000, cpu=4)
    @step
    def start(self):
        #import numpy
        import time
        #big_matrix = numpy.random.ranf((800, 800))
        t = time.time()
        self.sum = 2
        self.took = time.time() - t
        self.next(self.end)

    @step
    def end(self):
        print("The sum is %f." % self.sum)
        print("Computing it took %dms." % (self.took * 1000))

if __name__ == '__main__':
    BigSum()
