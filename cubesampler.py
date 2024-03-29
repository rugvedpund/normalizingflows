import numpy as np

class Cube:
    """
    Simplest possible uniform grid sampler
    """

    def __init__(self,likefunc,limits):
        print('starting Cube for',limits)
        self.likefunc = likefunc
        self.limits = limits
        self.ndim = len(limits)
        self.npoints = [71,70,69]

    def run(self):
        self.axes = np.array([np.linspace(ximin,ximax,npoints) for (ximin,ximax),npoints in zip(self.limits,self.npoints)])
        self.cube = np.meshgrid(*self.axes)
        self.samples = np.vstack([c.ravel() for c in self.cube]).T
        self.nsamples = np.prod(self.npoints)
        self.nchunks = self.nsamples//200000 + 1
        print('running',self.nsamples,'samples in',self.nchunks,'chunks')
        self.cubelikes = np.zeros(self.nsamples)
        for chunk in range(self.nchunks): 
            print('chunk',chunk+1,'of',self.nchunks)
            start = chunk*200000
            end = min((chunk+1)*200000,self.nsamples)
            self.cubelikes[start:end] = self.likefunc(self.samples[start:end])
        print('done')
