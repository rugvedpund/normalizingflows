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
        self.N1 = 60

    def run(self):
        self.axes = np.array([np.linspace(ximin,ximax,self.N1) for ximin,ximax in self.limits])
        self.cube = np.meshgrid(*self.axes)
        self.samples = np.vstack([c.ravel() for c in self.cube]).T
        self.cubelikes = self.likefunc(self.samples)
        print('done')
