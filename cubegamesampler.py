import numpy as np
import gamesampler


class CubeGame:
    """
    First make a uniform grid of points in parameter space. Then, for the top n likelihoods,
    drop a GameSampler and run it.

    """

    def __init__(self, likefunc, limits):
        print('starting CubeGame for', limits)
        self.likefunc = likefunc
        self.limits = limits
        self.ndim = len(limits)
        self.N1 = 50
        self.axes = np.array(
            [np.linspace(ximin, ximax, self.N1) for ximin, ximax in self.limits]
        )

        # neat hack to generate uniform cube samples in one array from the meshgrid
        self.cube = np.meshgrid(*self.axes)
        self.samples = np.vstack([c.ravel() for c in self.cube]).T

    def run(self, nGame=3):
        self.nGame = nGame
        self.Games = []
        self.cubelikes = self.likefunc(self.samples)
        self.topnlikes = np.argsort(self.cubelikes)[-self.nGame:]
        self.topnsamples = self.samples[self.topnlikes]
        print('found top',self.nGame, 'likelihoods', self.topnlikes)
        print('for top', self.nGame, 'samples\n', self.topnsamples)

        for i, sample in enumerate(self.topnsamples):
            print('running game for', sample)
            ga = gamesampler.Game(self.likefunc, sample, sigreg=[0.7] * self.ndim)
            ga.N1 = 1000
            ga.mineffsamp = 1000
            ga.run()
            print('finished game', i)
            self.Games.append(ga)
