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
        self.N1 = 20
        self.axes = np.array(
            [np.linspace(ximin, ximax, self.N1) for ximin, ximax in self.limits]
        )

        # neat hack to generate uniform cube samples in one array from the meshgrid
        self.cube = np.meshgrid(*self.axes)
        self.cubesamples = np.vstack([c.ravel() for c in self.cube]).T

    def run(self, nGame=3):
        self.nGame = nGame
        self.Games = []
        self.cubelikes = self.likefunc(self.cubesamples)
        self.topnlikes = np.argsort(self.cubelikes)[-self.nGame:]
        self.topnsamples = self.cubesamples[self.topnlikes]
        print('found top',self.nGame, 'likelihoods', self.cubelikes[self.topnlikes])
        print('for top', self.nGame, 'samples\n', self.topnsamples)

        for i, sample in enumerate(self.topnsamples):
            print(f'{"--"*40}')
            print('running game for', sample)
            ga = gamesampler.Game(self.likefunc, sample, sigreg=[0.9] * self.ndim)
            ga.fixedcov = True
            ga.N1 = 1000
            ga.mineffsamp = 100
            ga.maxiter=30
            ga.run()
            print('finished game', i)
            self.Games.append(ga)

        self.gamesamples = np.vstack([[sa.pars for sa in ga.sample_list] for ga in self.Games])
        self.samples = np.vstack([self.gamesamples,self.cubesamples])

        self.gamelikes = np.hstack([[sa.like for sa in ga.sample_list] for ga in self.Games])
        self.gameweights = np.hstack([[sa.we for sa in ga.sample_list] for ga in self.Games])
        self.loglikelihoods = np.hstack([self.gamelikes,self.cubelikes])

        return self.samples, self.loglikelihoods
