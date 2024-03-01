import numpy as np
import NormalizingFlow as nf
from corner.core import quantile

class RandomWalker:
    def __init__(self, walkerparams, nsteps, stepsizefactor=0.1):
        self.nsteps = nsteps
        self.walkerparams = walkerparams
        self.steps = self.doSteps(stepsizefactor)

    def doSteps(self, stepsizefactor):
        self.steps = np.zeros((self.nsteps, 3))
        self.steps[0, :] = [
            self.walkerparams["astart"],
            self.walkerparams["wstart"],
            self.walkerparams["nstart"],
        ]
        for i in range(1, self.nsteps):
            stepsize = np.array(
                [
                    self.walkerparams["amax"] - self.walkerparams["amin"],
                    self.walkerparams["wmax"] - self.walkerparams["wmin"],
                    self.walkerparams["nmax"] - self.walkerparams["nmin"],
                ]
            )
            self.steps[i, :] = (
                self.steps[i - 1, :] + np.random.normal(0, stepsizefactor, 3) * stepsize
            )
            for ivs, vs in enumerate(["A", "W", "N"]):
                if self.steps[i, ivs] < 0.01:
                    self.steps[i, ivs] = 0.01
        return self.steps


class Walkers:
    def __init__(self, args, nwalkers, nsteps):
        self.args = args
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.cosmicdawn = True if args.fgFITS == "gsm16.fits" else False

    def setInitialKWargs(self):
        # start with 1D amplitude plot
        print("setting initial kwargs...")
        self.kwargs = {
            "amin": 0.001,
            "amax": 100.0,
            "wmin": 1.0,
            "wmax": 50.0,
            "nmin": 1.0,
            "nmax": 50.0,
            "logspace": True,
        }
        if self.cosmicdawn:
            self.kwargs["nmin"] = 50.0
            self.kwargs["nmax"] = 100.0
        print(self.kwargs)

    def runInitial1DLikelihoods(self, flow, npoints1, cmb=False):
        truths = [1.0, 20.0, 67.5] if self.cosmicdawn else [1.0, 14.0, 16.4]

        # samples is a (3*npoints)x3 array of samples where each row is a sample of (amp,width,numin), and the first npoints rows are for A, the next npoints rows are for W, and the last npoints rows are for N
        samples = np.tile(truths, (3 * npoints1, 1))
        print("generating initial samples")
        for ivs, vs in enumerate(["A", "W", "N"]):
            param = np.logspace(
                np.log10(self.kwargs[vs.lower() + "min"]),
                np.log10(self.kwargs[vs.lower() + "max"]),
                npoints1,
            )
            samples[ivs * npoints1 : (ivs + 1) * npoints1, ivs] = param

        t21vs = np.zeros((len(flow.freqs), 3 * npoints1))
        pt21vs = np.zeros_like(t21vs)
        print("getting t21")
        for ii, s in enumerate(samples):
            a, w, n = s
            t21vs[:, ii] = nf.T_DA(
                flow.freqs, a, w, n, cmb=False, cosmicdawn=self.cosmicdawn
            )
        print("projecting t21")
        pt21vs = flow.proj_t21(t21vs)
        print("getting likelihood")

        loglikelihood = flow.get_likelihood(
            pt21vs, flow.args.freqFluctuationLevel, flow.args.DA_factor, debugfF=False
        )
        print("initial loglikelihoods ready")
        return samples, loglikelihood

    def extractWalkerStart(self, s, ll, npoints1):
        limits = [0.001, 0.5, 0.9999]
        results = {
            "astart": 0,
            "amin": 0,
            "amax": 0,
            "wstart": 0,
            "wmin": 0,
            "wmax": 0,
            "nstart": 0,
            "nmin": 0,
            "nmax": 0,
        }
        print("getting initial walker start")
        for ivs, vs in enumerate(["A", "W", "N"]):
            samples = s[ivs * npoints1 : (ivs + 1) * npoints1, ivs]
            likelihood = nf.exp(ll[ivs * npoints1 : (ivs + 1) * npoints1])
            quantiles = quantile(samples, limits, weights=likelihood)
            maxlike = samples[np.argmax(likelihood)]
            print(vs, quantiles, maxlike)
            results[vs.lower() + "start"] = maxlike
            results[vs.lower() + "min"] = max(
                maxlike - (quantiles[2] - quantiles[0]), 0.01
            )
            results[vs.lower() + "max"] = quantiles[1] + (quantiles[2] - quantiles[1])
            self.initWalkerParams = results
        return self.initWalkerParams

    def walkWalkers(self, walkerparams):
        print("starting random walk for", self.nwalkers, "walkers")
        self.initWalkerSteps = (
            [[1.0, 20.0, 67.5]] if self.cosmicdawn else [[1.0, 14.0, 16.4]]
        )
        for i in range(self.nwalkers):
            walker = RandomWalker(walkerparams, self.nsteps)
            self.initWalkerSteps = np.vstack((self.initWalkerSteps, walker.steps))
        return self.initWalkerSteps

    def getWalkerLogLikelihood(self, args, flow, walker_s, cmb=False):
        print("getting likelihood")
        t21vs = np.zeros((len(flow.freqs), len(walker_s)))
        pt21vs = np.zeros_like(t21vs)
        for ii, s in enumerate(walker_s):
            a, w, n = s
            t21vs[:, ii] = nf.T_DA(
                flow.freqs, a, w, n, cmb=cmb, cosmicdawn=self.cosmicdawn
            )
        pt21vs = flow.proj_t21(t21vs)
        loglikelihood = flow.get_likelihood(
            pt21vs, args.freqFluctuationLevel, args.DA_factor, debugfF=False
        )
        return walker_s, loglikelihood

    def extractBetterWalkerStart(self, walker_s, walker_ll):
        bestwalker = walker_s[np.argmax(walker_ll)]
        print("best walker start:", bestwalker)
        results = {
            "astart": bestwalker[0],
            "amin": self.initWalkerParams["amin"],
            "amax": self.initWalkerParams["amax"],
            "wstart": bestwalker[1],
            "wmin": self.initWalkerParams["wmin"],
            "wmax": self.initWalkerParams["wmax"],
            "nstart": bestwalker[2],
            "nmin": self.initWalkerParams["nmin"],
            "nmax": self.initWalkerParams["nmax"],
        }
        self.betterWalkerParams = results
        return self.betterWalkerParams

    def rewalkWalkers(self, betterWalkerParams, nsteps2):
        print("rewalking walkers with nsteps2", nsteps2)
        self.newWalkerSteps = np.zeros((nsteps2 * self.nwalkers, 3))
        for i in range(self.nwalkers):
            walker = RandomWalker(betterWalkerParams, nsteps2, stepsizefactor=0.01)
            self.newWalkerSteps[
                i * nsteps2 : (i + 1) * nsteps2, :
            ] = walker.steps.copy()
        return self.newWalkerSteps

    def getAllWalkersAndLLikelihoods(self, s, s2, s3, ll, ll2, ll3):
        self.allWalkerSteps = np.vstack((s, s2, s3))
        self.allLogLikelihoods = np.hstack((ll, ll2, ll3))
        return self.allWalkerSteps, self.allLogLikelihoods
