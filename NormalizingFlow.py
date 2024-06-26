# 30 Aug 2022, Rugved Pund

import torch
import matplotlib.pyplot as plt
import numpy as np
import lusee
import matplotlib
import healpy as hp
import fitsio
import os
import corner

verbose = True

# specify path to save/load models and likelihood results
try:
    root = os.environ["NF_WORKDIR"]
except:
    root = os.path.abspath("./")
    print("Now using the current dir", root)

##---------------------------------------------------------------------------##


class NormalizingFlow:
    def __init__(self, nocuda, loadPath=""):
        self.device = torch.device("cpu") if nocuda else torch.device("cuda")
        try:
            self.model = torch.load(loadPath).to(self.device)
            self.nlayer = len(self.model.layer)
            if verbose:
                print("model loaded from ", loadPath)
                print("number of layers in model: ", len(self.model.layer))
        except FileNotFoundError:
            if verbose:
                print("no file found, need to train")
        print("--" * 40, "\n")

    def train(
        self,
        train_data,
        validate_data,
        nocuda,
        savePath,
        retrain,
        alpha=None,
        delta_logp=np.inf,
        verbose=False,
    ):
        """
        data must be of shape (nsamples, ndim)
        """
        self.nsamples, self.ndim = train_data.shape
        if verbose:
            print(
                "now training model with nsamples", self.nsamples, " of ndim", self.ndim
            )
        self.train_data = self._toTensor(train_data)
        self.validate_data = self._toTensor(validate_data)
        if retrain:
            from sinf import GIS

            print("retraining...")
            self.model = None
        self.model = GIS.GIS(
            self.train_data.clone(),
            self.validate_data.clone(),
            nocuda=nocuda,
            alpha=alpha,
            delta_logp=delta_logp,
            verbose=verbose,
        )
        self.nlayer = len(self.model.layer)
        print("Total number of iterations: ", len(self.model.layer))
        torch.save(self.model, savePath)
        print("model saved at", savePath)

    def _toTensor(self, nparray):
        return torch.from_numpy(nparray).float().to(self.device)

    def _likelihood(self, data, end=None):
        data = self._toTensor(data)
        return self.model.evaluate_density(data, end=end)


class FlowAnalyzerV2(NormalizingFlow):
    def __init__(self, loadPath, nocuda=False):
        if verbose:
            print("loading Normalizing Flow module...")
        super().__init__(loadPath=loadPath, nocuda=nocuda)

    def set_fg(self, args):
        """
        sigma: smoothing scale in deg
        chromatic: bool, whether to use chromatic smoothing
        galcut: galactic cut in deg
        subsampleSigma: subsampling scale in deg
        noise_K: noise level in K
        noiseSeed: seed for noise
        combineSigma: list of smoothing scales in deg
        gFLevel: gain fluctuation level
        gFdebug: 0,1,2,3 for no gF, gF, template, gF+template
        nPCA: list of nPCA modes to remove
        diffCombineSigma: bool, whether to do fg_cS[sigma]-fg_cS[sigma-1]
        avgAdjacentFreqBins: bool, whether to combine adjacent freq bins
        fgFITS: foreground map of shape (nfreq,npix)
        freqs: start and end freqs separated by space
        """

        self.args = args
        self.sigma = args.sigma
        self.chromatic = args.chromatic
        self.galcut = args.galcut
        self.SNRpp = args.SNRpp
        self.noise_K = args.noise
        self.noiseSeed = args.noiseSeed
        self.subsampleSigma = args.subsampleSigma
        self.combineSigma = [float(c) for c in args.combineSigma.split()]
        self.gainFluctuationLevel = args.gainFluctuationLevel
        self.gFdebug = args.gFdebug
        self.diffCombineSigma = args.diffCombineSigma
        self.avgAdjacentFreqBins = args.avgAdjacentFreqBins
        self.fmin, self.fmax = [int(f) for f in args.freqs.split()]
        self.freqs = np.arange(self.fmin, self.fmax)
        self.cosmicdawn = True if args.fgFITS == "gsm16.fits" else False

        print(f"setting noise seed {args.noiseSeed} and torch seed {args.torchSeed}")
        np.random.seed(args.noiseSeed)
        torch.manual_seed(args.torchSeed)
        torch.cuda.manual_seed_all(args.torchSeed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

        if verbose:
            print(f"loading foreground map {args.fgFITS}")
        fgpath = os.path.join(root, args.fgFITS)
        self.fg = fitsio.read(fgpath)

        # fg
        self.nfreq, self.npix = self.fg.shape
        if self.nfreq != len(self.freqs):
            print("fg shape and given freqs do not match")
            print("cropping fg to match freqs")
            self.fg = self.fg[-len(self.freqs) :, :]
            self.nfreq = len(self.freqs)
            print("new fg shape is:", self.fg.shape)
        # self.fg=np.random.rand(self.nfreq,self.npix)
        np.random.seed(self.noiseSeed)
        # generate noise
        if self.SNRpp is not None:
            if verbose:
                print(f"generating radiometer noise with SNRpp={self.SNRpp:.0e}")
            self.noise = self.generate_radiometer_noise(
                self.fg.copy(), self.SNRpp, self.subsampleSigma
            )
        else:
            if verbose:
                print(f"generating noise with noise_K={self.noise_K}")
            self.noise = self.generate_noise(
                self.fg.copy(), self.noise_K, self.subsampleSigma
            )

        # do combineSigma
        sigmas = [self.sigma] + self.combineSigma
        if verbose:
            print("doing sigmas", sigmas)
        self.nsigmas = len(sigmas)
        self.fg_cS = np.zeros((self.nsigmas, *self.fg.shape))
        self.fgnoisy = self.fg.copy() + self.noise.copy()

        # f=30
        # hp.mollview(self.fgnoisy[f,:],title='noisy')
        for isig, sig in enumerate(sigmas):
            self.fg_cS[isig, :, :] = self.smooth(
                self.fgnoisy.copy(), sig, self.chromatic
            )
            # hp.mollview(self.fg_cS[isig,f,:],title=f'smooth {sig}')

        self.fgmeans_smooth = self.fg_cS.reshape(self.nsigmas * self.nfreq, -1).mean(
            axis=1
        )

        if self.gainFluctuationLevel != 0.0:
            self.gainFmap = self.getGainFluctuationMap()
            # hp.mollview(self.gainFmap[f,:],title=f'gainF map {self.gainFluctuationLevel}')
            for isig, sig in enumerate(sigmas):
                self.fg_cS[isig, :, :] = self.multiplyGainFluctuations(
                    self.fg_cS[isig, :, :].copy(), sig
                )
                # hp.mollview(self.fg_cS[isig,f,:],title=f'smooth*gainF {self.gainFluctuationLevel} {sig}')

        self.fgsmooth = self.fg_cS.reshape(self.nsigmas * self.nfreq, -1)
        if verbose:
            print(f"{self.fgsmooth.shape=}")

        self.fgmeans_smoothgF = self.fg_cS.reshape(self.nsigmas * self.nfreq, -1).mean(
            axis=1
        )

        # do diffCombineSigma
        if self.diffCombineSigma:
            self.fgsmooth_diff = np.zeros_like(self.fgsmooth)
            # difference between fg_cS and fg_cS[0] except for fg_cS[0]
            self.fgsmooth_diff[: self.nfreq, :] = self.fg_cS[0, :, :].copy()
            for isig, sig in enumerate(sigmas):
                if isig == 0:
                    continue
                if verbose:
                    print("doing diffCombineSigma for combineSigma")
                self.fgsmooth_diff[isig * self.nfreq : (isig + 1) * self.nfreq, :] = (
                    self.fg_cS[isig, :, :].copy() - self.fg_cS[isig - 1, :, :].copy()
                )
            self.fgsmooth = self.fgsmooth_diff.copy()
            if verbose:
                print(f"{self.fgsmooth.shape=}")

        # do avgAdjacentFreqBins
        if self.avgAdjacentFreqBins:
            if verbose:
                print("combining adjacent freq bins")
            self.nfreq = self.nfreq // 2
            self.fgsmooth_cAFB = np.zeros((self.nsigmas, self.nfreq, self.npix))
            self.fgsmooth_cAFB = (self.fg_cS[:, ::2, :] + self.fg_cS[:, 1::2, :]) / 2
            self.fgsmooth = self.fgsmooth_cAFB.reshape(self.nsigmas * self.nfreq, -1)
            if verbose:
                print(f"{self.fgsmooth.shape=}")

        # #do PCA with full map with galcut (no subsampling)
        self.fgcut = self.galaxy_cut(self.fgsmooth.copy(), self.galcut)
        # self.cov=np.cov(self.fgcut)
        # eva,eve=np.linalg.eig(self.cov)
        # s=np.argsort(np.abs(eva))[::-1]
        # self.eva,self.eve=np.abs(eva[s]),eve[:,s]

        self.fgmeans_cut = self.fgcut.mean(axis=1)

        # subsample and project
        self.fgss = self.subsample(
            self.fgsmooth.copy(), self.subsampleSigma, self.galcut
        )

        self.fgmeans_ss = self.fgss.mean(axis=1)

        # do SVD
        print('doing SVD...')
        self.eve, self.eva, self.vt = np.linalg.svd(
            self.fgss - self.fgss.mean(axis=1)[:, None]
        )
        # print('new')
        if verbose:
            print(f"{self.fgss.shape=}")
        proj_fg = self.eve.T @ (self.fgss - self.fgss.mean(axis=1)[:, None])
        self.pfg = proj_fg.copy()
        self.rms = np.sqrt(proj_fg.var(axis=1))
        out = proj_fg / self.rms[:, None]
        self.data = np.random.permutation(out.T)  # transpose and permute for torch
        self.fgmeansdata = (self.eve.T @ self.fgss.mean(axis=1)) / self.rms
        if verbose:
            print(f"{self.data.shape=} {self.fgmeansdata.shape=} {self.eve.shape=}")

        # remove nPCA modes
        self.nPCA = (
            [0, self.nfreq * self.nsigmas]
            if len(args.nPCA) == 0
            else [int(p) for p in args.nPCA.split()]
        )
        if verbose:
            pcadebug = "all" if args.nPCA == "" else args.nPCA
            print(f"using PCA modes {pcadebug}")
        self.nPCAarr = np.hstack(
            [
                np.arange(self.nPCA[0]),
                np.arange(self.nPCA[1], self.nfreq * self.nsigmas),
            ]
        )
        self.data = np.delete(
            self.data, self.nPCAarr, axis=1
        )  # delete last nPCA columns
        self.fgmeansdata = np.delete(self.fgmeansdata, self.nPCAarr)
        if verbose:
            print(f"{self.data.shape=} {self.fgmeansdata.shape=} {self.eve.shape=}")

        ndata, _ = self.data.shape
        ntrain = int(0.8 * ndata)
        self.train_data = self.data[:ntrain, :].copy()
        self.validate_data = self.data[ntrain:, :].copy()
        if verbose:
            print(f"done! {self.train_data.shape=},{self.validate_data.shape=} ready")

    def set_t21(self, t21):
        t21 = t21.copy()
        if self.args.noisyT21:
            t21 += self.noise.mean(axis=1)
        if self.avgAdjacentFreqBins:
            if verbose:
                print("combining adjacent freq bins for t21")
            t21 = (t21[::2] + t21[1::2]) / 2
        if self.diffCombineSigma:
            self.t21 = np.zeros(self.nfreq * self.nsigmas)
            self.t21[: self.nfreq] = t21.copy()
        else:
            self.t21 = np.tile(t21, self.nsigmas)
        self.pt21 = self.eve.T @ self.t21
        self.t21data = self.eve.T @ self.t21 / self.rms
        self.t21data = np.delete(self.t21data, self.nPCAarr)
        if verbose:
            print(f"{self.t21data.shape=} ready")
        if verbose:
            print("ready to train/calculate likelihoods!")

    def proj_t21(self, t21_vs):
        include_noise = self.args.noisyT21
        assert t21_vs.shape == (self.nfreq, t21_vs.shape[1])
        if include_noise:
            t21_noisy = t21_vs + self.noise.mean(axis=1)[:, None]
        else:
            t21_noisy = t21_vs.copy()

        if self.avgAdjacentFreqBins:
            t21_noisy = (t21_noisy[::2, :] + t21_noisy[1::2, :]) / 2
        if self.diffCombineSigma:
            t21cS = np.zeros((self.nfreq * self.nsigmas, t21_noisy.shape[1]))
            t21cS[: self.nfreq, :] = t21_noisy.copy()
        else:
            t21cS = np.tile(t21_noisy, (self.nsigmas, 1))
        if verbose:
            print("Calculating likelihood for npoints = ", t21cS.shape[1])
        proj_t21 = (self.eve.T @ t21cS) / self.rms[:, None]
        proj_t21 = np.delete(proj_t21, self.nPCAarr, axis=0)
        return proj_t21

    def generate_noise(self, fg, noise_K, subsampleSigma):
        nfreq, ndata = fg.shape
        nside = self.sigma2nside(subsampleSigma)
        npix = hp.nside2npix(nside)
        noise_sigma = noise_K * np.sqrt(npix)
        noise = np.random.normal(0, noise_sigma, (nfreq, ndata))
        return noise

    def generate_radiometer_noise(self, fg, snrperpixel, subsampleSigma):
        nfreq, ndata = fg.shape
        nside = self.sigma2nside(subsampleSigma)
        npix = hp.nside2npix(nside)
        noise_sigma = fg.mean(axis=1) * np.sqrt(npix) / snrperpixel
        noise = np.vstack([np.random.normal(0, s, ndata) for s in noise_sigma])
        return noise

    def sigma2nside(self, sigma):
        # done by hand
        sigma2nside_map = {0.5: 64, 1.0: 32, 2.0: 16, 4.0: 8, 8.0: 4}
        try:
            nside = sigma2nside_map[sigma]
        except KeyError:
            if verbose:
                print("sigmas must be either 0.5, 1.0, 2.0, 4.0, or 8.0")
            raise NotImplementedError
        return sigma2nside_map[sigma]

    def smooth(self, fg, sigma, chromatic):
        if verbose:
            print(f"smoothing with {sigma} deg, and chromatic {chromatic}")
            # print(f"smoothing with {sigma} deg, and chromatic {chromatic}, with new pivot at 70MHz")
        fgsmooth = np.zeros_like(fg)
        for fi, f in enumerate(self.freqs):
            sigma_f = sigma * (10.0 / (f)) if chromatic else sigma
            # sigma_f = sigma * (70.0 / (f)) if chromatic else sigma
            fgsmooth[fi, :] = hp.sphtfunc.smoothing(
                fg[fi, :], sigma=np.deg2rad(sigma_f)
            )
        return fgsmooth

    def galaxy_cut(self, fg, b_min):
        if verbose:
            print(f"doing galcut for b_min={b_min} deg")
        _, npix = fg.shape
        nside = np.sqrt(npix / 12).astype(int)
        col_min = np.pi / 2 - np.deg2rad(b_min)
        col_max = np.pi / 2 + np.deg2rad(b_min)
        mask_pix = hp.query_strip(nside, col_min, col_max, inclusive=False)
        cutfg = np.delete(fg, mask_pix, axis=1)
        return cutfg

    def subsample(self, fg, subsampleSigma, galcut):
        nside = self.sigma2nside(subsampleSigma)
        ipix = np.arange(hp.nside2npix(nside))
        theta, phi = hp.pix2ang(nside, ipix)
        ipix_128 = hp.ang2pix(128, theta, phi)
        gal = hp.query_strip(
            nside,
            np.pi / 2 - np.deg2rad(galcut),
            np.pi / 2 + np.deg2rad(galcut),
            inclusive=False,
        )
        galpix_128 = hp.ang2pix(*(128, *hp.pix2ang(nside, gal)))
        maskpix = np.setdiff1d(ipix_128, galpix_128)
        fg_subsample = fg[:, maskpix].copy()
        return fg_subsample

    def get_likelihood(
        self, t21_vsdata, freqFluctuationLevel, DA_factor=1.0, debugfF=False, end=None
    ):
        assert t21_vsdata.shape == (self.nfreq * self.nsigmas, t21_vsdata.shape[1])
        freqs = np.tile(self.freqs, self.nsigmas)
        flowt21data = self.t21data.copy()
        if debugfF:
            raise NotImplementedError
        if freqFluctuationLevel!=0.0:
            print(f'apply freqFluctuation {freqFluctuationLevel*100}%')
            fFluctuation = freqFluctuationLevel * np.cos(6 * np.pi / 50 * freqs)
            flowt21data += self.t21data.copy() * fFluctuation

        l = (
            self.fgmeansdata[:, None] + DA_factor * flowt21data[:, None] - t21_vsdata
        ).T
        return self._likelihood(l, end).cpu().numpy()

    def get_likelihoodFromSamples(self, samples, cmb=False):
        assert samples.shape == (samples.shape[0], 3)
        t21vs = np.zeros((self.nfreq, samples.shape[0]))
        pt21vs = np.zeros_like(t21vs)
        for ii, s in enumerate(samples):
            a, w, n = s
            t21vs[:, ii] = T_DA(
                self.freqs, a, w, n, cmb=cmb, cosmicdawn=self.cosmicdawn
            )

        pt21vs = self.proj_t21(t21vs)
        loglikelihood = self.get_likelihood(
            pt21vs, self.args.freqFluctuationLevel, self.args.DA_factor, debugfF=False
        )
        return samples, loglikelihood

    def get_likelihoodFromSamplesGAME(
        self, samples, cmb=False, priorlow=[0.01, 1, 1], priorhigh=[10, 40, 40]
    ):
        arr = np.array(samples)  # need this for game.py
        assert arr.shape == (arr.shape[0], 3)

        _, loglikelihood = self.get_likelihoodFromSamples(arr, cmb=cmb)

        # apply prior
        print("applying priors: low =", priorlow, ", high =", priorhigh)
        abovelowidx = (arr > priorlow).all(axis=1)
        belowhighidx = (arr < priorhigh).all(axis=1)
        inprioridx = np.logical_and(abovelowidx, belowhighidx)
        outprioridx = np.logical_not(inprioridx)
        loglikelihood[outprioridx] = -1e9  # assign a super low loglikelihood

        # apply loglikelihood blowupfactor
        print("blowing up likelihoods by ", self.args.llblowupfactor)
        loglikelihood /= self.args.llblowupfactor

        return loglikelihood

    def get_likelihoodFromSampleEMCEE(self, sample):
        arr = np.array(sample)[None, :]
        return self.get_likelihoodFromSamplesGAME(arr)

    def getGainFluctuationMap(self):
        # does not work for chromatic maps
        assert self.args.chromatic == False
        gF = np.random.normal(0.0, self.gainFluctuationLevel, self.npix)
        self.gFtiled = np.tile(gF, (self.nfreq, 1))
        if verbose:
            print(
                f"getting base gainF map for gFLevel {self.gainFluctuationLevel} and sigma {self.sigma}"
            )
        self.gFsmooth = self.smooth(self.gFtiled, self.sigma, self.chromatic)
        std = np.std(self.gFsmooth[0, :])
        return np.nan_to_num(self.gainFluctuationLevel / std * self.gFsmooth)

    def multiplyGainFluctuations(self, fgsmooth, sigma):
        gFmap = self.gainFmap
        freqs = np.arange(1, 51)
        template = lusee.MonoSkyModels.T_DarkAges_Scaled(
            freqs, nu_rms=14, nu_min=16.4, A=0.04
        )
        if sigma == self.sigma:
            if verbose:
                print("multiplying gainF")
            gFresmooth = gFmap.copy()
        else:
            if verbose:
                print(f"resmoothing by {sigma} and multiplying gainF")
            gFresmooth = self.smooth(gFmap.copy(), sigma, chromatic=False)
        self.fgsmoothgF = fgsmooth.copy() * gFresmooth.copy()
        self.t21smoothgF = template[:, None] * gFresmooth.copy()

        return fgsmooth.copy() + self.fgsmoothgF + self.t21smoothgF


def exp(l, numpy=True):
    if numpy:
        return np.exp((l - max(l)))
    else:
        return [np.exp((ll - max(l)) / 2) for ll in l]


def T_CMB(freqs):
    return 2.725 * np.ones_like(freqs)


def T_DA(freqs, amp, width, nu_min, cmb=False, cosmicdawn=False):
    if cmb:
        return amp * T_CMB(freqs)
    if cosmicdawn:
        return amp * lusee.MonoSkyModels.T_CosmicDawn_Scaled(
            freqs, nu_rms=width, nu_min=nu_min
        )
    return amp * lusee.MonoSkyModels.T_DarkAges_Scaled(
        freqs, nu_rms=width, nu_min=nu_min
    )


def get_amp_width_numin(
    npoints,
    amin=0,
    amax=3.0,
    wmin=10.0,
    wmax=20.0,
    nmin=10.0,
    nmax=20.0,
    logspace=False,
):
    if logspace:
        amp = np.logspace(np.log10(amin), np.log10(amax), num=npoints)
        width = np.logspace(np.log10(wmin), np.log10(wmax), num=npoints)
        nu_min = np.logspace(np.log10(nmin), np.log10(nmax), num=npoints)
    else:
        amp = np.linspace(amin, amax, num=npoints)
        width = np.linspace(wmin, wmax, num=npoints)
        nu_min = np.linspace(nmin, nmax, num=npoints)

    # amp=np.logspace(np.log10(amin),np.log10(amax),num=npoints) if logspace else np.linspace(amin,amax,num=npoints) #default is 1.0
    # width=np.linspace(wmin,wmax,num=npoints) #default is 14 MHz
    # nu_min=np.linspace(nmin,nmax, num=npoints) #default is 16.4 MHz
    return amp, width, nu_min


def uniform_grid(npoints, **kwargs):
    amp, width, nu_min = get_amp_width_numin(npoints, **kwargs)
    g = np.meshgrid(amp, width, nu_min)
    samples = np.vstack(list(map(np.ravel, g))).T
    return samples


def get_t21vs(freqs, npoints, cmb=False, cosmicdawn=False, **kwargs):
    assert freqs.shape == (50,)
    samples = uniform_grid(npoints, **kwargs)
    assert samples.shape == (npoints ** 3, 3)
    t21_vs = np.zeros((len(freqs), npoints ** 3))
    for i, (a, w, n) in enumerate(samples):
        t21_vs[:, i] = T_DA(freqs, a, w, n, cmb=cmb, cosmicdawn=cosmicdawn)
    return samples, t21_vs


def uniform_grid2d(npoints, vs, **kwargs):  # vs= WvA NvA WvN
    amp, width, nu_min = get_amp_width_numin(npoints, **kwargs)
    if vs == "WvA":
        g = np.meshgrid(width, amp)
    if vs == "NvA":
        g = np.meshgrid(nu_min, amp)
    if vs == "WvN":
        g = np.meshgrid(width, nu_min)
    samples = np.vstack(list(map(np.ravel, g))).T
    return samples


def get_t21vs2d(npoints, vs, **kwargs):  # vs= WvA NvA WvN
    samples = uniform_grid2d(npoints, vs, **kwargs)
    t21_vs = np.zeros((50, npoints ** 2))
    for i, (x, y) in enumerate(samples):
        if vs == "WvA":
            t21_vs[:, i] = T_DA(amp=y, width=x, nu_min=16.4)
        if vs == "NvA":
            t21_vs[:, i] = T_DA(amp=y, width=14.0, nu_min=x)
        if vs == "WvN":
            t21_vs[:, i] = T_DA(amp=1.0, width=x, nu_min=y)
    return samples, t21_vs


def get_t21vs1d(freqs, npoints, vs, cmb=False, cosmicdawn=False, **kwargs):
    amp, width, nu_min = get_amp_width_numin(npoints, **kwargs)
    if vs == "A":
        samples = amp
        if cosmicdawn:
            tDA = lambda xx: T_DA(
                freqs, amp=xx, width=20.0, nu_min=67.5, cmb=cmb, cosmicdawn=cosmicdawn
            )
        else:
            tDA = lambda xx: T_DA(freqs, amp=xx, width=14.0, nu_min=16.4, cmb=cmb)
    if vs == "W":
        samples = width
        if cosmicdawn:
            tDA = lambda xx: T_DA(
                freqs, amp=1.0, width=xx, nu_min=67.5, cmb=cmb, cosmicdawn=cosmicdawn
            )
        else:
            tDA = lambda xx: T_DA(freqs, amp=1.0, width=xx, nu_min=16.4, cmb=cmb)
    if vs == "N":
        samples = nu_min
        if cosmicdawn:
            tDA = lambda xx: T_DA(
                freqs, amp=1.0, width=20.0, nu_min=xx, cmb=cmb, cosmicdawn=cosmicdawn
            )
        else:
            tDA = lambda xx: T_DA(freqs, amp=1.0, width=14.0, nu_min=xx, cmb=cmb)
    t21_vs = np.zeros((len(freqs), npoints))
    for i, xx in enumerate(samples):
        t21_vs[:, i] = tDA(xx)
    return samples, t21_vs


def cbrt(n):
    c = int(np.cbrt(n))
    if c ** 3 == n:
        return c
    if (c + 1) ** 3 == n:
        return c + 1
    if (c - 1) ** 3 == n:
        return c - 1
    else:
        raise ValueError("stupid cbrt")


def get_fname(args):
    if args.old:
        print("using old model")
    """for saving model after training"""
    cS = ",".join(args.combineSigma.split())
    pca = ",".join(args.nPCA.split())
    fqs = ",".join(args.freqs.split())
    fgname = args.fgFITS.split(".")[0]

    fname = f"{root}GIS_{fgname}_nside128_sigma{args.sigma}_subsample{args.subsample_factor}_galcut{args.galcut}_noPCA{args.noPCA}_chromaticBeam{args.chromatic}_combineSigma{cS}"
    if args.SNRpp is not None:
        fname += f"_SNRpp{args.SNRpp:.0e}"
    else:
        fname += f"_noise{args.noise}"
    fname += f"_seed{args.noiseSeed}_subsampleSigma{args.subsampleSigma}"
    if args.gainFluctuationLevel is not None:
        fname += f"_gainFluctuation{args.gainFluctuationLevel}_gFdebug{args.gFdebug}"
    if args.append:
        fname += args.append
    if args.nPCA:
        fname += f"_nPCA{pca}"
    if not args.old:
        fname += f"_freqs{fqs}"
    return fname


def get_lname(args, plot):
    """for saving likelihood results"""
    if args.old:
        print("using old likelihood")
    lname = get_fname(args)
    if args.appendLik:
        lname += args.appendLik if args.old else f"like{args.appendLik}"

    lname += f"_noisyT21{args.noisyT21}_vs{plot}_DAfactor{args.DA_factor}_freqFluctuationLevel{args.freqFluctuationLevel}"
    if args.old:
        paths = lname.split("/")
        paths.insert(6, "corner")
        lname = "/".join(paths)
    return lname


def get_samplesAndLikelihood(args, plot, verbose=False):
    lname = get_lname(args, plot)
    if verbose:
        print(f"loading corner likelihood results from {lname}")
    f = np.loadtxt(lname, unpack=True)
    likelihood = f[-1]
    samples = f[:-1].T
    return samples, likelihood


def get_constraints(s, ll):
    # assume amp,width,numin
    params = ["amp", "width", "numin"]
    out = {
        "amp": 0.0,
        "amp+": 0.0,
        "amp-": 0.0,
        "width": 0.0,
        "width+": 0.0,
        "width-": 0.0,
        "numin": 0.0,
        "numin+": 0.0,
        "numin-": 0.0,
    }
    for i in range(3):
        l2, l, m, h, h2 = corner.core.quantile(
            s[:, i].copy(), [0.05, 0.16, 0.5, 0.84, 0.95], weights=exp(ll.copy())
        )
        out[params[i]] = m
        out[params[i] + "+"] = h - m
        out[params[i] + "-"] = m - l
        out[params[i] + "+-"] = h2 - l2
        out[params[i] + "++"] = h2
    return out


class Args:
    def __init__(
        self,
        sigma=2.0,
        subsample_factor=None,
        galcut=20.0,
        noPCA=False,
        chromatic=False,
        combineSigma="",
        SNRpp=None,
        noise=0.0,
        noiseSeed=0,
        torchSeed=0,
        subsampleSigma=2.0,
        noisyT21=True,
        gainFluctuationLevel=0.0,
        gFdebug=0,
        append="_SVD",
        DA_factor=1.0,
        plot="all",  # 1dAmplitude 1dWidth 1dNuMin WvA NvA WvN
        freqFluctuationLevel=0.0,
        nPCA="",
        diffCombineSigma=True,
        avgAdjacentFreqBins=False,
        retrain=False,
        appendLik="",
        fgFITS="ulsa.fits",
        freqs="1 51",
        old=False,
        llblowupfactor=1.0,
    ):

        self.sigma = sigma
        self.subsample_factor = subsample_factor
        self.galcut = galcut
        self.noPCA = noPCA
        self.chromatic = chromatic
        self.combineSigma = combineSigma
        self.SNRpp = SNRpp
        self.noise = noise
        self.noiseSeed = noiseSeed
        self.torchSeed = torchSeed
        self.subsampleSigma = subsampleSigma
        self.noisyT21 = noisyT21
        self.gainFluctuationLevel = gainFluctuationLevel
        self.gFdebug = gFdebug
        self.append = append
        self.DA_factor = DA_factor
        self.plot = plot
        self.freqFluctuationLevel = freqFluctuationLevel
        self.nPCA = nPCA
        self.diffCombineSigma = diffCombineSigma
        self.avgAdjacentFreqBins = avgAdjacentFreqBins
        self.retrain = retrain
        self.appendLik = appendLik
        self.fgFITS = fgFITS
        self.freqs = freqs
        self.old = old
        self.llblowupfactor = llblowupfactor

    def prettyprint(self):
        print("Using the following args:")
        for arg in vars(self):
            val = str(getattr(self, arg))
            print(f"{arg:20s} {val:20s}")
        print("--" * 40, "\n")
