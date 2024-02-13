# 30 Aug 2022, Rugved Pund

import torch
from sinf import GIS
import matplotlib.pyplot as plt
import numpy as np
import lusee
import matplotlib
import healpy as hp
import fitsio
import os
import corner

root = os.environ[
    "NF_WORKDIR"
]  # specify path to save/load models and likelihood results


def plot_corner(args):
    """plot corner plots for all parameters"""
    samples, likelihood = get_samplesAndLikelihood(args, "all")
    if args.fgFITS == "ulsa.fits":
        amp = 40.0
        truths = [amp, 14.0, 16.4]
    if args.fgFITS == "gsm16.fits":
        amp = 130.0
        truths = [amp, 14.0, 16.4]
    samples[:, 0] *= amp
    fig = corner.corner(
        samples,
        weights=exp(likelihood),
        labels=[r"$A$", r"$\nu_{\rm rms}$", r"$\nu_{\rm min}$"],
        plot_datapoints=False,
        truths=truths,
        truth_color="gray",
        levels=[1 - np.exp(-0.5), 1 - np.exp(-2)],
        bins=50,
        show_titles=True,
        hist_kwargs={"density": True},
    )
    return fig


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


def get_fname(args, old=False):
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
    if not old:
        fname += f"_freqs{fqs}"
    return fname


def get_lname(args, plot, old=False):
    """for saving likelihood results"""
    lname = get_fname(args, old)
    if args.appendLik:
        lname += f"like{args.appendLik}"

    lname += f"_noisyT21{args.noisyT21}_vs{plot}_DAfactor{args.DA_factor}_freqFluctuationLevel{args.freqFluctuationLevel}"
    if old:
        paths = lname.split("/")
        paths.insert(6, "corner")
        lname = "/".join(paths)
    return lname


def get_samplesAndLikelihood(args, plot, verbose=False, old=False):
    lname = get_lname(args, plot, old)
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
    # out['amp']*=40.0
    # out['amp+']*=40.0
    # out['amp-']*=40.0
    return out


def get_log(fname):
    with open(fname) as f:
        lines = f.readlines()
    n = len(lines)
    trainlogp, vallogp, iteration, best = np.zeros((4, n))
    for il, l in enumerate(lines):
        lsplit = l.split()
        trainlogp[il] = lsplit[1]
        vallogp[il] = lsplit[2]
        iteration[il] = lsplit[-3]
        best[il] = lsplit[-1]
    return iteration, trainlogp, vallogp, best


def get_nLayerBest(iteration, trainlogp, vallogp, best, delta_logp=10):
    b = int(best[-1])
    n = int(iteration[np.where(trainlogp - vallogp < delta_logp)][-1])
    return min(n - 1, b - 1)


class NormalizingFlow:
    def __init__(self, nocuda, loadPath=""):
        self.device = torch.device("cpu") if nocuda else torch.device("cuda")
        try:
            self.model = torch.load(loadPath).to(self.device)
            self.nlayer = len(self.model.layer)
            print("model loaded from ", loadPath)
        except FileNotFoundError:
            print("no file found, need to train")
        self.precompute_data_after = dict()

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
        print("now training model with nsamples", self.nsamples, " of ndim", self.ndim)
        self.train_data = self._toTensor(train_data)
        self.validate_data = self._toTensor(validate_data)
        if retrain:
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
        print('loading new NF.py!')
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

        print(f"loading foreground map {args.fgFITS}")
        self.fg = fitsio.read(f"{root}{args.fgFITS}")

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
            print(f"generating radiometer noise with SNRpp={self.SNRpp:.0e}")
            self.noise = self.generate_radiometer_noise(
                self.fg.copy(), self.SNRpp, self.subsampleSigma
            )
        else:
            print(f"generating noise with noise_K={self.noise_K}")
            self.noise = self.generate_noise(
                self.fg.copy(), self.noise_K, self.subsampleSigma
            )

        # do combineSigma
        sigmas = [self.sigma] + self.combineSigma
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
                print("doing diffCombineSigma for combineSigma")
                self.fgsmooth_diff[isig * self.nfreq : (isig + 1) * self.nfreq, :] = (
                    self.fg_cS[isig, :, :].copy() - self.fg_cS[isig - 1, :, :].copy()
                )
            self.fgsmooth = self.fgsmooth_diff.copy()
            print(f"{self.fgsmooth.shape=}")

        # do avgAdjacentFreqBins
        if self.avgAdjacentFreqBins:
            print("combining adjacent freq bins")
            self.nfreq = self.nfreq // 2
            self.fgsmooth_cAFB = np.zeros((self.nsigmas, self.nfreq, self.npix))
            self.fgsmooth_cAFB = (self.fg_cS[:, ::2, :] + self.fg_cS[:, 1::2, :]) / 2
            self.fgsmooth = self.fgsmooth_cAFB.reshape(self.nsigmas * self.nfreq, -1)
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
        self.eve, self.eva, self.vt = np.linalg.svd(
            self.fgss - self.fgss.mean(axis=1)[:, None]
        )
        # print('new')
        print(f"{self.fgss.shape=}")
        proj_fg = self.eve.T @ (self.fgss - self.fgss.mean(axis=1)[:, None])
        self.pfg = proj_fg.copy()
        self.rms = np.sqrt(proj_fg.var(axis=1))
        out = proj_fg / self.rms[:, None]
        self.data = np.random.permutation(out.T)  # transpose and permute for torch
        self.fgmeansdata = (self.eve.T @ self.fgss.mean(axis=1)) / self.rms
        print(f"{self.data.shape=} {self.fgmeansdata.shape=} {self.eve.shape=}")

        # remove nPCA modes
        self.nPCA = (
            [0, self.nfreq * self.nsigmas]
            if len(args.nPCA) == 0
            else [int(p) for p in args.nPCA.split()]
        )
        print(f"using modes between {args.nPCA}")
        self.nPCAarr = np.hstack(
            [
                np.arange(self.nPCA[0]),
                np.arange(self.nPCA[1], self.nfreq * self.nsigmas),
            ]
        )
        print(self.nPCAarr)
        self.data = np.delete(
            self.data, self.nPCAarr, axis=1
        )  # delete last nPCA columns
        self.fgmeansdata = np.delete(self.fgmeansdata, self.nPCAarr)
        print(f"{self.data.shape=} {self.fgmeansdata.shape=} {self.eve.shape=}")

        ndata, _ = self.data.shape
        ntrain = int(0.8 * ndata)
        self.train_data = self.data[:ntrain, :].copy()
        self.validate_data = self.data[ntrain:, :].copy()
        print(f"done! {self.train_data.shape=},{self.validate_data.shape=} ready")

    def set_t21(self, t21, include_noise):
        if include_noise:
            t21 += self.noise.mean(axis=1)
        if self.avgAdjacentFreqBins:
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
        print(f"{self.t21data.shape=} ready")

    def proj_t21(self, t21_vs, include_noise):
        assert t21_vs.shape == (self.nfreq, t21_vs.shape[1])
        t21_noisy = (
            t21_vs + self.noise.mean(axis=1)[:, None]
            if include_noise
            else np.zeros_like(t21_vs)
        )
        if self.avgAdjacentFreqBins:
            t21_noisy = (t21_noisy[::2, :] + t21_noisy[1::2, :]) / 2
        if self.diffCombineSigma:
            t21cS = np.zeros((self.nfreq * self.nsigmas, t21_noisy.shape[1]))
            t21cS[: self.nfreq, :] = t21_noisy.copy()
        else:
            t21cS = np.tile(t21_noisy, (self.nsigmas, 1))
        print(t21cS.shape)
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
            print("sigmas must be either 0.5, 1.0, 2.0, 4.0, or 8.0")
            raise NotImplementedError
        return sigma2nside_map[sigma]

    def smooth(self, fg, sigma, chromatic):
        print(f"smoothing with {sigma} deg, and chromatic {chromatic}")
        fgsmooth = np.zeros_like(fg)
        for fi, f in enumerate(self.freqs):
            sigma_f = sigma * (10.0 / (f)) if chromatic else sigma
            fgsmooth[fi, :] = hp.sphtfunc.smoothing(
                fg[fi, :], sigma=np.deg2rad(sigma_f)
            )
        return fgsmooth

    def galaxy_cut(self, fg, b_min):
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
        flowt21data_fF = (
            self.t21data
            if debugfF
            else (1 + freqFluctuationLevel * np.cos(6 * np.pi / 50 * freqs))
            * self.t21data
        )
        l = (
            self.fgmeansdata[:, None] + DA_factor * flowt21data_fF[:, None] - t21_vsdata
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

        pt21vs = self.proj_t21(t21vs, include_noise=self.args.noisyT21)
        loglikelihood = self.get_likelihood(
            pt21vs, self.args.freqFluctuationLevel, self.args.DA_factor, debugfF=False
        )
        return samples, loglikelihood

    def getGainFluctuationMap(self):
        # does not work for chromatic maps
        assert self.args.chromatic == False
        gF = np.random.normal(0.0, self.gainFluctuationLevel, self.npix)
        self.gFtiled = np.tile(gF, (self.nfreq, 1))
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
            print("multiplying gainF")
            gFresmooth = gFmap.copy()
        else:
            print(f"resmoothing by {sigma} and multiplying gainF")
            gFresmooth = self.smooth(gFmap.copy(), sigma, chromatic=False)
        self.fgsmoothgF = fgsmooth.copy() * gFresmooth.copy()
        self.t21smoothgF = template[:, None] * gFresmooth.copy()

        return fgsmooth.copy() + self.fgsmoothgF + self.t21smoothgF


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
            t21vs[:, ii] = T_DA(
                flow.freqs, a, w, n, cmb=False, cosmicdawn=self.cosmicdawn
            )
        print("projecting t21")
        pt21vs = flow.proj_t21(t21vs, include_noise=self.args.noisyT21)
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
            likelihood = exp(ll[ivs * npoints1 : (ivs + 1) * npoints1])
            quantiles = corner.core.quantile(samples, limits, weights=likelihood)
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
            t21vs[:, ii] = T_DA(
                flow.freqs, a, w, n, cmb=cmb, cosmicdawn=self.cosmicdawn
            )
        pt21vs = flow.proj_t21(t21vs, include_noise=args.noisyT21)
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

    def print(self):
        for arg in vars(self):
            print(arg, getattr(self, arg))
