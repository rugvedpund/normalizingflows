import torch
import NormalizingFlow as nf
import numpy as np
import parser
import lusee


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda")
torch.set_default_tensor_type("torch.cuda.FloatTensor")


class FlowLikelihood(nf.FlowAnalyzerV2):
    def __init__(self, args):
        print("loading new FlowLikelihood module")
        loadPath = nf.get_fname(args)
        super().__init__(loadPath=loadPath, nocuda=False)
        self.set_fg(args)
        self.tnoise = torch.from_numpy(self.noise.copy()).to(device)
        self.teve = torch.from_numpy(self.eve.copy()).to(device).float()
        self.trms = torch.from_numpy(self.rms.copy()).to(device)
        self.tfgmeansdata = torch.from_numpy(self.fgmeansdata.copy()).to(device)

        if args.fgFITS == "ulsa.fits":
            print("using DA model")
            t21 = lusee.MonoSkyModels.T_DarkAges_Scaled(
                self.freqs, nu_rms=14, nu_min=16.4, A=0.04
            )
            cosmicdawn = False
        elif args.fgFITS == "gsm16.fits":
            print("using CD model")
            t21 = lusee.MonoSkyModels.T_CosmicDawn_Scaled(
                self.freqs, nu_rms=20, nu_min=67.5, A=0.130
            )
            cosmicdawn = True
        self.set_t21(t21)
        if args.retrain:
            self.train(
                self.train_data,
                self.validate_data,
                nocuda=False,
                savePath=fname,
                retrain=True,
            )
        self.tt21data = torch.from_numpy(self.t21data.copy()).to(device)

    def proj_t21(self, t21block):
        tt21block = t21block.to(device)
        assert tt21block.shape == (self.nfreq, tt21block.shape[1])
        if self.args.noisyT21:
            tt21block = tt21block + self.tnoise.mean(axis=1)[:, None]

        if self.args.avgAdjacentFreqBins:
            raise NotImplementedError
        if not self.args.diffCombineSigma:
            raise NotImplementedError
        if len(self.nPCAarr) != 0:
            raise NotImplementedError

        tt21cS = torch.zeros(
            (self.nfreq * self.nsigmas, tt21block.shape[1]), device=device, dtype=torch.float
        )
        tt21cS[: self.nfreq, :] = tt21block

        print("Calculating likelihood for npoints = ", tt21cS.shape[1])
        tpt21 = (self.teve.T @ tt21cS) / self.trms[:, None]

        return tpt21

    def likelihood(self, samples, priorlow=None, priorhigh=None, cmb=False):

        arr = torch.tensor(samples, device=device)
        npoints, ndim = arr.shape
        assert ndim == 3

        t21 = torch.zeros((self.nfreq, npoints))
        for si, ss in enumerate(samples):
            a, w, n = ss
            t21[:, si] = torch.from_numpy(nf.T_DA(
                self.freqs, a, w, n, cmb=cmb, cosmicdawn=self.cosmicdawn
            )).to(device)

        pt21 = self.proj_t21(t21)

        if self.args.freqFluctuationLevel != 0.0:
            raise NotImplementedError

        final = (self.tfgmeansdata[:, None] - pt21).float().T

        return self.model.evaluate_density(final).cpu().numpy()


if __name__ == "__main__":
    args = nf.Args()
    args.SNRpp = 1e24
    args.combineSigma = "4 6"
    f = FlowLikelihood(args)
    f.likelihood([[1, 14, 16.4], [0.9, 13, 15.4]])
