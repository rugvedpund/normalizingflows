import torch
from SINF import GIS
import lusee
import numpy as np
import healpy as hp
import fitsio
import os
import corner


root=os.environ['NF_WORKDIR']

class Args:

    def __init__(self,sigma=2.0,
        subsample_factor=None,
        galcut=20.0,
        noPCA=False,
        chromatic=False,
        combineSigma='',
        SNRpp=None,
        noise=0.0,
        noiseSeed=0,
        subsampleSigma=2.0,
        noisyT21=True,
        gainFluctuationLevel=0.0,
        gFdebug=0,
        append='_SVD',
        DA_factor=1.0,
        plot='all', # 1dAmplitude 1dWidth 1dNuMin WvA NvA WvN
        freqFluctuationLevel=0.0,
        nPCA='',
        diffCombineSigma=True,
        avgAdjacentFreqBins=False,
        retrain=False,
        appendLik='',
        fgFITS='ulsa.fits',
        freqs='1 51'
        ):

        self.sigma=sigma
        self.subsample_factor=subsample_factor
        self.galcut=galcut
        self.noPCA=noPCA
        self.chromatic=chromatic
        self.combineSigma=combineSigma
        self.SNRpp=SNRpp
        self.noise=noise
        self.noiseSeed=noiseSeed
        self.subsampleSigma=subsampleSigma
        self.noisyT21=noisyT21
        self.gainFluctuationLevel=gainFluctuationLevel
        self.gFdebug=gFdebug
        self.append=append
        self.DA_factor=DA_factor
        self.plot=plot
        self.freqFluctuationLevel=freqFluctuationLevel
        self.nPCA=nPCA
        self.diffCombineSigma=diffCombineSigma
        self.avgAdjacentFreqBins=avgAdjacentFreqBins
        self.retrain=retrain
        self.appendLik=appendLik
        self.fgFITS=fgFITS
        self.freqs=freqs

    def print(self):
        for arg in vars(self):
            print(arg, getattr(self, arg))

    def get_fname(self):
        """for saving model after training"""
        cS=','.join(self.combineSigma.split())
        pca=','.join(self.nPCA.split())
        fqs=','.join(self.freqs.split())
        fgname=self.fgFITS.split('.')[0]
        fname=f'{root}GIS_{fgname}_nside128_sigma{self.sigma}_subsample{self.subsample_factor}_galcut{self.galcut}_noPCA{self.noPCA}_chromaticBeam{self.chromatic}_combineSigma{cS}'
        if self.SNRpp is not None: 
            fname+=f'_SNRpp{self.SNRpp:.0e}'
        else:
            fname+=f'_noise{self.noise}'
        fname+=f'_seed{self.noiseSeed}_subsampleSigma{self.subsampleSigma}'
        if self.gainFluctuationLevel is not None: fname+=f'_gainFluctuation{self.gainFluctuationLevel}_gFdebug{self.gFdebug}'
        if self.append: fname+=self.append
        if self.nPCA: fname+=f'_nPCA{pca}'
        if not args.old: fname+=f'_freqs{fqs}'
        return fname

    def get_lname(self,plot):
        """for saving likelihood results"""
        lname=self.get_fname(self)
        if self.appendLik: lname+=f'like{self.appendLik}'
        lname+=f'_noisyT21{self.noisyT21}_vs{plot}_DAfactor{self.DA_factor}_freqFluctuationLevel{self.freqFluctuationLevel}'
        if self.old: 
            paths=lname.split('/')
            paths.insert(6,'corner')
            lname='/'.join(paths)
        return lname

    def get_samplesAndLikelihood(self,plot,verbose=False):
        lname=self.get_lname(self,plot)
        if verbose: print(f'loading corner likelihood results from {lname}')
        f=np.loadtxt(lname,unpack=True)
        likelihood=f[-1]
        samples=f[:-1].T
        return samples,likelihood

class NormalizingFlow:

    def __init__(self,nocuda,loadPath=''):
        self.device=torch.device('cpu') if nocuda else torch.device('cuda')
        self.loadModel(loadPath)

    def loadModel(self,loadPath)
        try: 
            self.model=torch.load(loadPath).to(self.device)
            self.nlayer=len(self.model.layer)
            print('model loaded from ',loadPath)
        except FileNotFoundError:
            print('no file found, need to train')

    def train(self,train_data,validate_data,nocuda,savePath,retrain,alpha=None,delta_logp=np.inf,
              verbose=False):
        """
        data must be of shape (nsamples, ndim)
        """
        self.nsamples,self.ndim=train_data.shape
        print('now training model with nsamples', self.nsamples,' of ndim',self.ndim)
        self.train_data=self._toTensor(train_data)
        self.validate_data=self._toTensor(validate_data)
        if retrain: 
            print('retraining...')
            self.model=None
        self.model=GIS.GIS(self.train_data.clone(),self.validate_data.clone(),nocuda=nocuda,
                           alpha=alpha,delta_logp=delta_logp,verbose=verbose)
        self.nlayer=len(self.model.layer)
        print('Total number of iterations: ', len(self.model.layer))
        torch.save(self.model,savePath)
        print('model saved at', savePath)

    def _toTensor(self,nparray):
        return torch.from_numpy(nparray).float().to(self.device)

    def _likelihood(self,data,end=None):
        data=self._toTensor(data)
        return self.model.evaluate_density(data,end=end)

class ForegroundModel:

    def __init__(self,args):
        self.args=args
        self.setParamsfromArgs()
        self.fg=self.loadfgFITS(self.args.fgFITS)
        self.freqs=self.setFrequencyRange(self.args.freqs)
        self.noise=self.generateNoise()
        self.fg_cS=self.getCombineSigmas(self.fg.copy())
        self.fg_gF=self.multiplyGainFluctuations(self.fg_cS.copy())
        self.fg_dCS=self.doDiffCombineSigma(self.fg_gF.copy())
        self.fgsmooth=self.fg_dCS.copy().reshape(self.nsigmas*self.nfreq,-1)
        print(f'foreground model initialized with shape {self.fgsmooth.shape}')
        self.fgss=self.subsample(self.fgsmooth.copy(),self.subsampleSigma,self.galcut)
        self.data=self.doPCA(self.fgss.copy())
        self.train_data,self.validate_data=self.getTrainValidateData()
        
    def setParamsfromArgs(self):
        print('setting params from args')
        self.fmin,self.fmax=[int(f) for f in self.args.freqs.split()]
        self.nfreq,self.npix=self.fg.shape
        self.sigmas=[self.sigma]+[float(s) for s in self.args.combineSigma.split()]
        self.nsigmas=len(self.sigmas)

    def loadfgFITS(self,fgFITS):
        print('loading fg map ',fgFITS)
        return fitsio.read(root+fgFITS)

    def setFrequencyRange(self,freqs):
        freqrange=np.arange(self.fmin,self.fmax)
        assert self.nfreq==len(freqrange)
        return freqrange

    def generateNoise(self):
            if self.args.SNRpp is not None:
                return self.generateRadiometerNoise()
            else:
                return self.generateFlatNoise()

    def generateRadiometerNoise(self):
        print('generating radiometer noise with SNRpp',self.args.SNRpp)
        return np.zeros_like(self.fg)

    def generateFlatNoise(self):
        print('generating flat noise with level',self.args.noise)
        return np.zeros_like(self.fg)

    def smooth(self,fg,sigma,chromatic):
        print(f'smoothing fg with sigma {sigma} and chromatic {chromatic}')
        fgsmooth=np.zeros_like(fg)
        for fi,f in enumerate(self.freqs):
            sigma_f=sigma*(10.0/(f)) if chromatic else sigma
            fgsmooth[fi,:]=hp.sphtfunc.smoothing(fg[fi,:],sigma=np.deg2rad(sigma_f))
        return fgsmooth

    def getCombineSigmas(self,fg):
        print('doing self.sigmas',self.sigmas)
        nsigmas=len(self.sigmas)
        nfreq,npix=fg.shape
        fgsmooth=np.zeros((nsigmas,nfreq,npix))
        fgnoisy=fg.copy()+self.noise.copy()
        for isig,sig in enumerate(self.sigmas):
            fgsmooth[isig,:,:]=self.smooth(fgnoisy.copy(),sig,self.args.chromatic)
        return fgsmooth

    def multiplyGainFluctuations(self,fg_cS):
        if self.args.gainFluctuationLevel==0.0: return fg_cS
        print('multiplying gain fluctuations with level',self.args.gainFluctuationLevel)
        raise NotImplementedError
        # print('multiplying gain fluctuations with level',self.args.gainFluctuationLevel)
        # gainFmap=self.getGainFluctuationMap()
        # for isig,sig in enumerate(self.sigmas):
        #     fg_cS[isig,:,:]*=gainFmap

    def doDiffCombineSigma(self,fg):
        if not self.args.diffCombineSigma: return fg
        print('doing diffCombineSigma for combineSigmas',self.sigmas)
        assert fg.shape==(self.nsigmas,self.nfreq,self.npix)
        fg_dCS=np.zeros_like(fg)

        for isig,sig in enumerate(self.sigmas):
            if isig==0: fg_dCS[isig,:,:]=fg[isig,:,:]
            fg_dCS[isig,:,:]=fg[isig,:,:]-fg[isig-1,:,:]
        return fg_dCS

    def sigma2nside(self, sigma):
        #done by hand
        sigma2nside_map={0.5:64, 1.0:32, 2.0:16, 4.0:8, 8.0:4}
        try:
            nside=sigma2nside_map[sigma]
        except KeyError:
            print('sigmas must be either 0.5, 1.0, 2.0, 4.0, or 8.0')
            raise NotImplementedError
        return sigma2nside_map[sigma]

    def subsample(self,fg,subsampleSigma,galcut):
        print('subsample fg with subsampleSigma',subsampleSigma,'and galcut',galcut)
        nside=self.sigma2nside(subsampleSigma)
        ipix=np.arange(hp.nside2npix(nside))
        theta,phi=hp.pix2ang(nside,ipix)
        ipix_128=hp.ang2pix(128,theta,phi)
        gal=hp.query_strip(nside,np.pi/2-np.deg2rad(galcut),
                           np.pi/2+np.deg2rad(galcut), inclusive=False)
        galpix_128=hp.ang2pix(*(128,*hp.pix2ang(nside,gal)))
        maskpix=np.setdiff1d(ipix_128,galpix_128)
        fg_subsample=fg[:,maskpix].copy()
        return fg_subsample

    def doPCA(self,fgss):
        print('doing SVD for fgss of shape',fgss.shape)
        #do SVD
        self.eve,self.eva,self.vt=np.linalg.svd(fgss - fgss.mean(axis=1)[:,None])
        # print('new')
        print(f'{fgss.shape=}')
        proj_fg=self.eve.T@(fgss - fgss.mean(axis=1)[:,None])
        self.pfg=proj_fg.copy()
        self.rms=np.sqrt(proj_fg.var(axis=1))
        out=proj_fg/self.rms[:,None]
        self.data=np.random.permutation(out.T) #transpose and permute for torch
        self.fgmeansdata=(self.eve.T@fgss.mean(axis=1))/self.rms
        print(f'{self.data.shape=} {self.fgmeansdata.shape=} {self.eve.shape=}')
        return self.data

    def getTrainValidateData(self):
        ndata=self.data.shape[0]
        ntrain=int(ndata*0.8)
        train_data=self.data[:ntrain,:].copy()
        validate_data=self.data[ntrain:,:].copy()
        print('training and validation data ready', train_data.shape, validate_data.shape)
        return train_data,validate_data
        

class SignalModel:
    def __init__(self,args,fgModel):
        self.args=args
        self.fgModel=fgModel
        self.t21=self.getT21()
        self.addNoise()

    def getT21(self):
        if 
    def addNoise(self):
        print('adding noise from ForegroundModel')
        self.
