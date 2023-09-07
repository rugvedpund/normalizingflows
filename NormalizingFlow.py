#30 Aug 2022, Rugved Pund

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

root=os.environ['LUSEE_ML'] #specify path to save/load models and likelihood results

def exp(l,numpy=True):
    if numpy: 
        return np.exp((l-max(l)))
    else: 
        return [np.exp((ll-max(l))/2) for ll in l]

def T_CMB(freqs):
    return 2.718*np.ones_like(freqs)

def T_DA(amp,width,nu_min,cmb=False):
    freqs=np.arange(1,51)
    if cmb: return amp*T_CMB(freqs)
    return amp*lusee.MonoSkyModels.T_DarkAges_Scaled(freqs,nu_rms=width,nu_min=nu_min)

def get_amp_width_numin(npoints,amin=0,amax=3.0,wmin=10.0,wmax=20.0,nmin=10.0,nmax=20.0,logspace=False):
    amp=np.logspace(np.log10(amin),np.log10(amax),num=npoints) if logspace else np.linspace(amin,amax,num=npoints) #default is 1.0
    width=np.linspace(wmin,wmax,num=npoints) #default is 14 MHz
    nu_min=np.linspace(nmin,nmax, num=npoints) #default is 16.4 MHz
    return amp,width,nu_min

def uniform_grid(npoints,**kwargs):
    amp,width,nu_min=get_amp_width_numin(npoints,**kwargs)
    g=np.meshgrid(amp,width,nu_min)
    samples=np.vstack(list(map(np.ravel,g))).T
    return samples

def get_t21vs(npoints,cmb=False,**kwargs):
    samples=uniform_grid(npoints,**kwargs)
    t21_vs=np.zeros((50,npoints**3))
    for i,(a,w,n) in enumerate(samples):
        t21_vs[:,i]=T_DA(a,w,n,cmb=cmb)
    return samples,t21_vs

def uniform_grid2d(npoints,vs,**kwargs): #vs= WvA NvA WvN
    amp,width,nu_min=get_amp_width_numin(npoints,**kwargs)
    if vs=='WvA': g=np.meshgrid(width,amp)
    if vs=='NvA': g=np.meshgrid(nu_min,amp)
    if vs=='WvN': g=np.meshgrid(width,nu_min)
    samples=np.vstack(list(map(np.ravel,g))).T
    return samples

def get_t21vs2d(npoints,vs,**kwargs):  #vs= WvA NvA WvN
    samples=uniform_grid2d(npoints,vs,**kwargs)
    t21_vs=np.zeros((50,npoints**2))
    for i,(x,y) in enumerate(samples):
        if vs=='WvA': t21_vs[:,i]=T_DA(amp=y,width=x,nu_min=16.4)
        if vs=='NvA': t21_vs[:,i]=T_DA(amp=y,width=14.0,nu_min=x)
        if vs=='WvN': t21_vs[:,i]=T_DA(amp=1.0,width=x,nu_min=y)
    return samples,t21_vs

def get_t21vs1d(npoints,vs,cmb=False,**kwargs):
    amp,width,nu_min=get_amp_width_numin(npoints,**kwargs)
    if vs=='A': 
        samples=amp
        tDA=lambda xx:T_DA(amp=xx,width=14.0,nu_min=16.4,cmb=cmb)
    if vs=='W': 
        samples=width
        tDA=lambda xx:T_DA(amp=1.0,width=xx,nu_min=16.4)
    if vs=='N': 
        samples=nu_min
        tDA=lambda xx:T_DA(amp=1.0,width=14.0,nu_min=xx)
    t21_vs=np.zeros((50,npoints))
    for i,xx in enumerate(samples):
        t21_vs[:,i]=tDA(xx)
    return samples,t21_vs

def get_fname(args):
    """for saving model after training"""
    cS=','.join(args.combineSigma.split())
    pca=','.join(args.nPCA.split())

    fname=f'{root}GIS_ulsa_nside128_sigma{args.sigma}_subsample{args.subsample_factor}_galcut{args.galcut}_noPCA{args.noPCA}_chromaticBeam{args.chromatic}_combineSigma{cS}_noise{args.noise}_seed{args.noiseSeed}_subsampleSigma{args.subsampleSigma}'
    if args.gainFluctuationLevel is not None: fname+=f'_gainFluctuation{args.gainFluctuationLevel}_gFdebug{args.gFdebug}'
    if args.append: fname+=args.append
    if args.nPCA: fname+=f'_nPCA{pca}'
    return fname

def get_lname(args,plot):
    """for saving likelihood results"""
    cS=','.join(args.combineSigma.split())
    pca=','.join(args.nPCA.split())
    lname=f'{root}corner/GIS_ulsa_nside128_sigma{args.sigma}_subsample{args.subsample_factor}_galcut{args.galcut}_noPCA{args.noPCA}_chromaticBeam{args.chromatic}_combineSigma{cS}_noise{args.noise}_seed{args.noiseSeed}_subsampleSigma{args.subsampleSigma}'
    if args.nPCA: lname+=f'_nPCA{pca}'
    if args.gainFluctuationLevel is not None: lname+=f'_gainFluctuation{args.gainFluctuationLevel}_gFdebug{args.gFdebug}'
    if args.append: lname+=args.append
    if args.appendLik: lname+=args.appendLik

    lname+=f'_noisyT21{args.noisyT21}_vs{plot}_DAfactor{args.DA_factor}_freqFluctuationLevel{args.freqFluctuationLevel}'
    return lname

def get_samplesAndLikelihood(args,plot,verbose=False):
    lname=get_lname(args,plot)
    if verbose: print(f'loading corner likelihood results from {lname}')
    f=np.loadtxt(lname,unpack=True)
    likelihood=f[-1]
    samples=f[:-1].T
    return samples,likelihood

def get_constraints(s,ll):
    #assume amp,width,numin
    params=['amp','width','numin']
    out={'amp':0.0,'amp+':0.0,'amp-':0.0,
         'width':0.0,'width+':0.0,'width-':0.0,
         'numin':0.0,'numin+':0.0,'numin-':0.0}
    for i in range(3):
        l2,l,m,h,h2=corner.core.quantile(s[:,i].copy(),[0.05,0.16,0.5,0.84,0.95],
                                   weights=exp(ll.copy()))
        out[params[i]]=m
        out[params[i]+'+']=h-m
        out[params[i]+'-']=m-l
        out[params[i]+'+-']=h2-l2
    # out['amp']*=40.0
    # out['amp+']*=40.0
    # out['amp-']*=40.0
    return out

def get_log(fname):
    with open(fname) as f:
        lines=f.readlines()
    n=len(lines)
    trainlogp,vallogp,iteration,best=np.zeros((4,n))
    for il,l in enumerate(lines):
        lsplit=l.split()
        trainlogp[il]=lsplit[1]
        vallogp[il]=lsplit[2]
        iteration[il]=lsplit[-3]
        best[il]=lsplit[-1]
    return iteration,trainlogp,vallogp,best

def get_nLayerBest(iteration,trainlogp,vallogp,best,delta_logp=10):
    b=int(best[-1])
    n=int(iteration[np.where(trainlogp-vallogp<delta_logp)][-1])
    return min(n-1,b-1)
    
class NormalizingFlow:
    def __init__(self,nocuda,loadPath=''):
        self.device=torch.device('cpu') if nocuda else torch.device('cuda')
        try: 
            self.model=torch.load(loadPath).to(self.device)
            self.nlayer=len(self.model.layer)
            print('model loaded from ',loadPath)
        except FileNotFoundError:
            print('no file found, need to train')
        self.precompute_data_after=dict()
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

class FlowAnalyzerV2(NormalizingFlow):
    def __init__(self,loadPath,nocuda=False):
        super().__init__(loadPath=loadPath,nocuda=nocuda)
    def set_fg(self,args):
        """
        fg: foreground map of shape (nfreq,npix)
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
        """

        self.fg=fitsio.read(f'{root}200.fits')
        self.sigma=args.sigma
        self.chromatic=args.chromatic
        self.galcut=args.galcut
        self.noise_K=args.noise
        self.noiseSeed=args.noiseSeed
        self.subsampleSigma=args.subsampleSigma
        self.combineSigma=[float(c) for c in args.combineSigma.split()]
        self.gainFluctuationLevel=args.gainFluctuationLevel
        self.gFdebug=args.gFdebug
        self.diffCombineSigma=args.diffCombineSigma
        self.avgAdjacentFreqBins=args.avgAdjacentFreqBins

        #fg 
        self.nfreq,self.npix=self.fg.shape
        #set seed
        np.random.seed(self.noiseSeed)
        #generate noise
        self.noise=self.generate_noise(self.fg.copy(),self.noise_K,self.subsampleSigma)
        sigmas=[self.sigma]+self.combineSigma
        print('doing sigmas', sigmas)
        self.nsigmas=len(sigmas)
        self.fg_cS=np.zeros((self.nsigmas,*self.fg.shape))
        self.fgnoisy=self.fg.copy()+self.noise.copy()
        
        for isig,sig in enumerate(sigmas):
            self.fg_cS[isig,:,:]=self.smooth(self.fgnoisy.copy(),sig,self.chromatic)
            self.fg_cS[isig,:,:]=self.includeGainFluctuations(self.fg_cS[isig,:,:].copy(),
                                                              self.gainFluctuationLevel,self.gFdebug)
        self.fgsmooth=self.fg_cS.reshape(self.nsigmas*self.nfreq,-1)
        print(f'{self.fgsmooth.shape=}')

        #do diffCombineSigma
        if self.diffCombineSigma:
            self.fgsmooth_diff=np.zeros_like(self.fgsmooth)
            # difference between fg_cS and fg_cS[0] except for fg_cS[0]
            self.fgsmooth_diff[:self.nfreq,:]=self.fg_cS[0,:,:].copy()
            for isig,sig in enumerate(sigmas):
                if isig==0: continue
                print('doing diffCombineSigma for combineSigma')
                self.fgsmooth_diff[isig*self.nfreq:(isig+1)*self.nfreq,:]=self.fg_cS[isig,:,:].copy()-self.fg_cS[isig-1,:,:].copy()
            self.fgsmooth=self.fgsmooth_diff.copy()
            print(f'{self.fgsmooth.shape=}')
        
        #do avgAdjacentFreqBins
        if self.avgAdjacentFreqBins:
            print('combining adjacent freq bins')
            self.nfreq=self.nfreq//2
            self.fgsmooth_cAFB=np.zeros((self.nsigmas,self.nfreq,self.npix))
            self.fgsmooth_cAFB=(self.fg_cS[:,::2,:]+self.fg_cS[:,1::2,:])/2
            self.fgsmooth=self.fgsmooth_cAFB.reshape(self.nsigmas*self.nfreq,-1)
            print(f'{self.fgsmooth.shape=}')

        # #do PCA with full map with galcut (no subsampling)
        self.fgcut=self.galaxy_cut(self.fgsmooth.copy(),self.galcut)
        # self.cov=np.cov(self.fgcut)
        # eva,eve=np.linalg.eig(self.cov)
        # s=np.argsort(np.abs(eva))[::-1]
        # self.eva,self.eve=np.abs(eva[s]),eve[:,s]
        
        #subsample and project
        self.fgss=self.subsample(self.fgsmooth.copy(),self.subsampleSigma,self.galcut)

        #do SVD
        self.eve,self.eva,self.vt=np.linalg.svd(self.fgss)

        print(f'{self.fgss.shape=}')
        proj_fg=self.eve.T@(self.fgss - self.fgss.mean(axis=1)[:,None])
        self.pfg=proj_fg.copy()
        self.rms=np.sqrt(proj_fg.var(axis=1))
        out=proj_fg/self.rms[:,None]
        self.data=np.random.permutation(out.T) #transpose and permute for torch
        self.fgmeansdata=(self.eve.T@self.fgss.mean(axis=1))/self.rms
        print(f'{self.data.shape=} {self.fgmeansdata.shape=} {self.eve.shape=}')
        
        #remove nPCA modes
        self.nPCA=[0,self.nfreq*self.nsigmas] if len(args.nPCA)==0 else [int(p) for p in args.nPCA.split()]
        print(f'using modes between {args.nPCA}')
        self.nPCAarr=np.hstack([np.arange(self.nPCA[0]),np.arange(self.nPCA[1],self.nfreq*self.nsigmas)])
        print(self.nPCAarr)
        self.data = np.delete(self.data,self.nPCAarr,axis=1) #delete last nPCA columns
        self.fgmeansdata = np.delete(self.fgmeansdata,self.nPCAarr)
        print(f'{self.data.shape=} {self.fgmeansdata.shape=} {self.eve.shape=}')

        ndata,_=self.data.shape
        ntrain=int(0.8*ndata)
        self.train_data=self.data[:ntrain,:].copy()
        self.validate_data=self.data[ntrain:,:].copy()
        print(f'done! {self.train_data.shape=},{self.validate_data.shape=} ready')
    
    def set_t21(self,t21,include_noise):
        if include_noise: t21+=self.noise.mean(axis=1)
        if self.avgAdjacentFreqBins: 
            print('combining adjacent freq bins for t21')
            t21=(t21[::2]+t21[1::2])/2
        self.t21=np.tile(t21,self.nsigmas)
        self.pt21=self.eve.T@self.t21
        self.t21data=self.eve.T@self.t21/self.rms
        self.t21data=np.delete(self.t21data,self.nPCAarr)
        print(f'{self.t21data.shape=} ready')
    
    def proj_t21(self,t21_vs,include_noise):
        t21_noisy=t21_vs+self.noise.mean(axis=1)[:,None] if include_noise else np.zeros_like(t21_vs)
        if self.avgAdjacentFreqBins:
            t21_noisy=(t21_noisy[::2,:]+t21_noisy[1::2,:])/2
        t21cS=np.tile(t21_noisy,(self.nsigmas,1))
        proj_t21=(self.eve.T@t21cS)/self.rms[:,None]
        proj_t21=np.delete(proj_t21,self.nPCAarr,axis=0)
        return proj_t21
        
    def generate_noise(self,fg,noise_K,subsampleSigma):
        nfreq,ndata=fg.shape
        nside=self.sigma2nside(subsampleSigma)
        npix=hp.nside2npix(nside)
        noise_sigma=noise_K*np.sqrt(npix)
        noise=np.random.normal(0,noise_sigma,(nfreq,ndata))
        return noise
    
    def sigma2nside(self, sigma):
        #done by hand
        sigma2nside_map={0.5:64, 1.0:32, 2.0:16, 4.0:8, 8.0:4}
        try:
            nside=sigma2nside_map[sigma]
        except KeyError:
            print('sigmas must be either 0.5, 1.0, 2.0, 4.0, or 8.0')
            raise NotImplementedError
        return sigma2nside_map[sigma]
    
    def smooth(self,fg,sigma,chromatic):
        print(f'smoothing with {sigma} deg, and chromatic {chromatic}')
        fgsmooth=np.zeros_like(fg)
        nfreq,ndata=fg.shape
        for f in range(nfreq):
            sigma_f=sigma*(10.0/(f+1)) if chromatic else sigma
            fgsmooth[f,:]=hp.sphtfunc.smoothing(fg[f,:],sigma=np.deg2rad(sigma_f))
        return fgsmooth
    
    def includeGainFluctuations(self, fgsmooth, gainFluctuationLevel, gFdebug):
        print(f'including gain fluctuations of level: {gainFluctuationLevel}')
        _,ndata=fgsmooth.shape
        freqs=np.arange(1,51)
        self.gainF=np.random.normal(0,gainFluctuationLevel if gainFluctuationLevel is not None else 0.0,ndata)
        template=lusee.MonoSkyModels.T_DarkAges_Scaled(freqs,nu_rms=14,nu_min=16.4,A=0.04)
        fgmeans=fgsmooth.mean(axis=1)
        if gFdebug==0: return fgsmooth.copy()
        if gFdebug==1: return fgsmooth.copy() + fgsmooth.copy()*self.gainF.copy()
        if gFdebug==2: return fgsmooth.copy() + np.outer(template,self.gainF.copy())
        if gFdebug==3: return fgsmooth.copy() + fgsmooth.copy()*self.gainF.copy() + np.outer(template,1.0+self.gainF.copy())
        # if anything else
        raise NotImplementedError
    
    def galaxy_cut(self,fg,b_min):
        print(f'doing galcut for b_min={b_min} deg')
        _,npix=fg.shape
        nside=np.sqrt(npix/12).astype(int)
        col_min = np.pi/2 - np.deg2rad(b_min)
        col_max = np.pi/2 + np.deg2rad(b_min)
        mask_pix = hp.query_strip(nside, col_min, col_max, inclusive=False)
        cutfg = np.delete(fg, mask_pix, axis=1)
        return cutfg 
    
    def subsample(self,fg,subsampleSigma,galcut):
        nside=self.sigma2nside(subsampleSigma)
        ipix=np.arange(hp.nside2npix(nside))
        theta,phi=hp.pix2ang(nside,ipix)
        ipix_128=hp.ang2pix(128,theta,phi)
        gal=hp.query_strip(nside,np.pi/2-np.deg2rad(galcut),np.pi/2+np.deg2rad(galcut),
                   inclusive=False)
        galpix_128=hp.ang2pix(*(128,*hp.pix2ang(nside,gal)))
        maskpix=np.setdiff1d(ipix_128,galpix_128)
        fg_subsample=fg[:,maskpix].copy()
        return fg_subsample
        
    def get_likelihood(self,t21_vsdata,freqFluctuationLevel,DA_factor=1.0, debugfF=False,end=None):
        freqs=np.tile(np.arange(1,51),self.nsigmas)
        flowt21data_fF=self.t21data if debugfF else (1+freqFluctuationLevel*np.cos(6*np.pi/50*freqs))*self.t21data
        l=(self.fgmeansdata[:,None]+DA_factor*flowt21data_fF[:,None]-t21_vsdata).T
        return self._likelihood(l,end).cpu().numpy()

class Args:
    def __init__(self,sigma=2.0,
        subsample_factor=None,
        galcut=20.0,
        noPCA=False,
        chromatic=False,
        combineSigma='',
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
        appendLik=''
        ):

        self.sigma=sigma
        self.subsample_factor=subsample_factor
        self.galcut=galcut
        self.noPCA=noPCA
        self.chromatic=chromatic
        self.combineSigma=combineSigma
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
    
    def print(self):
        for arg in vars(self):
            print(arg, getattr(self, arg))