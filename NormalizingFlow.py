#30 Aug 2022, Rugved Pund

import torch
from sinf import GIS
import matplotlib.pyplot as plt
import numpy as np
import lusee
import matplotlib
import healpy as hp
import fitsio

def exp(l,numpy=True):
    if numpy: 
        return np.exp((l-max(l)))
    else: 
        return [np.exp((ll-max(l))/2) for ll in l]

def T_DA(amp,width,nu_min):
    freqs=np.arange(1,51)
    return amp*lusee.mono_sky_models.T_DarkAges_Scaled(freqs,nu_rms=width,nu_min=nu_min)

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

def get_t21vs(npoints,**kwargs):
    samples=uniform_grid(npoints,**kwargs)
    t21_vs=np.zeros((50,npoints**3))
    for i,(a,w,n) in enumerate(samples):
        t21_vs[:,i]=T_DA(a,w,n)
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

def get_t21vs1d(npoints,vs,**kwargs):
    amp,width,nu_min=get_amp_width_numin(npoints,**kwargs)
    if vs=='A': 
        samples=amp
        tDA=lambda xx:T_DA(amp=xx,width=14.0,nu_min=16.4)
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
    root='/home/rugved/Files/LuSEE/ml/'
    cS=','.join(args.combineSigma.split())

    fname=f'{root}GIS_ulsa_nside128_sigma{args.sigma}_subsample{args.subsample_factor}_galcut{args.galcut}_noPCA{args.noPCA}_chromaticBeam{args.chromatic}_combineSigma{cS}_noise{args.noise}_seed{args.noiseSeed}_subsampleSigma{args.subsampleSigma}'
    if args.gainFluctuationLevel is not None: fname+=f'_gainFluctuation{args.gainFluctuationLevel}_gFdebug{args.gFdebug}'
    if args.append: fname+=args.append
    if args.nPCA: fname+=f'_nPCA{pca}'
    return fname

def get_lname(args,plot):
    """for saving likelihood results"""
    root='/home/rugved/Files/LuSEE/ml/'
    cS=','.join(args.combineSigma.split())
    pca=','.join(args.nPCA.split())
    lname=f'{root}corner/GIS_ulsa_nside128_sigma{args.sigma}_subsample{args.subsample_factor}_galcut{args.galcut}_noPCA{args.noPCA}_chromaticBeam{args.chromatic}_combineSigma{cS}_noise{args.noise}_seed{args.noiseSeed}_subsampleSigma{args.subsampleSigma}'
    if args.nPCA: lname+=f'_nPCA{pca}'
    if args.gainFluctuationLevel is not None: lname+=f'_gainFluctuation{args.gainFluctuationLevel}_gFdebug{args.gFdebug}'
    if args.append: lname+=args.append

    lname+=f'_noisyT21{args.noisyT21}_vs{plot}_DAfactor{args.DA_factor}_freqFluctuationLevel{args.freqFluctuationLevel}'
    return lname

def get_samplesAndLikelihood(args,plot):
    lname=get_lname(args,plot)
    print(f'loading corner likelihood results from {lname}')
    f=np.loadtxt(lname,unpack=True)
    likelihood=f[-1]
    samples=f[:-1].T
    return samples,likelihood

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
    def train(self,train_data,validate_data,nocuda,savePath,retrain,alpha=None,delta_logp=np.inf):
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
                           alpha=alpha,delta_logp=delta_logp)
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
    def set_fg(self,fg,sigma,chromatic,galcut,subsampleSigma,noise_K,noiseSeed,combineSigma,gFLevel,gFdebug,nPCA,
               diffCombineSigma,avgAdjacentFreqBins):
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


        self.fg=fg
        self.sigma=sigma
        self.chromatic=chromatic
        self.galcut=galcut
        self.noise_K=noise_K
        self.noiseSeed=noiseSeed
        self.subsampleSigma=subsampleSigma
        self.combineSigma=[float(c) for c in combineSigma.split()]
        self.gainFluctuationLevel=gFLevel
        self.gFdebug=gFdebug
        self.diffCombineSigma=diffCombineSigma
        self.avgAdjacentFreqBins=avgAdjacentFreqBins

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
            print('doing diffCombineSigma for combineSigma')
            self.fgsmooth_diff=np.zeros_like(self.fgsmooth)
            # difference between fg_cS and fg_cS[0] except for fg_cS[0]
            self.fgsmooth_diff[:self.nfreq,:]=self.fg_cS[0,:,:].copy()
            for isig,sig in enumerate(sigmas):
                if isig==0: continue
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

        #do PCA with full map with galcut (no subsampling)
        self.fgcut=self.galaxy_cut(self.fgsmooth.copy(),self.galcut)
        self.cov=np.cov(self.fgcut)
        eva,eve=np.linalg.eig(self.cov)
        s=np.argsort(np.abs(eva))[::-1]
        self.eva,self.eve=np.abs(eva[s]),eve[:,s]
        
        #subsample and project
        self.fgss=self.subsample(self.fgsmooth.copy(),self.subsampleSigma,self.galcut)
        print(f'{self.fgss.shape=}')
        proj_fg=self.eve.T@(self.fgss - self.fgss.mean(axis=1)[:,None])
        self.rms=np.sqrt(proj_fg.var(axis=1))
        out=proj_fg/self.rms[:,None]
        self.data=np.random.permutation(out.T) #transpose and permute for torch
        self.fgmeansdata=(self.eve.T@self.fgss.mean(axis=1))/self.rms
        print(f'{self.data.shape=} {self.fgmeansdata.shape=} {self.eve.shape=}')
        
        #remove nPCA modes
        self.nPCA=[0,self.nfreq*self.nsigmas] if len(nPCA)==0 else [int(p) for p in nPCA.split()]
        print(f'using modes between {nPCA}')
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
        self.t21data=self.eve.T@self.t21/self.rms
        self.t21data=np.delete(self.t21data,self.nPCAarr)
        print(f'{self.t21data.shape=} ready')
    
    def proj_t21(self,t21_vs,include_noise):
        if include_noise: t21_noisy=t21_vs+self.noise.mean(axis=1)[:,None]
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
        template=lusee.mono_sky_models.T_DarkAges_Scaled(freqs,nu_rms=14,nu_min=16.4,A=0.04)
        fgmeans=fgsmooth.mean(axis=1)
        #does not work for combineSigma
        if gFdebug==0: return fgsmooth
        if gFdebug==1: return fgsmooth + fgsmooth*self.gainF
        if gFdebug==2: return fgsmooth + np.outer(template,self.gainF)
        if gFdebug==3: return fgsmooth + fgsmooth*self.gainF + np.outer(template,self.gainF)
        
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
<<<<<<< Updated upstream
=======

# def load_ulsa_t21_tcmb_freqs():
#     root='/home/rugved/Files/LuSEE/luseepy/Drive/'
#     fg=fitsio.read(root+'Simulations/SkyModels/ULSA_32_ddi_smooth.fits')
#     freqs=np.arange(1,51)
#     t21=lusee.mono_sky_models.T_DarkAges(freqs)
#     tcmb=np.ones_like(t21)*2.73
#     return fg, t21, tcmb, freqs

# def smooth(fg,sigma,chromatic):
#     fgsmooth=np.zeros_like(fg)
#     nfreq,_=fg.shape
#     for f in range(nfreq):
#         sigma_f=sigma*(10.0/(f+1)) if chromatic else sigma
#         fgsmooth[f,:]=hp.sphtfunc.smoothing(fg[f,:],sigma=np.deg2rad(sigma_f))
#     return fgsmooth

# def get_projected_data(data, vector=None):
#     """
#     data should be of shape (nfreqs,ndata)
#     vector should be of shape (nfreqs)
#     out is of shape (ndata,nfreqs) - because torch needs it that way
#     """
#     assert data.shape[0]<data.shape[1]
#     # if vector is not None: assert vector.shape==data.shape[0]
#     cov=np.cov(data)
#     eva,eve=np.linalg.eig(cov)
#     s=np.argsort(np.abs(eva))[::-1] 
#     eva,eve=np.real(eva[s]),np.real(eve[:,s])
#     proj_data=eve.T@(data-data.mean(axis=1)[:,None]) #subtract mean and project
#     rms=np.sqrt(proj_data.var(axis=1))
#     # out=(proj_data/rms[:,None]).T #divide by rms
#     # out=np.random.permutation(out)
#     out=(proj_data/rms[:,None])
#     if vector is None: 
#         return out
#     else:
#         return out, (eve.T@vector)/rms

def exp(l,numpy=True):
    if numpy: 
        return np.exp((l-max(l)))
    else: 
        return [np.exp((ll-max(l))/2) for ll in l]
# def exp2d(a):
#     return np.exp(a-a.max())
# def lin2d(a):
#     return a-a.max()
# def normexp(a):
#     b=exp2d(a)
#     return b/b.sum()
# def plot_contours(m,xlim='amp',**kwargs):
#     maxlike=m.max()
#     llist=np.sort(m.flatten())[::-1]

#     for l in llist: 
#         if m[np.where(m>l)].sum()<0.3935: m1sigma=l
#         if m[np.where(m>l)].sum()<0.8647: m2sigma=l

#     ny,nx=m.shape
#     y=np.linspace(10,20,num=ny)
#     x=np.linspace(0,3,num=nx) if xlim=='amp' else np.linspace(10,20,num=nx)
#     xx,yy=np.meshgrid(x,y)
#     z=m.copy()
#     ax=plt.gca()
#     ax.contour(xx,yy,z,[m2sigma],**kwargs)
#     ax.contourf(xx,yy,z,[m1sigma,1.0],**kwargs)    
    
def T_DA(amp,width,nu_min):
    freqs=np.arange(1,51)
    return amp*lusee.mono_sky_models.T_DarkAges_Scaled(freqs,nu_rms=width,nu_min=nu_min)

def get_amp_width_numin(npoints,amin=0,amax=3.0,wmin=10.0,wmax=20.0,nmin=10.0,nmax=20.0,logspace=False):
    amp=np.logspace(np.log10(amin),np.log10(amax),num=npoints) if logspace else np.linspace(amin,amax,num=npoints) #default is 1.0
    width=np.logspace(np.log10(wmin),np.log10(wmax),num=npoints) if logspace else np.linspace(wmin,wmax,num=npoints) #default is 14 MHz
    nu_min=np.linspace(nmin,nmax, num=npoints) #default is 16.4 MHz
    return amp,width,nu_min

def uniform_grid(npoints,**kwargs):
    amp,width,nu_min=get_amp_width_numin(npoints,**kwargs)
    g=np.meshgrid(amp,width,nu_min)
    samples=np.vstack(list(map(np.ravel,g))).T
    return samples

def get_t21vs(npoints,**kwargs):
    samples=uniform_grid(npoints,**kwargs)
    t21_vs=np.zeros((50,npoints**3))
    for i,(a,w,n) in enumerate(samples):
        t21_vs[:,i]=T_DA(a,w,n)
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

def get_t21vs1d(npoints,vs,**kwargs):
    amp,width,nu_min=get_amp_width_numin(npoints,**kwargs)
    if vs=='A': 
        samples=amp
        tDA=lambda xx:T_DA(amp=xx,width=14.0,nu_min=16.4)
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
    root='/astro/u/rugvedpund/LuSEE/ml/'
    cS=','.join(args.combineSigma.split())

    fname=f'{root}GIS_ulsa_nside128_sigma{args.sigma}_subsample{args.subsample_factor}_galcut{args.galcut}_noPCA{args.noPCA}_chromaticBeam{args.chromatic}_combineSigma{cS}_noise{args.noise}_seed{args.noiseSeed}_subsampleSigma{args.subsampleSigma}'
    if args.gainFluctuationLevel is not None: fname+=f'_gainFluctuation{args.gainFluctuationLevel}_gFdebug{args.gFdebug}'
    if args.append: fname+=args.append
    if args.nPCA: fname+=f'_nPCA{pca}'
    return fname

def get_lname(args,plot):
    """for saving likelihood results"""
    root='/astro/u/rugvedpund/LuSEE/ml/'
    cS=','.join(args.combineSigma.split())
    pca=','.join(args.nPCA.split())
    lname=f'{root}corner/GIS_ulsa_nside128_sigma{args.sigma}_subsample{args.subsample_factor}_galcut{args.galcut}_noPCA{args.noPCA}_chromaticBeam{args.chromatic}_combineSigma{cS}_noise{args.noise}_seed{args.noiseSeed}_subsampleSigma{args.subsampleSigma}'
    if args.nPCA: lname+=f'_nPCA{pca}'
    if args.gainFluctuationLevel is not None: lname+=f'_gainFluctuation{args.gainFluctuationLevel}_gFdebug{args.gFdebug}'
    if args.append: lname+=args.append

    lname+=f'_noisyT21{args.noisyT21}_vs{plot}_DAfactor{args.DA_factor}_freqFluctuationLevel{args.freqFluctuationLevel}'
    return lname

def get_samplesAndLikelihood(args,plot):
    lname=get_lname(args,plot)
    print(f'loading corner likelihood results from {lname}')
    f=np.loadtxt(lname,unpack=True)
    likelihood=f[-1]
    samples=f[:-1].T
    return samples,likelihood

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
    
    

# class FlowAnalyzer(NormalizingFlow):
#     def __init__(self, nocuda, loadPath):
#         super().__init__(nocuda=nocuda,loadPath=loadPath)
#     def precomputeIterations(self,data,niterationSteps):
#         self._precomputeIterations(data,niterationSteps)
#     def set_fg(self, fg, sigma, chromatic, galcut, noPCA, subsample, noise_K, noiseSeed, combineSigma, subsampleSigma, gainFluctuationLevel,gFdebug):
#         self.fg=fg
#         self.sigma=sigma
#         self.chromatic=chromatic
#         self.galcut=galcut
#         self.noPCA=noPCA
#         self.subsample=subsample
#         self.noise_K=noise_K
#         self.noiseSeed=noiseSeed
#         self.combineSigma=combineSigma
#         # self.ncS=len(combineSigma) if combineSigma is not None else 0
#         self.subsampleSigma=subsampleSigma
#         self.gainFluctuationLevel=gainFluctuationLevel
#         self.gFdebug=gFdebug
        
#         self.noise=self.generate_noise(self.fg,self.noise_K,self.subsample,self.subsampleSigma,self.noiseSeed)
#         self.fg_noisy=self.fg.copy()+self.noise
#         self.fgsmooth=self.smooth(self.fg_noisy,self.sigma,self.chromatic)
        
#         if self.galcut is not None:
#             self.fgsmooth_cut=self.galaxy_cut(self.fgsmooth.copy(),self.galcut)
#         else:
#             self.fgsmooth_cut=self.fgsmooth.copy()
            
#         if self.subsample is not None:
#             if self.noPCA: 
#                 self.data,self.fgmeansdata=self.get_data_noPCA(self.fgsmooth_cut, vector=self.fgsmooth_cut.mean(axis=1))
#             else:
#                 self.data,self.fgmeansdata=self.get_data_PCA(self.fgsmooth_cut, vector=self.fgsmooth_cut.mean(axis=1))

#             if self.combineSigma is not None:
#                 self.fgsmooth2=self.smooth(self.fg,self.combineSigma,self.chromatic)
#                 if self.galcut is not None:
#                     self.fgsmooth_cut2=self.galaxy_cut(self.fgsmooth2,self.galcut)
#                 else:
#                     self.fgsmooth_cut2=self.fgsmooth2
#                 self.fgsmoothcut_combineSigma=np.vstack([self.fgsmooth_cut,self.fgsmooth_cut2])
#                 if self.noPCA: 
#                     self.data,self.fgmeansdata=self.get_data_noPCA(self.fgsmoothcut_combineSigma, vector=self.fgsmoothcut_combineSigma.mean(axis=1))
#                 else:
#                     self.data,self.fgmeansdata=self.get_data_PCA(self.fgsmoothcut_combineSigma, vector=self.fgsmoothcut_combineSigma.mean(axis=1))
#             assert self.data.shape[0]>self.data.shape[1]
#             print(f'subsampling {self.subsample}')
#             self.train_data=self.data[::self.subsample,:]
#             self.validate_data=self.data[1::self.subsample,:]
#             self.test_data=self.data[2::self.subsample,:]
#             print('done! self.train_data,validate_data and test_data ready')

#         else:
#             if self.combineSigma is None:
#                 self.data, self.fgmeansdata=self.better_subsample(self.fgsmooth.copy(), self.subsampleSigma, self.galcut, 
#                                                                   vector=self.fgsmooth_cut.mean(axis=1), 
#                                                                   gainFluctuationLevel=self.gainFluctuationLevel,
#                                                                  gFdebug=self.gFdebug)
#                 assert self.data.shape[0]>self.data.shape[1]
#                 print(f'done with smart subsampling')
#                 ndata,nfreq=self.data.shape
#                 ntrain=int(0.8*ndata)
#                 self.train_data=self.data[:ntrain,:].copy()
#                 self.validate_data=self.data[ntrain:,:].copy()
#                 print('done! better self.train_data,validate_data and test_data ready')
#             else:
#                 print('doing new combineSigma...')
#                 # self.combine_fgsmooth=np.zeros()
#                 # for icS,cS in enumerate(self.combineSigma):
#                 #     self.combine_fgsmooth
#                 self.fgsmooth2=self.smooth(self.fg_noisy.copy(),self.combineSigma,self.chromatic)
#                 self.fgsmooth_combineSigma=np.vstack([self.fgsmooth.copy(),self.fgsmooth2.copy()])
#                 self.fgsmoothcut_combineSigma=self.galaxy_cut(self.fgsmooth_combineSigma.copy(),self.galcut)
#                 self.data, self.fgmeansdata=self.better_subsample(self.fgsmooth_combineSigma.copy(), self.subsampleSigma, self.galcut,
#                                                                   vector=self.fgsmoothcut_combineSigma.mean(axis=1),
#                                                                   gainFluctuationLevel=self.gainFluctuationLevel,
#                                                                  gFdebug=self.gFdebug)
                
#                 assert self.data.shape[0]>self.data.shape[1]
#                 print(f'better subsampling')
#                 ndata,nfreq=self.data.shape
#                 ntrain=int(0.8*ndata)
#                 self.train_data=self.data[:ntrain,:].copy()
#                 self.validate_data=self.data[ntrain:,:].copy()
#                 print('done! better self.train_data,validate_data and test_data ready')
                
                
                
#     def generate_noise(self,fg,noise_K,subsample,subsampleSigma=None,seed=0):
#         nfreq,ndata=fg.shape
#         np.random.seed(seed)
#         if subsample is not None:
#             npix=ndata/subsample
#         else:
#             assert subsampleSigma is not None
#             nside=self.sigma2nside(subsampleSigma)
#             npix=hp.nside2npix(nside)

#         noise_sigma=noise_K*np.sqrt(npix)
#         noise=np.random.normal(0,noise_sigma,(nfreq,ndata))
#         return noise
    
#     def smooth(self,fg,sigma,chromatic):
#         print(f'smoothing with {sigma} deg, and chromatic {chromatic}')
#         fgsmooth=np.zeros_like(fg)
#         nfreq,ndata=fg.shape
#         for f in range(nfreq):
#             sigma_f=sigma*(10.0/(f+1)) if chromatic else sigma
#             fgsmooth[f,:]=hp.sphtfunc.smoothing(fg[f,:],sigma=np.deg2rad(sigma_f))
#         return fgsmooth

#     def galaxy_cut(self,fg,b_min):
#         print(f'doing galcut for b_min={b_min} deg')
#         _,npix=fg.shape
#         nside=np.sqrt(npix/12).astype(int)
#         col_min = np.pi/2 - np.deg2rad(b_min)
#         col_max = np.pi/2 + np.deg2rad(b_min)
#         mask_pix = hp.query_strip(nside, col_min, col_max, inclusive=False)
#         cutfg = np.delete(fg, mask_pix, axis=1)
#         return cutfg

#     def get_data_PCA(self, data, vector=None):
#         """
#         data should be of shape (nfreqs,ndata)
#         vector should be of shape (nfreqs)
#         out is of shape (ndata,nfreqs) - because torch needs it that way
#         """
#         print('doing PCA')
#         assert data.shape[0]<data.shape[1]
#         # if vector is not None: assert vector.shape==data.shape[0]
#         cov=np.cov(data)
#         eva,eve=np.linalg.eig(cov)
#         s=np.argsort(np.abs(eva))[::-1] 
#         eva,eve=np.real(eva[s]),np.real(eve[:,s])
#         proj_data=eve.T@(data-data.mean(axis=1)[:,None]) #subtract mean and project
#         rms=np.sqrt(proj_data.var(axis=1))
#         out=(proj_data/rms[:,None]).T #divide by rms
#         out=np.random.permutation(out)
#         if vector is None: 
#             return out
#         else:
#             return out, (eve.T@vector.copy())/rms

#     def get_data_noPCA(self, data, vector=None):
#         print('not doing PCA')
#         assert data.shape[0]<data.shape[1]
#         rms=np.sqrt(data.var(axis=1))
#         out=((data - data.mean(axis=1)[:,None])/rms[:,None]).T
#         out=np.random.permutation(out)
#         if vector is None:
#             return out
#         else:
#             return out, vector.copy()/rms

#     def better_subsample(self, fgsmooth, subsampleSigma, galcut, gFdebug, vector=None, gainFluctuationLevel=0.0):
#         print(f'doing smart sampling for sigma={subsampleSigma} using nside={self.sigma2nside(subsampleSigma)}')
        
#         fgsmooth_gainF=self.includeGainFluctuations(fgsmooth.copy(), gainFluctuationLevel,gFdebug)
#         #subsample fgsmooth
#         nside=self.sigma2nside(subsampleSigma)
#         ipix=np.arange(hp.nside2npix(nside))
#         theta,phi=hp.pix2ang(nside,ipix)
#         ipix_128=hp.ang2pix(128,theta,phi)
#         gal=hp.query_strip(nside,np.pi/2-np.deg2rad(galcut),np.pi/2+np.deg2rad(galcut),
#                    inclusive=False)
#         galpix_128=hp.ang2pix(*(128,*hp.pix2ang(nside,gal)))
#         maskpix=np.setdiff1d(ipix_128,galpix_128)
#         print(fgsmooth_gainF.shape)
#         fg_subsample=fgsmooth_gainF[:,maskpix].copy()
#         self.fg_bettersample=fg_subsample.copy()
#         print('shape after subsampling:', fg_subsample.shape)
        
#         #galcut and do PCA
#         fgsmooth_cut=self.galaxy_cut(fgsmooth_gainF,galcut)
#         cov=np.cov(fgsmooth_cut)
#         eva,eve=np.linalg.eig(cov)
#         s=np.argsort(np.abs(eva))[::-1] 
#         eva,eve=np.real(eva[s]),np.real(eve[:,s])
        
#         #project subsampled fg
#         proj_fg=eve.T@(fg_subsample-fg_subsample.mean(axis=1)[:,None])
#         rms=np.sqrt(proj_fg.var(axis=1))
#         print('calculated eva, eve, rms')
#         self.eva=eva.copy()
#         self.eve=eve.copy()
#         self.rms=rms.copy()
#         out=(proj_fg/rms[:,None]).T #divide by rms, transpose for pytorch
#         out=np.random.permutation(out) #shuffles only along first index
#         # print(f'{fgsmooth.shape=}, {fg_subsample.shape=}, {proj_fg.shape=}, {out.shape=}')
#         if vector is not None:
#             return out, (eve.T@vector)/rms
#         else:
#             return out

#     def includeGainFluctuations(self, fgsmoothcut, gainFluctuationLevel, gFdebug):
#         print(f'including gain fluctuations of level: {gainFluctuationLevel}')
#         _,ndata=fgsmoothcut.shape
#         freqs=np.arange(1,51)
#         self.gainF=np.random.normal(0,gainFluctuationLevel if gainFluctuationLevel is not None else 0.0,ndata)
#         template=lusee.mono_sky_models.T_DarkAges_Scaled(freqs,nu_rms=14,nu_min=16.4,A=0.04)
#         fgmeans=fgsmoothcut.mean(axis=1)
#         #does not work for combineSigma
#         print('try fix with .copy()')
#         if gFdebug==0: return fgsmoothcut
#         if gFdebug==1: return fgsmoothcut + fgsmoothcut*self.gainF
#         if gFdebug==2: return fgsmoothcut + np.outer(template,self.gainF)
#         if gFdebug==3: return fgsmoothcut + fgsmoothcut*self.gainF + np.outer(template,self.gainF)
        
#         raise NotImplementedError
        
#     def sigma2nside(self, sigma):
#         #done by hand
#         sigma2nside_map={0.5:64, 1.0:32, 2.0:16, 4.0:8, 8.0:4}
#         try:
#             nside=sigma2nside_map[sigma]
#         except KeyError:
#             print('sigmas must be either 0.5, 1.0, 2.0, 4.0, or 8.0')
#             raise NotImplementedError
#         return sigma2nside_map[sigma]
    
#     def proj_t21(self,t21_vs,include_noise):
#         if include_noise: t21_vs+=self.noise.mean(axis=1)[:,None]
#         proj_t21=(self.eve.T@t21_vs)/self.rms[:,None]
#         return proj_t21
    
#     def set_t21(self, t21, include_noise, overwrite):
#         if include_noise: 
#                 t21+=self.noise.mean(axis=1)
#                 print(f'including noise {self.noise_K} in t21')
        
#         if self.combineSigma is not None:
#             print(f'combining sigma {self.sigma} with combineSigma {self.combineSigma}')
#             combine_t21=np.hstack([t21,t21])
#             if overwrite: self.t21=combine_t21.copy()
#             print('combine_t21.shape=',combine_t21.shape, 'fg_smooth_cut_combine.shape=',self.fgsmoothcut_combineSigma.shape)
#             if self.subsample is not None:
#                 if self.noPCA:
#                     _,t21data=self.get_data_noPCA(self.fgsmoothcut_combineSigma.copy(), combine_t21)
#                 else:
#                     _,t21data=self.get_data_PCA(self.fgsmoothcut_combineSigma.copy(), combine_t21)
#             else:
#                 _,t21data=self.better_subsample(self.fgsmooth_combineSigma.copy(), self.subsampleSigma, self.galcut, self.gFdebug, vector=combine_t21)
#         else:
#             if overwrite: self.t21=t21.copy()
#             if self.subsample is not None:
#                 if self.noPCA:
#                     _,t21data=self.get_data_noPCA(self.fgsmooth_cut.copy(), t21)
#                 else:
#                     _,t21data=self.get_data_PCA(self.fgsmooth_cut.copy(), t21)
#             else:
#                 t21data=self.eve.T@t21/self.rms
#                 # _,t21data=self.better_subsample(self.fgsmooth.copy(), self.subsampleSigma, self.galcut, vector=t21, gainFluctuationLevel=self.gainFluctuationLevel,gFdebug=self.gFdebug)
        
#         if overwrite:
#             self.t21data=t21data.copy()
#             print('done! self.t21data ready')
#         else:
#             print('done! t21data returned')
#             return t21data
>>>>>>> Stashed changes
