#30 Aug 2022, Rugved Pund

import torch
from sinf import GIS
import matplotlib.pyplot as plt
import numpy as np
import lusee
import matplotlib
import healpy as hp
import fitsio

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
    def train(self,train_data,validate_data,nocuda,savePath):
        """
        data must be of shape (nsamples, ndim)
        """
        self.nsamples,self.ndim=train_data.shape
        print('now training model with nsamples', self.nsamples,' of ndim',self.ndim)
        self.train_data=self._toTensor(train_data)
        self.validate_data=self._toTensor(validate_data)
        self.model=GIS.GIS(self.train_data.clone(),self.validate_data.clone(),nocuda=nocuda)
        self.nlayer=len(self.model.layer)
        print('Total number of iterations: ', len(self.model.layer))
        torch.save(self.model,savePath)
        print('model saved at', savePath)
    def _toTensor(self,nparray):
        return torch.from_numpy(nparray).float().to(self.device)
    def plotDataToGaussian(self,data,label=None, niterationSteps=8):
        data=self._toTensor(data)
        dn=(self.nlayer-1)//niterationSteps
        plt.figure(figsize=(15,12))
        for ii,i in enumerate(range(0,self.nlayer,dn)):
            plt.subplot(niterationSteps//4+1,4,ii+1)
            data_after=np.array(self.model.forward(data, end=i)[0])
            plt.title(f'after {i} iterations')
            plt.hist2d(data_after[:,0],data_after[:,1],bins=50)
        plt.suptitle(('Data' if label is None else label)+' to Gaussian Flow')
        plt.show()
    def plotGaussianToData(self,gaussdata, label=None, niterationSteps=8):
        gaussdata=self._toTensor(gaussdata)
        dn=(self.nlayer-1)//niterationSteps
        plt.figure(figsize=(15,12))
        for ii,i in enumerate(range(0,self.nlayer,dn)):
            plt.subplot(niterationSteps//4+1,4,ii+1)
            gaussdata_after=np.array(self.model.inverse(gaussdata, start=i)[0]) #weird convention
            plt.title(f'after {i} iterations')
            plt.hist2d(gaussdata_after[:,0],gaussdata_after[:,1],bins=50)
        plt.suptitle('Gaussian to ' + ('Data' if label is None else label))
        plt.show()
    def plotLikelihood(self, testgrid,logscale=False, niterationSteps=8):
        testgrid=self._toTensor(testgrid)
        dn=(self.nlayer-1)//niterationSteps
        plt.figure(figsize=(15,12))
        for ii,i in enumerate(range(0,self.nlayer,dn)):
            plt.subplot(niterationSteps//4+1,4,ii+1)
            logp=np.array(self.model.evaluate_density(testgrid, end=i)).reshape(100,100)
            plt.title(f'after {i} iterations')
            plt.imshow(np.exp(logp),extent=[-3,2,-3,2],
                      norm=matplotlib.colors.LogNorm() if logscale else None)
        plt.colorbar()
        plt.suptitle('Likelihood')
        plt.show()
    def plotHistograms(self,data,modeidxToPlot,other=None,means=None, t21=None,logscale=False,niterationSteps=1, fname=None):
        self._precomputeIterations(data,niterationSteps)
        data=self._toTensor(data)
        other=self._toTensor(other) if other is not None else None
        means=self._toTensor(means) if means is not None else None
        t21=self._toTensor(t21) if t21 is not None else None
        
        dn=(self.nlayer-1)//niterationSteps
        kwargs={'bins':50,'density':True,'norm':matplotlib.colors.LogNorm() if logscale else None}
        for ii,i in enumerate(range(0,self.nlayer,dn)):
            plt.figure(figsize=(15,15))
            data_after=self.precompute_data_after.get(i)
            other_after=(self.model.forward(other,end=i)[0]).numpy() if other is not None else None
            means_after=(self.model.forward(means,end=i)[0]).cpu().numpy() if means is not None else None
            t21_after=(self.model.forward(t21,end=i)[0]).cpu().numpy() if t21 is not None else None
            for jj,j in enumerate(modeidxToPlot):
                for kk,k in enumerate(modeidxToPlot):
                    plt.subplot(len(modeidxToPlot),len(modeidxToPlot),len(modeidxToPlot)*jj+kk+1)
                    if j==k: 
                        plt.hist(data_after[:,k])
                        if means is not None: plt.axvline(means_after[:,k],c='r')
                        if t21 is not None: plt.axvline(t21_after[:,k],c='g')
                    else:
                        plt.hist2d(data_after[:,k],data_after[:,j],**kwargs,cmap='Blues')
                    if other is not None: plt.hist2d(other_after[:,k],other_after[:,j],**kwargs,
                                                    cmap='cividis')
                    if means is not None and j!=k: 
                        plt.scatter(means_after[:,k],means_after[:,j],marker='o',c='r',s=12**2)
                    if t21 is not None and j!=k: 
                        plt.scatter(t21_after[:,k],t21_after[:,j],marker='D',c='green',s=12**2)
                    plt.xlabel(f'mode {k+1}')
                    plt.ylabel(f'mode {j+1}')
            # plt.suptitle(f'iteration {i}')
            plt.tight_layout()
            plt.suptitle(fname)
            plt.savefig(f"{fname}_{i}_"+"_".join([str(m) for m in modeidxToPlot])+'.png')
            plt.show()
    def plotCovMat(self,data,logscale=False, niterationSteps=1):
        self._precomputeIterations(data,niterationSteps)
        data=self._toTensor(data)
        dn=(self.nlayer-1)//niterationSteps
        plt.figure(figsize=(15,15))
        for ii,i in enumerate(range(0,self.nlayer,dn)):
            plt.subplot(niterationSteps//3+1,3,ii+1)
            data_after=self.precompute_data_after.get(i)
            plt.imshow(np.cov(data_after.T),cmap='RdBu',
                       norm=matplotlib.colors.SymLogNorm(linthresh=1e-3, vmax=1,vmin=-1) if logscale else None)
            plt.title(f'after {i} iterations')
            plt.colorbar(shrink=0.5)
        plt.show()
    def _precomputeIterations(self,data,niterationSteps=1):
        data=self._toTensor(data)
        assert type(data)==type(self._toTensor(np.array([]))) #why?
        dn=(self.nlayer-1)//niterationSteps
        for ii,i in enumerate(range(0,self.nlayer,dn)):
            check=self.precompute_data_after.get(i,None)
            if check is None:
                print(f'precomputing iteration {i} of', list(range(0,self.nlayer,dn)))        
                data_after,_=self.model.forward(data,end=i)
                self.precompute_data_after[i]=data_after.detach().cpu().numpy()
    def _distanceFromIdentity(self,cov):
        #norm(A_ij)=sqrt(sum(A_ij**2))
        return np.linalg.norm(cov-np.eye(*cov.shape))
    def plotDistanceFromIdentity(self,data,niterationSteps=4):
        self._precomputeIterations(data,niterationSteps)
        data=self._toTensor(data)
        dn=(self.nlayer-1)//niterationSteps
        distFromId=[]
        for ii,i in enumerate(range(0,self.nlayer,dn)):
            data_after=self.precompute_data_after.get(i)
            distFromId.append(self._distanceFromIdentity(np.cov(data_after.T)))
        plt.plot(range(0,self.nlayer,dn),distFromId,'o-')
        plt.title('Frobenius Distance of Covmat from Identity')
        plt.xlabel('iteration')
        plt.show()
    def _minmaxfinder(self,data):
        nsamples,ndim=data.shape
        self.minmax=np.zeros((ndim,2))
        self.minmax[:,0]=data.min(axis=0)
        self.minmax[:,1]=data.max(axis=0)
    def _likelihood(self,data):
        data=self._toTensor(data)
        return self.model.evaluate_density(data)
        
class FlowAnalyzer(NormalizingFlow):
    def __init__(self, nocuda, loadPath):
        super().__init__(nocuda=nocuda,loadPath=loadPath)
    def precomputeIterations(self,data,niterationSteps):
        self._precomputeIterations(data,niterationSteps)
    def set_fg(self, fg, sigma, chromatic, galcut, noPCA, subsample, noise_K, noiseSeed, combineSigma, subsampleSigma):
        self.fg=fg
        self.sigma=sigma
        self.chromatic=chromatic
        self.galcut=galcut
        self.noPCA=noPCA
        self.subsample=subsample
        self.noise_K=noise_K
        self.noiseSeed=noiseSeed
        self.combineSigma=combineSigma
        self.subsampleSigma=subsampleSigma
        
        self.noise=self.generate_noise(self.fg,self.noise_K,self.subsample,self.sigma,self.noiseSeed)
        self.fg_noisy=self.fg+self.noise
        self.fgsmooth=self.smooth(self.fg_noisy,self.sigma,self.chromatic)
        
        if self.galcut is not None:
            self.fgsmooth_cut=self.galaxy_cut(self.fgsmooth,self.galcut)
        else:
            self.fgsmooth_cut=self.fgsmooth
            
        if self.subsample is not None:
            if self.noPCA: 
                self.data,self.fgmeansdata=self.get_data_noPCA(self.fgsmooth_cut, vector=self.fgsmooth_cut.mean(axis=1))
            else:
                self.data,self.fgmeansdata=self.get_data_PCA(self.fgsmooth_cut, vector=self.fgsmooth_cut.mean(axis=1))

            if self.combineSigma is not None:
                self.fgsmooth2=self.smooth(self.fg,self.combineSigma,self.chromatic)
                if self.galcut is not None:
                    self.fgsmooth_cut2=self.galaxy_cut(self.fgsmooth2,self.galcut)
                else:
                    self.fgsmooth_cut2=self.fgsmooth2
                self.fgsmoothcut_combineSigma=np.vstack([self.fgsmooth_cut,self.fgsmooth_cut2])
                if self.noPCA: 
                    self.data,self.fgmeansdata=self.get_data_noPCA(self.fgsmoothcut_combineSigma, vector=self.fgsmoothcut_combineSigma.mean(axis=1))
                else:
                    self.data,self.fgmeansdata=self.get_data_PCA(self.fgsmoothcut_combineSigma, vector=self.fgsmoothcut_combineSigma.mean(axis=1))
            assert self.data.shape[0]>self.data.shape[1]
            print(f'subsampling {self.subsample}')
            self.train_data=self.data[::self.subsample,:]
            self.validate_data=self.data[1::self.subsample,:]
            self.test_data=self.data[2::self.subsample,:]
            print('done! self.train_data,validate_data and test_data ready')

        else:
            if self.combineSigma is None:
                self.data, self.fgmeansdata=self.better_subsample(self.fgsmooth, self.subsampleSigma, self.galcut,
                                                                 vector=self.fgsmooth_cut.mean(axis=1))
                assert self.data.shape[0]>self.data.shape[1]
                print(f'better subsampling')
                ndata,nfreq=self.data.shape
                ntrain=int(0.8*ndata)
                self.train_data=self.data[:ntrain,:]
                self.validate_data=self.data[ntrain:,:]
                print('done! better self.train_data,validate_data and test_data ready')
            else:
                print('doing combineSigma...')
                self.fgsmooth2=self.smooth(self.fg_noisy,self.combineSigma,self.chromatic)
                self.fgsmooth_combineSigma=np.vstack([self.fgsmooth,self.fgsmooth2])
                self.fgsmoothcut_combineSigma=self.galaxy_cut(self.fgsmooth_combineSigma,self.galcut)
                self.data, self.fgmeansdata=self.better_subsample(self.fgsmooth_combineSigma, self.subsampleSigma, self.galcut,
                                                                 vector=self.fgsmoothcut_combineSigma.mean(axis=1))
                
                assert self.data.shape[0]>self.data.shape[1]
                print(f'better subsampling')
                ndata,nfreq=self.data.shape
                ntrain=int(0.8*ndata)
                self.train_data=self.data[:ntrain,:]
                self.validate_data=self.data[ntrain:,:]
                print('done! better self.train_data,validate_data and test_data ready')
                
                
                
    def generate_noise(self,fg,noise_K,subsample,sigma=None,seed=0):
        nfreq,ndata=fg.shape
        np.random.seed(seed)
        if subsample is not None:
            npix=ndata/subsample
        else:
            assert sigma is not None
            nside=self.sigma2nside(sigma)
            npix=hp.nside2npix(nside)

        noise_sigma=noise_K*np.sqrt(npix)
        noise=np.random.normal(0,noise_sigma,(nfreq,ndata))
        return noise
    
    def smooth(self,fg,sigma,chromatic):
        print(f'smoothing with {sigma} deg, and chromatic {chromatic}')
        fgsmooth=np.zeros_like(fg)
        nfreq,ndata=fg.shape
        for f in range(nfreq):
            sigma_f=sigma*(10.0/(f+1)) if chromatic else sigma
            fgsmooth[f,:]=hp.sphtfunc.smoothing(fg[f,:],sigma=np.deg2rad(sigma_f))
        return fgsmooth

    def galaxy_cut(self,fg,b_min):
        print(f'doing galcut for b_min={b_min} deg')
        _,npix=fg.shape
        nside=np.sqrt(npix/12).astype(int)
        col_min = np.pi/2 - np.deg2rad(b_min)
        col_max = np.pi/2 + np.deg2rad(b_min)
        mask_pix = hp.query_strip(nside, col_min, col_max, inclusive=False)
        cutfg = np.delete(fg, mask_pix, axis=1)
        return cutfg

    def get_data_PCA(self, data, vector=None):
        """
        data should be of shape (nfreqs,ndata)
        vector should be of shape (nfreqs)
        out is of shape (ndata,nfreqs) - because torch needs it that way
        """
        print('doing PCA')
        assert data.shape[0]<data.shape[1]
        # if vector is not None: assert vector.shape==data.shape[0]
        cov=np.cov(data)
        eva,eve=np.linalg.eig(cov)
        s=np.argsort(np.abs(eva))[::-1] 
        eva,eve=np.real(eva[s]),np.real(eve[:,s])
        proj_data=eve.T@(data-data.mean(axis=1)[:,None]) #subtract mean and project
        rms=np.sqrt(proj_data.var(axis=1))
        out=(proj_data/rms[:,None]).T #divide by rms
        out=np.random.permutation(out)
        if vector is None: 
            return out
        else:
            return out, (eve.T@vector.copy())/rms

    def get_data_noPCA(self, data, vector=None):
        print('not doing PCA')
        assert data.shape[0]<data.shape[1]
        rms=np.sqrt(data.var(axis=1))
        out=((data - data.mean(axis=1)[:,None])/rms[:,None]).T
        out=np.random.permutation(out)
        if vector is None:
            return out
        else:
            return out, vector.copy()/rms

    def better_subsample(self, fgsmooth, subsampleSigma, galcut, vector=None):
        print(f'doing smart sampling for sigma={subsampleSigma} using nside={self.sigma2nside(subsampleSigma)}')

        #subsample fgsmooth
        nside=self.sigma2nside(subsampleSigma)
        ipix=np.arange(hp.nside2npix(nside))
        theta,phi=hp.pix2ang(nside,ipix)
        ipix_128=hp.ang2pix(128,theta,phi)
        gal=hp.query_strip(nside,np.pi/2-np.deg2rad(galcut),np.pi/2+np.deg2rad(galcut),
                   inclusive=False)
        galpix_128=hp.ang2pix(*(128,*hp.pix2ang(nside,gal)))
        maskpix=np.setdiff1d(ipix_128,galpix_128)
        print(fgsmooth.shape)
        fg_subsample=fgsmooth[:,maskpix].copy()
        self.fg_bettersample=fg_subsample.copy()
        print('shape after subsampling:', fg_subsample.shape)
        
        #galcut and do PCA
        print('doing internal PCA...')
        fgsmooth_cut=self.galaxy_cut(fgsmooth,galcut)
        cov=np.cov(fgsmooth_cut)
        eva,eve=np.linalg.eig(cov)
        s=np.argsort(np.abs(eva))[::-1] 
        eva,eve=np.real(eva[s]),np.real(eve[:,s])

        #project subsampled fg
        proj_fg=eve.T@(fg_subsample-fg_subsample.mean(axis=1)[:,None])
        rms=np.sqrt(proj_fg.var(axis=1))
        out=(proj_fg/rms[:,None]).T #divide by rms, transpose for pytorch
        out=np.random.permutation(out)
        # print(f'{fgsmooth.shape=}, {fg_subsample.shape=}, {proj_fg.shape=}, {out.shape=}')
        if vector is not None:
            return out, (eve.T@vector)/rms
        else:
            return out


    def sigma2nside(self, sigma):
        #done by hand
        sigma2nside_map={0.5:64, 1.0:32, 2.0:16, 4.0:8, 8.0:4}
        try:
            nside=sigma2nside_map[sigma]
        except KeyError:
            print('sigmas must be either 0.5, 1.0, 2.0, 4.0, or 8.0')
            raise NotImplementedError
        return sigma2nside_map[sigma]

    def set_t21(self, t21, include_noise, overwrite):
        if self.combineSigma is not None:
            print(f'combining sigma {self.sigma} with combineSigma {self.combineSigma}')
            if include_noise: 
                t21+=self.noise.mean(axis=1)
                print('including noise in t21')
            combine_t21=np.hstack([t21,t21])
            if overwrite: self.t21=combine_t21
            print('combine_t21.shape=',combine_t21.shape, 'fg_smooth_cut_combine.shape=',self.fgsmoothcut_combineSigma.shape)
            print(f'projecting t21 according to noPCA={self.noPCA}')
            if self.subsample is not None:
                if self.noPCA:
                    _,t21data=self.get_data_noPCA(self.fgsmoothcut_combineSigma, combine_t21)
                else:
                    _,t21data=self.get_data_PCA(self.fgsmoothcut_combineSigma, combine_t21)
            else:
                _,t21data=self.better_subsample(self.fgsmooth_combineSigma, self.subsampleSigma, self.galcut, combine_t21)
        else:
            if include_noise: 
                t21+=self.noise.mean(axis=1)
                print('including noise in t21')
            if overwrite: self.t21=t21
            print(f'projecting t21 according to noPCA={self.noPCA}')
            if self.subsample is not None:
                if self.noPCA:
                    _,t21data=self.get_data_noPCA(self.fgsmooth_cut, t21)
                else:
                    _,t21data=self.get_data_PCA(self.fgsmooth_cut, t21)
            else:
                _,t21data=self.better_subsample(self.fgsmooth, self.subsampleSigma, self.galcut, vector=t21)
        
        if overwrite:
            self.t21data=t21data
            print('done! self.t21data ready')
        else:
            print('done! t21data returned')
            return t21data
        
def load_ulsa_t21_tcmb_freqs():
    root='/home/rugved/Files/LuSEE/luseepy/Drive/'
    fg=fitsio.read(root+'Simulations/SkyModels/ULSA_32_ddi_smooth.fits')
    freqs=np.arange(1,51)
    t21=lusee.mono_sky_models.T_DarkAges(freqs)
    tcmb=np.ones_like(t21)*2.73
    return fg, t21, tcmb, freqs

def smooth(fg,sigma,chromatic):
    fgsmooth=np.zeros_like(fg)
    nfreq,_=fg.shape
    for f in range(nfreq):
        sigma_f=sigma*(10.0/(f+1)) if chromatic else sigma
        fgsmooth[f,:]=hp.sphtfunc.smoothing(fg[f,:],sigma=np.deg2rad(sigma_f))
    return fgsmooth

def get_projected_data(data, vector=None):
    """
    data should be of shape (nfreqs,ndata)
    vector should be of shape (nfreqs)
    out is of shape (ndata,nfreqs) - because torch needs it that way
    """
    assert data.shape[0]<data.shape[1]
    # if vector is not None: assert vector.shape==data.shape[0]
    cov=np.cov(data)
    eva,eve=np.linalg.eig(cov)
    s=np.argsort(np.abs(eva))[::-1] 
    eva,eve=np.real(eva[s]),np.real(eve[:,s])
    proj_data=eve.T@(data-data.mean(axis=1)[:,None]) #subtract mean and project
    rms=np.sqrt(proj_data.var(axis=1))
    out=(proj_data/rms[:,None]).T #divide by rms
    out=np.random.permutation(out)
    if vector is None: 
        return out
    else:
        return out, (eve.T@vector)/rms

