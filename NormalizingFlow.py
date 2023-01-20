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
    def train(self,train_data,validate_data,nocuda,savePath=None):
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
        if savePath is not None: 
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
    def plotHistograms(self,data,modeidxToPlot,other=None,means=None, t21=None,logscale=False,niterationSteps=1):
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
            plt.savefig(f"{i}_"+"_".join([str(m) for m in modeidxToPlot]))
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
    def set_fg(self,fg, sigma=None, one_over_f=False,subsample=1):
        if sigma is None:
            self.fg=fg
        else:
            self.fg=smooth(fg,sigma,one_over_f)
        self.nfreqs,self.ndata=self.fg.shape
        self.proj_fg,self.proj_fgmeans=get_projected_data(self.fg,self.fg.mean(axis=1))
        self.proj_fg=self.proj_fg[::subsample]
    def set_t21(self,t21):
        _,self.proj_t21=get_projected_data(self.fg,t21) 
        self.ref10x=self.proj_fgmeans+10*self.proj_t21
        self.ref1x=self.proj_fgmeans+self.proj_t21
    
        
def load_ulsa_t21_tcmb_freqs():
    root='/home/rugved/Files/LuSEE/luseepy/Drive/'
    fg=fitsio.read(root+'Simulations/SkyModels/ULSA_32_ddi_smooth.fits')
    freqs=np.arange(1,51)
    t21=lusee.mono_sky_models.T_DarkAges(freqs)
    tcmb=np.ones_like(t21)*2.73
    return fg, t21, tcmb, freqs

def smooth(fg,sigma,one_over_f):
    fgsmooth=np.zeros_like(fg)
    nfreq,_=fg.shape
    for f in range(nfreq):
        sigma_f=sigma*(10.0/(f+1)) if one_over_f else sigma
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


