# %%
import numpy as np
import torch
import fitsio
import NormalizingFlow as nf
import lusee
import argparse
import os
import corner
import matplotlib.pyplot as plt
# %%
#muust provide --noisyT21 and --diffCombineSigma!!!

parser = argparse.ArgumentParser(description="Normalizing Flow for ULSA/GSM maps")
parser.add_argument('--fgFITS', type=str, required=False, default='ulsa.fits')
parser.add_argument('--sigma', type=float, default=2.0, required=False)
parser.add_argument('--subsample_factor', type=int, default=None, required=False)
parser.add_argument('--chromatic', action='store_true')
parser.add_argument('--galcut', type=float, default=20.0, required=False)
parser.add_argument('--noPCA', action='store_true')
parser.add_argument('--freqs', type=str, default='1 51', required=True) #e.g. '1 51' separated by space

parser.add_argument('--combineSigma', type=str, required=False, default='') #e.g. '4 6' separated by space

parser.add_argument('--SNRpp', type=float, default=None, required=False)
parser.add_argument('--noise', type=float, default=0.0, required=False)
parser.add_argument('--noiseSeed', type=int, default=0, required=False)
parser.add_argument('--torchSeed', type=int, default=0, required=False)
parser.add_argument('--subsampleSigma', type=float, default=2.0, required=False)

parser.add_argument('--noisyT21', action='store_true')
parser.add_argument('--gainFluctuationLevel', type=float, default=0.0, required=False)
parser.add_argument('--gFdebug', type=int, default=0, required=False)
parser.add_argument('--append', type=str, default='_SVD')

parser.add_argument('--DA_factor', type=float, required=False, default=1.0)
parser.add_argument('--plot', type=str, default='all', required=False) # 1dAmplitude 1dWidth 1dNuMin WvA NvA WvN
parser.add_argument('--freqFluctuationLevel',type=float, required=False, default=0.0)
parser.add_argument('--nPCA', type=str, default='') #e.g 'nmin nmax' separated by space

parser.add_argument('--diffCombineSigma',action='store_true') #bool, whether to do fg_cS[sigma]-fg_cS[sigma-1]
parser.add_argument('--avgAdjacentFreqBins',action='store_true') #bool, whether to average adjacent freq bins

parser.add_argument('--retrain', action='store_true')
parser.add_argument('--appendLik', type=str, default='', required=False)
args=parser.parse_args()

#must have --noisyT21 and --diffCombineSigma!!!
args.noisyT21=True
args.diffCombineSigma=True

refargs=nf.Args()

#interesting hack to print bold text in terminal
BOLD = '\033[1m'
RED='\033[91m'
END='\033[0m'
for arg in vars(args):
    if getattr(refargs,arg)!=getattr(args,arg): print("==>",BOLD,RED,arg,getattr(args,arg),END)
    else: print(arg, getattr(args, arg))

# args=nf.Args()
# args.combineSigma='4'
# args.SNRpp=1e24

###-------------------------------------------------------------------------------------------------###

# %%

#set seed
print(f'setting noise seed {args.noiseSeed} and torch seed {args.torchSeed}')
np.random.seed(args.noiseSeed)
torch.manual_seed(args.torchSeed)
torch.cuda.manual_seed_all(args.torchSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda')
torch.set_default_tensor_type('torch.cuda.FloatTensor')


fname=nf.get_fname(args)
print(f'loading flow from {fname}')
flow=nf.FlowAnalyzerV2(nocuda=False,loadPath=fname)
flow.set_fg(args)

if args.fgFITS=='ulsa.fits': 
    print('using DA model')
    t21=lusee.MonoSkyModels.T_DarkAges_Scaled(flow.freqs,nu_rms=14,nu_min=16.4,A=0.04)
    cosmicdawn=False
elif args.fgFITS=='gsm16.fits':
    print('using CD model')
    t21=lusee.MonoSkyModels.T_CosmicDawn_Scaled(flow.freqs,nu_rms=20,nu_min=67.5,A=0.130)
    cosmicdawn=True
flow.set_t21(t21, include_noise=args.noisyT21)
if args.retrain: flow.train(flow.train_data, flow.validate_data, nocuda=False, savePath=fname,retrain=True)
# %%
class RandomWalker:
    def __init__(self,walkerparams,nsteps,stepsizefactor=0.1):
        self.nsteps=nsteps
        self.walkerparams=walkerparams
        self.steps=self.doSteps(stepsizefactor)
    def doSteps(self,stepsizefactor):
        self.steps=np.zeros((self.nsteps,3))
        self.steps[0,:]=[self.walkerparams['astart'],self.walkerparams['wstart'],self.walkerparams['nstart']]
        for i in range(1,self.nsteps):
            stepsize=np.array([self.walkerparams['amax']-self.walkerparams['amin'],self.walkerparams['wmax']-self.walkerparams['wmin'],self.walkerparams['nmax']-self.walkerparams['nmin']])
            self.steps[i,:]=self.steps[i-1,:]+np.random.normal(0,stepsizefactor,3)*stepsize
            for ivs,vs in enumerate(['A','W','N']):
                if self.steps[i,ivs]<0.01: self.steps[i,ivs]=0.01
        return self.steps

class Walkers:
    def __init__(self,args,nwalkers,nsteps):
        self.args=args
        self.nwalkers=nwalkers
        self.nsteps=nsteps

    def setInitialKWargs(self):
        # start with 1D amplitude plot
        print('setting initial kwargs...')
        self.kwargs={'amin':0.001,'amax':100.0,'wmin':1.0,'wmax':50.0,'nmin':1.0,'nmax':50.0, 'logspace':True}
        if cosmicdawn: kwargs['nmin']=50.0; kwargs['nmax']=100.0
        print(self.kwargs)

    def runInitial1DLikelihoods(self,flow,npoints1,cmb=False):
        truths=[1.0,20.0,67.5] if cosmicdawn else [1.0,14.0,16.4]

        # samples is a (3*npoints)x3 array of samples where each row is a sample of (amp,width,numin), and the first npoints rows are for A, the next npoints rows are for W, and the last npoints rows are for N
        samples=np.tile(truths,(3*npoints1,1))
        print('generating initial samples')
        for ivs,vs in enumerate(['A','W','N']):
            param=np.logspace(np.log10(self.kwargs[vs.lower()+'min']),np.log10(self.kwargs[vs.lower()+'max']),npoints1)
            samples[ivs*npoints1:(ivs+1)*npoints1,ivs]=param

        t21vs=np.zeros((len(flow.freqs),3*npoints1))
        pt21vs=np.zeros_like(t21vs)
        print('getting t21')
        for ii,s in enumerate(samples):
            a,w,n=s
            t21vs[:,ii]=nf.T_DA(flow.freqs,a,w,n,cmb=False,cosmicdawn=cosmicdawn)
        print('projecting t21')
        pt21vs=flow.proj_t21(t21vs,include_noise=self.args.noisyT21)
        print('getting likelihood')

        loglikelihood=flow.get_likelihood(pt21vs,args.freqFluctuationLevel,args.DA_factor, debugfF=False)
        print('initial loglikelihoods ready')
        return samples,loglikelihood
    
    def extractWalkerStart(self,s,ll,npoints1):
        limits=[0.001,0.5,0.9999]
        results={'astart':0,'amin':0,'amax':0,
                'wstart':0,'wmin':0,'wmax':0,
                'nstart':0,'nmin':0,'nmax':0}
        print('getting initial walker start')
        for ivs,vs in enumerate( ['A','W','N'] ):
            samples=s[ivs*npoints1:(ivs+1)*npoints1,ivs]
            likelihood=nf.exp(ll[ivs*npoints1:(ivs+1)*npoints1])
            quantiles=corner.core.quantile(samples,limits,weights=likelihood)
            maxlike=samples[np.argmax(likelihood)]
            print(vs,quantiles,maxlike)
            results[vs.lower()+'start']= maxlike
            results[vs.lower()+'min']=max(maxlike - (quantiles[2]-quantiles[0]),0.01)
            results[vs.lower()+'max']=quantiles[1] + (quantiles[2]-quantiles[1])
            self.initWalkerParams=results
        return self.initWalkerParams

    def walkWalkers(self,walkerparams):
        print('starting random walk for', self.nwalkers, 'walkers')
        self.initWalkerSteps=[[1.,20.,67.5]] if cosmicdawn else [[1.,14.,16.4]]
        for i in range(self.nwalkers):
            walker=RandomWalker(walkerparams,self.nsteps)
            self.initWalkerSteps=np.vstack((self.initWalkerSteps,walker.steps))
        return self.initWalkerSteps
    
    def getWalkerLogLikelihood(self,args,flow,walker_s,cmb=False):
        print('getting likelihood')
        t21vs=np.zeros((len(flow.freqs),len(walker_s)))
        pt21vs=np.zeros_like(t21vs)
        cosmicdawn=True if args.fgFITS=='gsm16.fits' else False
        for ii,s in enumerate(walker_s):
            a,w,n=s
            t21vs[:,ii]=nf.T_DA(flow.freqs,a,w,n,cmb=cmb,cosmicdawn=cosmicdawn) #TODO: cosmic dawn and cmb are hardcoded!
        pt21vs=flow.proj_t21(t21vs,include_noise=args.noisyT21)
        loglikelihood=flow.get_likelihood(pt21vs,args.freqFluctuationLevel,args.DA_factor, debugfF=False)
        return walker_s,loglikelihood
    
    def extractBetterWalkerStart(self,walker_s,walker_ll):
        bestwalker=walker_s[np.argmax(walker_ll)]
        print('best walker start:',bestwalker)
        results={'astart':bestwalker[0],'amin':self.initWalkerParams['amin'],'amax':self.initWalkerParams['amax'],
                'wstart':bestwalker[1],'wmin':self.initWalkerParams['wmin'],'wmax':self.initWalkerParams['wmax'],
                'nstart':bestwalker[2],'nmin':self.initWalkerParams['nmin'],'nmax':self.initWalkerParams['nmax']}
        self.betterWalkerParams=results
        return self.betterWalkerParams
    
    def rewalkWalkers(self,betterWalkerParams,nsteps2):
        print('rewalking walkers with nsteps2',nsteps2)
        self.newWalkerSteps=np.zeros((nsteps2*self.nwalkers,3))
        for i in range(self.nwalkers):
            walker=RandomWalker(betterWalkerParams,nsteps2,stepsizefactor=0.01)
            self.newWalkerSteps=np.vstack((self.newWalkerSteps,walker.steps))
        return self.newWalkerSteps
    
    def getAllWalkersAndLLikelihoods(self,s,s2,s3,ll,ll2,ll3):
        self.allWalkerSteps = np.vstack((s,s2,s3))
        self.allLogLikelihoods = np.hstack((ll,ll2,ll3))
        return self.allWalkerSteps,self.allLogLikelihoods

# %%
    
npoints1=100
nwalkers=100
nsteps=10
nsteps2=100
walkers=Walkers(args,nwalkers,nsteps)
walkers.setInitialKWargs()
s,ll=walkers.runInitial1DLikelihoods(flow,npoints1,cmb=False)
walkerparams=walkers.extractWalkerStart(s,ll,npoints1)
wsteps=walkers.walkWalkers(walkerparams)
s2,ll2=walkers.getWalkerLogLikelihood(args,flow,wsteps,cmb=False)
betterwalkerparams=walkers.extractBetterWalkerStart(s2,ll2)
wsteps2=walkers.rewalkWalkers(betterwalkerparams,nsteps2)
s3,ll3=walkers.getWalkerLogLikelihood(args,flow,wsteps2,cmb=False)
samples,loglikelihoods=walkers.getAllWalkersAndLLikelihoods(s,s2,s3,ll,ll2,ll3)
print('done')

# %%

def plot3x1D(s,ll,npoints1):
    limits=[0.001,0.5,0.9999]
    fig,ax=plt.subplots(1,3,figsize=(10,4))
    for ivs,vs in enumerate( ['A','W','N'] ):
        samples=s[ivs*npoints1:(ivs+1)*npoints1,0]
        likelihood=nf.exp(ll[ivs*npoints1:(ivs+1)*npoints1])
        ax[ivs].plot(samples,likelihood)
        ax[ivs].set_xscale('log')
        quantiles=corner.core.quantile(samples,limits,weights=likelihood)
        for q in quantiles:
            ax[ivs].axvline(q,c='k',alpha=0.5,lw=0.5)
        ax[ivs].set_title(f'{vs} \n [{quantiles[0]:.2f},{quantiles[1]:.2f},{quantiles[-1]:.2f}]')
    return fig,ax

            
def plot3D(s,ll,**kwargs):
    return corner.corner(s,weights=nf.exp(ll),
                labels=['A',r'$\nu_{\rm rms}$',r'$\nu_{\rm min}$'],
                show_titles=True,
                hist_kwargs={'density':True},
                levels=[1-np.exp(-0.5),1-np.exp(-2)],**kwargs)
# %%
    
plot3D(samples,loglikelihoods,range=[(0.5,1.5),(13,15),(15,18)],plot_datapoints=True,bins=100,truths=[1.0,14.0,16.4])
plt.show()

lname=nf.get_lname(args,plot='all')
print(f'saving corner likelihood results to {lname}')
np.savetxt(lname,np.column_stack([samples,loglikelihoods]),header='amp,width,numin,loglikelihood')

# %%
