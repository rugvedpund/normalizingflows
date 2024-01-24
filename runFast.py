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

# parser = argparse.ArgumentParser(description="Normalizing Flow for ULSA/GSM maps")
# parser.add_argument('--fgFITS', type=str, required=False, default='ulsa.fits')
# parser.add_argument('--sigma', type=float, default=2.0, required=False)
# parser.add_argument('--subsample_factor', type=int, default=None, required=False)
# parser.add_argument('--chromatic', action='store_true')
# parser.add_argument('--galcut', type=float, default=20.0, required=False)
# parser.add_argument('--noPCA', action='store_true')
# parser.add_argument('--freqs', type=str, default='1 51', required=True) #e.g. '1 51' separated by space

# parser.add_argument('--combineSigma', type=str, required=False, default='') #e.g. '4 6' separated by space

# parser.add_argument('--SNRpp', type=float, default=None, required=False)
# parser.add_argument('--noise', type=float, default=0.0, required=False)
# parser.add_argument('--noiseSeed', type=int, default=0, required=False)
# parser.add_argument('--torchSeed', type=int, default=0, required=False)
# parser.add_argument('--subsampleSigma', type=float, default=2.0, required=False)

# parser.add_argument('--noisyT21', action='store_true')
# parser.add_argument('--gainFluctuationLevel', type=float, default=0.0, required=False)
# parser.add_argument('--gFdebug', type=int, default=0, required=False)
# parser.add_argument('--append', type=str, default='_SVD')

# parser.add_argument('--DA_factor', type=float, required=False, default=1.0)
# parser.add_argument('--plot', type=str, default='all', required=False) # 1dAmplitude 1dWidth 1dNuMin WvA NvA WvN
# parser.add_argument('--freqFluctuationLevel',type=float, required=False, default=0.0)
# parser.add_argument('--nPCA', type=str, default='') #e.g 'nmin nmax' separated by space

# parser.add_argument('--diffCombineSigma',action='store_true') #bool, whether to do fg_cS[sigma]-fg_cS[sigma-1]
# parser.add_argument('--avgAdjacentFreqBins',action='store_true') #bool, whether to average adjacent freq bins

# parser.add_argument('--retrain', action='store_true')
# parser.add_argument('--appendLik', type=str, default='', required=False)
# args=parser.parse_args()

# #must have --noisyT21 and --diffCombineSigma!!!
# args.noisyT21=True
# args.diffCombineSigma=True

# refargs=nf.Args()

# #interesting hack to print bold text in terminal
# BOLD = '\033[1m'
# RED='\033[91m'
# END='\033[0m'
# for arg in vars(args):
#     if getattr(refargs,arg)!=getattr(args,arg): print("==>",BOLD,RED,arg,getattr(args,arg),END)
#     else: print(arg, getattr(args, arg))

args=nf.Args()
args.combineSigma='4 6'
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

# start with 1D amplitude plot
print('setting initial kwargs...')
kwargs={'amin':0.001,'amax':100.0,'wmin':1.0,'wmax':50.0,'nmin':1.0,'nmax':50.0, 'logspace':True}
if cosmicdawn: kwargs['nmin']=50.0; kwargs['nmax']=100.0
print(kwargs)

#  %%

npoints1=100
truths=[1.0,20.0,67.5] if cosmicdawn else [1.0,14.0,16.4]

# samples is a (3*npoints)x3 array of samples where each row is a sample of (amp,width,numin), and the first npoints rows are for A, the next npoints rows are for W, and the last npoints rows are for N
samples=np.tile(truths,(3*npoints1,1))

for ivs,vs in enumerate(['A','W','N']):
    param=np.logspace(np.log10(kwargs[vs.lower()+'min']),np.log10(kwargs[vs.lower()+'max']),npoints1)
    samples[ivs*npoints1:(ivs+1)*npoints1,ivs]=param

t21vs=np.zeros((len(flow.freqs),3*npoints1))
pt21vs=np.zeros_like(t21vs)
print('getting t21')
for ii,s in enumerate(samples):
    a,w,n=s
    t21vs[:,ii]=nf.T_DA(flow.freqs,a,w,n,cmb=False,cosmicdawn=cosmicdawn)
print('projecting t21')
pt21vs=flow.proj_t21(t21vs,include_noise=args.noisyT21)
print('getting likelihood')

loglikelihood=flow.get_likelihood(pt21vs,args.freqFluctuationLevel,args.DA_factor, debugfF=False)



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
plot3x1D(samples,loglikelihood,npoints1)

# %%

def extract(s,ll,npoints1):
    limits=[0.001,0.5,0.9999]
    results={'astart':0,'amin':0,'amax':0,
             'wstart':0,'wmin':0,'wmax':0,
             'nstart':0,'nmin':0,'nmax':0}
    for ivs,vs in enumerate( ['A','W','N'] ):
        samples=s[ivs*npoints1:(ivs+1)*npoints1,ivs]
        likelihood=nf.exp(ll[ivs*npoints1:(ivs+1)*npoints1])
        quantiles=corner.core.quantile(samples,limits,weights=likelihood)
        maxlike=samples[np.argmax(likelihood)]
        print(vs,quantiles,maxlike)
        results[vs.lower()+'start']= maxlike
        results[vs.lower()+'min']=max(maxlike - (quantiles[2]-quantiles[0]),0.01)
        results[vs.lower()+'max']=quantiles[1] + (quantiles[2]-quantiles[1])
    return results

extract(samples,loglikelihood,npoints1)

# %%
walkerparams=extract(samples,loglikelihood,npoints1)

class RandomWalker:
    def __init__(self,walkerparams,nsteps):
        self.nsteps=nsteps
        self.walkerparams=walkerparams
        self.steps=self.doSteps()
    def doSteps(self):
        self.steps=np.zeros((self.nsteps,3))
        self.steps[0,:]=[self.walkerparams['astart'],self.walkerparams['wstart'],self.walkerparams['nstart']]
        for i in range(1,self.nsteps):
            stepsize=np.array([self.walkerparams['amax']-self.walkerparams['amin'],self.walkerparams['wmax']-self.walkerparams['wmin'],self.walkerparams['nmax']-self.walkerparams['nmin']])
            self.steps[i,:]=self.steps[i-1,:]+np.random.normal(0,1,3)*stepsize
        return self.steps

RandomWalker(walkerparams,nsteps).steps


# %%


nsteps=100
nwalkers=10
for i in range(1,nwalkers):
    walker=RandomWalker(walkerparams,nsteps)
    walker_s=np.vstack((walker_s,walker.steps))
    walker_ll=np.hstack((walker_ll,walker.getLogLikelihood(args,flow)))

def getLogLikelihood(args,flow):
    print('getting likelihood')
    t21vs=np.zeros((len(flow.freqs),self.nsteps))
    pt21vs=np.zeros_like(t21vs)
    for ii,s in enumerate(self.steps):
        a,w,n=s
        t21vs[:,ii]=nf.T_DA(flow.freqs,a,w,n,cmb=False,cosmicdawn=cosmicdawn) #TODO: cosmic dawn and cmb are hardcoded!
    pt21vs=flow.proj_t21(t21vs,include_noise=args.noisyT21)
    loglikelihood=flow.get_likelihood(pt21vs,args.freqFluctuationLevel,args.DA_factor, debugfF=False)
    return loglikelihood


# %%

def plot3D(s,ll):
    corner.corner(s,weights=nf.exp(ll), bins=30,
                  labels=['A',r'$\nu_{\rm rms}$',r'$\nu_{\rm min}$'],
                  show_titles=True, plot_datapoints=False,
                  hist_kwargs={'density':True},
                  levels=[1-np.exp(-0.5),1-np.exp(-2)])

plot3D(walker_s,walker_ll)


# %%




# samples1d,t21_vs1d=nf.get_t21vs1d(flow.freqs,npoints=2000,vs='A',cosmicdawn=cosmicdawn,**kwargs)
# t21_vsdata1d=flow.proj_t21(t21_vs1d,include_noise=True)
# likelihood1d=flow.get_likelihood(t21_vsdata1d, args.freqFluctuationLevel, args.DA_factor)
# lname=nf.get_lname(args,plot='A')
# print(f'saving 1d likelihood to {lname}')
# np.savetxt(lname,np.column_stack([samples1d,likelihood1d]),header='amp,width,numin,loglikelihood')

# limits=[0.001,0.5,0.9999]
# s,ll=nf.get_samplesAndLikelihood(args,plot='A')
# quantiles=corner.core.quantile(s[:,0],limits,weights=nf.exp(ll))
# print('quantiles:',quantiles)

# # #plot amp
# # for x in quantiles:
# #     plt.axvline(x,c='k',alpha=0.5,lw=0.5)
# # plt.axvline(1.0,color='k')
# # plt.plot(s,nf.exp(ll))
# # plt.xscale('log')
# # # plt.xlim(10,2e3)
# # plt.title(f'{quantiles=}')
# # plt.show()

# if quantiles[0]<0.01:
#     print('skipping 3D corner plot')
#     exit()

# truths=[1.0,20.0,67.5] if cosmicdawn else [1.0,14.0,16.4]

# print('getting 1D limits on width and numin...')
# for ivs,vs in enumerate(['W','N']):
#     samples1d,t21_vs1d=nf.get_t21vs1d(flow.freqs,npoints=2000,vs=vs,cosmicdawn=cosmicdawn,**kwargs)
#     t21_vsdata1d=flow.proj_t21(t21_vs1d,include_noise=True)
#     likelihood1d=flow.get_likelihood(t21_vsdata1d, args.freqFluctuationLevel, args.DA_factor)
#     lname=nf.get_lname(args,plot=vs)
#     print(f'saving 1d likelihood to {lname}')
#     np.savetxt(lname,np.column_stack([samples1d,likelihood1d]),header='amp,width,numin,loglikelihood')

# kwargs3D={}
# for ivs,vs in enumerate(['A','W','N']):
#     s,ll=nf.get_samplesAndLikelihood(args,plot=vs)
#     quantiles=corner.core.quantile(s[:,0],limits,weights=nf.exp(ll))
#     print('quantiles:',quantiles)
#     kwargs3D[vs.lower()+'min']=max(truths[ivs] - 2*(quantiles[-1]-truths[ivs]),0.01)
#     kwargs3D[vs.lower()+'max']=truths[ivs] + 2*(quantiles[-1] - truths[ivs])

#     # #plot
#     # for x in quantiles:
#     #     plt.axvline(x,c='k',alpha=0.5,lw=0.5)
#     # plt.plot(s,nf.exp(ll))
#     # plt.xscale('log')
#     # # plt.xlim(10,2e3)
#     # plt.title(f'{quantiles=}')
#     # plt.show()


# # kwargs3D={'amin':0.5,'amax':1.5,'wmin':13.5,'wmax':14.5,'nmin':16.0,'nmax':16.8}
# print(kwargs3D)

# print('getting 3D corner plot...')
# samples,t21_vs=nf.get_t21vs(flow.freqs,npoints=30,cosmicdawn=cosmicdawn,**kwargs3D)
# t21_vsdata=flow.proj_t21(t21_vs,include_noise=True)
# likelihood=flow.get_likelihood(t21_vsdata,args.freqFluctuationLevel,args.DA_factor, debugfF=False)
# lname=nf.get_lname(args,plot='all')
# print(f'saving corner likelihood results to {lname}')
# np.savetxt(lname,np.column_stack([samples,likelihood]),header='amp,width,numin,loglikelihood')

# #plot
# corner.corner(samples,weights=nf.exp(likelihood),bins=30,
#             labels=['Amplitude','Width',r'$\nu_{min}$'], truths=[1.0,20.0,67.5] if cosmicdawn else [1.0,14.0,16.4],
#             verbose=True, plot_datapoints=False, show_titles=True,
#             levels=[1-np.exp(-0.5),1-np.exp(-2)])
# # plt.suptitle(f'{lname.split("/")[-1]}')
# plt.show()
# %%
