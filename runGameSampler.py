# Example:
# python runFast.py --noise 0.0 --combineSigma '4 6' --freqs '1 51' --fgFITS 'ulsa.fits'

import gamex
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
# args.combineSigma=''
# args.fgFITS='ulsa.fits'
# args.freqs='1 51'
# args.SNRpp=1e24
# args.torchSeed=0

#------------------------------------------------------------------------------


# set seed
print(f'setting noise seed {args.noiseSeed} and torch seed {args.torchSeed}')
np.random.seed(args.noiseSeed)
torch.manual_seed(args.torchSeed)
torch.cuda.manual_seed_all(args.torchSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda')
torch.set_default_tensor_type('torch.cuda.FloatTensor')


fname=nf.get_fname(args)

breakpoint()
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

# -----------------------------------------------------------------------------
# main sampler block

ga=gamex.Game(flow.get_likelihoodFromSamplesGAME,[1.0,14.0,16.4],[0.7,0.7,0.7])
ga.N1=1000
ga.tweight=1.50
ga.mineffsamp=5000
sname='blah.pdf'
ga.run()

# -----------------------------------------------------------------------------
## now we plot

def plotel(G):
    global fig
    cov=G.cov
    print(G.cov)
    val,vec=np.linalg.eig(cov)
    vec=vec.T

    vec[0]*=np.sqrt(np.real(val[0]))
    vec[1]*=np.sqrt(np.real(val[1]))
    print(vec[0],'A')
    print(vec[1],'B')
    corner.overplot_points(fig,G.mean[None],c='b',marker='o')
    corner.overplot_points(fig,[G.mean-vec[0],
                                G.mean+vec[0]],c='r',marker='',linestyle='-')
    corner.overplot_points(fig,[G.mean-vec[1],
                                G.mean+vec[1]],c='r',marker='',linestyle='-')
    corner.overplot_points(fig,[G.mean-vec[2],
                                G.mean+vec[2]],c='r',marker='',linestyle='-')


xyz=np.array([sa.pars for sa in ga.sample_list])
ww=np.array([sa.we for sa in ga.sample_list]).flatten()

cornerkwargs={'show_titles':True,'levels':[1-np.exp(-0.5),1-np.exp(-2)],'bins':100,
        'range':[(0.8,1.2),(12,16),(15.6,17)],
        'labels':[r'A',r'$\nu_{\rm rms}$',r'$\nu_{\rm min}$'], 'truths':[1.0,14.0,16.4],
        'plot_datapoints':False}

# # just samples
# fig=corner.corner(xyz,**cornerkwargs)
# plt.suptitle('Just Samples')
# plt.show()

# weighted samples, with gaussians
wsumsa=ww/ww.sum()
fig=corner.corner(xyz,weights=wsumsa,**cornerkwargs)
for G in ga.Gausses:
    plotel(G)
plt.suptitle('Weighted Samples')
plt.show()

breakpoint()



