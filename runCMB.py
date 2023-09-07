import numpy as np
import fitsio
import NormalizingFlow as nf
import lusee
import argparse
import os

parser = argparse.ArgumentParser(description="Normalizing Flow for ULSA maps")
parser.add_argument('--sigma', type=float, default=2.0, required=False)
parser.add_argument('--subsample_factor', type=int, default=None, required=False)
parser.add_argument('--chromatic', action='store_true')
parser.add_argument('--galcut', type=float, default=20.0, required=False)
parser.add_argument('--noPCA', action='store_true')

parser.add_argument('--combineSigma', type=str, required=False, default='') #e.g. '4 6' separated by space

parser.add_argument('--noise', type=float, default=0.0, required=False)
parser.add_argument('--noiseSeed', type=int, default=0, required=False)
parser.add_argument('--subsampleSigma', type=float, default=2.0, required=False)

parser.add_argument('--noisyT21', action='store_true')
parser.add_argument('--gainFluctuationLevel', type=float, default=0.0, required=False)
parser.add_argument('--gFdebug', type=int, default=0, required=False)
parser.add_argument('--append')

parser.add_argument('--DA_factor', type=float, required=False, default=1.0)
parser.add_argument('--plot', type=str, default='all', required=False) # 1dAmplitude 1dWidth 1dNuMin WvA NvA WvN
parser.add_argument('--freqFluctuationLevel',type=float, required=False, default=0.0)
parser.add_argument('--nPCA', type=str, default='') #e.g 'nmin nmax' separated by space

parser.add_argument('--diffCombineSigma',action='store_true') #bool, whether to do fg_cS[sigma]-fg_cS[sigma-1]
parser.add_argument('--avgAdjacentFreqBins',action='store_true') #bool, whether to average adjacent freq bins

parser.add_argument('--retrain', action='store_true')
parser.add_argument('--appendLik', type=str, default='', required=False)
args=parser.parse_args()

for arg in vars(args):
    print(arg, getattr(args, arg))

    
###-------------------------------------------------------------------------------------------------###

root=os.environ['LUSEE_ML']

#load ulsa map
fg=fitsio.read(f'{root}200.fits')

#try loading
fname=nf.get_fname(args)
print(fname)
flow=nf.FlowAnalyzerV2(nocuda=False,loadPath=fname)
flow.set_fg(args)

print('setting t21...')
freqs=np.arange(1,51)
tcmb=100.0*np.ones_like(freqs)
flow.set_t21(tcmb, include_noise=args.noisyT21)
if args.retrain: flow.train(flow.train_data, flow.validate_data, nocuda=False, savePath=fname,retrain=True)

#1D
npoints=1000
kwargs={'amin':0.1,'amax':10000.0,'wmin':11.0,'wmax':17.0,'nmin':15.0,'nmax':18.0,
          'logspace':True}

for vs in ['A']:
    print(f'getting 1d likelihood for {vs}...')
    samples1d,t21_vs1d=nf.get_t21vs1d(npoints,vs,cmb=True,**kwargs)
    t21vsdata1d=flow.proj_t21(t21_vs1d,include_noise=True)
    likelihood1d=flow.get_likelihood(t21vsdata1d,args.freqFluctuationLevel,args.DA_factor,debugfF=False)
    lname=nf.get_lname(args,plot=vs)
    print(f'saving 1d likelihood results to {lname}')
    np.savetxt(lname,np.column_stack([samples1d,likelihood1d]),header='amp,width,numin,loglikelihood')