import numpy as np
import fitsio
import NormalizingFlow as nf
import lusee
import argparse
import os


parser = argparse.ArgumentParser(description="Normalizing Flow for ULSA maps")
parser.add_argument('sigma', type=float)
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
parser.add_argument('--plot', type=str, default='Width', required=True) # 1dAmplitude 1dWidth 1dNuMin WvA NvA WvN
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
t21=lusee.MonoSkyModels.T_DarkAges_Scaled(freqs,nu_rms=14,nu_min=16.4,A=0.04)
flow.set_t21(t21, include_noise=args.noisyT21)
if args.retrain: flow.train(flow.train_data, flow.validate_data, nocuda=False, savePath=fname,retrain=True)

# npoints=1000
# # kwargs={'amin':0.0,'amax':150.0,'wmin':1.0,'wmax':40.0,'nmin':1.0,'nmax':40.0}
# # kwargs={'amin':0.0,'amax':3.0,'wmin':10.0,'wmax':20.0,'nmin':10.0,'nmax':20.0}
# # kwargs={'amin':0.5,'amax':1.5,'wmin':11.0,'wmax':17.0,'nmin':15.0,'nmax':18.0}
# kwargs={'amin':0.01,'amax':100.0,'wmin':11.0,'wmax':17.0,'nmin':15.0,'nmax':18.0,
#          'logspace':True}

# #3D corner
# npoints=50
# kwargs={'amin':0.5,'amax':1.5,'wmin':13.5,'wmax':14.5,'nmin':16.0,'nmax':16.8}

# print('getting 3d likelihood...')
# samples,t21_vs=nf.get_t21vs(npoints,**kwargs)
# t21_vsdata=flow.proj_t21(t21_vs,include_noise=True)
# likelihood=flow.get_likelihood(t21_vsdata,args.freqFluctuationLevel,args.DA_factor, debugfF=True) #REMEMBER TO SWITCH debugfF
# #save
# lname=nf.get_lname(args,plot='all')
# print(f'saving corner likelihood results to {lname}')
# np.savetxt(lname,np.column_stack([samples,likelihood]),header='amp,width,numin,loglikelihood')

# #2D corner
# for vs in ['WvA','NvA','WvN']:
#     print(f'getting 2d likelihood for {vs}...')
#     samples2d,t21_vs2d=nf.get_t21vs2d(npoints,vs,**kwargs)
#     t21_vsdata2d=flow.proj_t21(t21_vs2d,include_noise=True)
#     likelihood2d=flow.get_likelihood(t21_vsdata2d,args.freqFluctuationLevel,args.DA_factor, debugfF=True)
#     lname=nf.get_lname(args,plot=vs)
#     print(f'saving 2d likelihood results to {lname}')
#     np.savetxt(lname,np.column_stack([samples2d,likelihood2d]),header='amp,width,numin,loglikelihood')

#1D
npoints=1000
# kwargs={'amin':0.01,'amax':100.0,'wmin':11.0,'wmax':17.0,'nmin':15.0,'nmax':18.0,
        #   'logspace':True}
kwargs={'amin':0.01,'amax':100.0,'wmin':11.0,'wmax':17.0,'nmin':15.0,'nmax':18.0,
         'logspace':True}
for vs in ['A']:
    print(f'getting 1d likelihood for {vs}...')
    samples1d,t21_vs1d=nf.get_t21vs1d(npoints,vs,**kwargs)
    t21vsdata1d=flow.proj_t21(t21_vs1d,include_noise=True)
    likelihood1d=flow.get_likelihood(t21vsdata1d,args.freqFluctuationLevel,args.DA_factor,debugfF=False)
    lname=nf.get_lname(args,plot=vs)
    print(f'saving 1d likelihood results to {lname}')
    np.savetxt(lname,np.column_stack([samples1d,likelihood1d]),header='amp,width,numin,loglikelihood')