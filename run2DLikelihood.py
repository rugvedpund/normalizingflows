import matplotlib.pyplot as plt
import numpy as np
import lusee
import matplotlib
import NormalizingFlow as nf
import fitsio
import argparse
import os


parser = argparse.ArgumentParser(description="Normalizing Flow for ULSA maps")
parser.add_argument('sigma', type=float)
parser.add_argument('--subsample_factor', type=int, default=None, required=False)
parser.add_argument('--chromatic', action='store_true')
parser.add_argument('--galcut', type=float, required=False)
parser.add_argument('--noPCA', action='store_true')
parser.add_argument('--append')
parser.add_argument('--combineSigma', type=float, required=False)
parser.add_argument('--noise', type=float, default=0.0, required=False)
parser.add_argument('--noiseSeed', type=int, default=0, required=False)
parser.add_argument('--noisyT21', action='store_true')
parser.add_argument('--vs', type=str, default='Width', required=True)
parser.add_argument('--vsValue', type=float, default=None, required=True)
parser.add_argument('--DA_factor', type=float, required=True)
args=parser.parse_args()

for arg in vars(args):
    print(arg, getattr(args, arg))

root='/home/rugved/Files/LuSEE/ml/'
fg=fitsio.read('/home/rugved/Files/LuSEE/ml/200.fits')


fname=f'GIS_ulsa_nside128_sigma{args.sigma}_subsample{args.subsample_factor}_galcut{args.galcut}_noPCA{args.noPCA}_chromaticBeam{args.chromatic}_combineSigma{args.combineSigma}_noise{args.noise}_seed{args.noiseSeed}'
if args.append: fname+=args.append
print(f'now doing {fname}')

try:
    os.path.exists(root+fname)
except:
    raise

flow=nf.FlowAnalyzer(nocuda=False,loadPath=root+fname)
print('setting fg...')
flow.set_fg(fg=fg,sigma=args.sigma,chromatic=args.chromatic,galcut=args.galcut,noPCA=args.noPCA,subsample=args.subsample_factor,noise_K=args.noise, noiseSeed=args.noiseSeed, combineSigma=args.combineSigma)

print('setting t21...')
freqs=np.arange(1,51)

nu_rms=args.vsValue if args.vs=='Width' else 14
nu_min=args.vsValue if args.vs=='NuMin' else 16.4

t21=lusee.mono_sky_models.T_DarkAges_Scaled(freqs,nu_rms=nu_rms,nu_min=nu_min,A=0.04)
flow.set_t21(t21, include_noise=args.noisyT21,overwrite=True)

l2=np.zeros((20,30))
a=np.linspace(max(0,args.DA_factor-2.0),args.DA_factor+2.0,num=l2.shape[1])


width=np.linspace(nu_rms-6.0,nu_rms+6.0,num=l2.shape[0]) #default is 14
nucenter=np.linspace(5.0,25.0, num=l2.shape[0]) #default is 16.4
vs=width if args.vs=='Width' else nucenter #plot this "vs" amplitude

for iv,v in enumerate(vs):
    for iaa,aa in enumerate(a):
        if args.vs=='Width':
            print('plotting amplitude vs width')
            t21=lusee.mono_sky_models.T_DarkAges_Scaled(freqs,nu_rms=v,A=0.04)
        elif args.vs=='NuMin':
            print('plotting amplitude vs nu_min')
            t21=lusee.mono_sky_models.T_DarkAges_Scaled(freqs,nu_min=v,A=0.04)
        else:
            print('use "Width" or "NuMin"')
        t21data=flow.set_t21(t21,include_noise=args.noisyT21,overwrite=False)
        l2[iv,iaa]=flow._likelihood(flow.fgmeansdata+(args.DA_factor*flow.t21data - aa*t21data)).cpu().numpy()

fname+=f'_noisyT21{args.noisyT21}_vs{args.vs}_vsValue{args.vsValue}_DAfactor{args.DA_factor}'
print(f'saving 2Dlikelihood results to {root}2Dlikelihood/{fname}')
np.savetxt(f'{root}2Dlikelihood/{fname}',l2)
