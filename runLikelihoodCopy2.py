import matplotlib.pyplot as plt
import numpy as np
import lusee
import matplotlib
import NormalizingFlowCopy2 as nf
import fitsio
import argparse
import os


parser = argparse.ArgumentParser(description="Normalizing Flow for ULSA maps")
parser.add_argument('sigma', type=float)
parser.add_argument('--subsample_factor', type=int, default=None, required=False)
parser.add_argument('--chromatic', action='store_true')
parser.add_argument('--galcut', type=float, required= False)
parser.add_argument('--noPCA', action='store_true')
parser.add_argument('--append')
parser.add_argument('--combineSigma', type=float, required=False)
parser.add_argument('--noise', type=float, default=0.0, required=False)
parser.add_argument('--noiseSeed', type=int, default=0, required=False)
parser.add_argument('--noisyT21', action='store_true')
parser.add_argument('--DA_factor', type=float, required=True)
parser.add_argument('--subsampleSigma', type=float, required=True)

args=parser.parse_args()

for arg in vars(args):
    print(arg, getattr(args, arg))

root='/home/rugved/Files/LuSEE/ml/'
fg=fitsio.read('/home/rugved/Files/LuSEE/ml/200.fits')


fname=f'GIS_ulsa_nside128_sigma{args.sigma}_subsample{args.subsample_factor}_galcut{args.galcut}_noPCA{args.noPCA}_chromaticBeam{args.chromatic}_combineSigma{args.combineSigma}_noise{args.noise}_seed{args.noiseSeed}_subsampleSigma{args.subsampleSigma}'
if args.append: fname+=args.append
print(f'now doing {fname}')

try:
    os.path.exists(root+fname)
except:
    raise

flow=nf.FlowAnalyzer(nocuda=False,loadPath=root+fname)
print('setting fg...')
flow.set_fg(fg=fg,sigma=args.sigma,chromatic=args.chromatic,galcut=args.galcut,noPCA=args.noPCA,
            subsample=args.subsample_factor,noise_K=args.noise, noiseSeed=args.noiseSeed, 
            combineSigma=args.combineSigma, subsampleSigma=args.subsampleSigma)

print('setting t21...')
freqs=np.arange(1,51)

t21=lusee.mono_sky_models.T_DarkAges_Scaled(freqs,nu_rms=14,A=0.04)
flow.set_t21(t21, include_noise=args.noisyT21,overwrite=True)

a=np.linspace(max(0,args.DA_factor-2.0),args.DA_factor+2.0,num=50)

l=[flow._likelihood(flow.fgmeansdata+(args.DA_factor-aa)*flow.t21data).cpu().numpy() for aa in a]
l=np.array(l).flatten()

fname+=f'_noisyT21{args.noisyT21}_DA_factor{args.DA_factor}_subsampleSigma{args.subsampleSigma}'
print(f'saving likelihood results to {root}likelihood/{fname}')
np.savetxt(f'{root}likelihood/{fname}',np.vstack([a,l]))
