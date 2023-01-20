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
parser.add_argument('subsample_factor', type=int)
parser.add_argument('--chromatic', action='store_true')
parser.add_argument('--galcut', type=float, required= False)
parser.add_argument('--noPCA', action='store_true')
parser.add_argument('--append')
parser.add_argument('--combineSigma', type=float, required=False)
parser.add_argument('--noise', type=float, default=0.0, required=False)
parser.add_argument('--noiseSeed', type=int, default=0, required=False)
parser.add_argument('--noisyT21', action='store_true')
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
t21=10*lusee.mono_sky_models.T_DarkAges_Scaled(freqs,nu_rms=14,A=0.04)
flow.set_t21(t21, include_noise=args.noisyT21)

a=np.linspace(9.0,11.0,num=100)
l=[flow._likelihood(flow.fgmeansdata+(10-aa)*flow.t21data).cpu().numpy() for aa in a]
l=np.array(l).flatten()

fname+=f'_noisyT21{args.noisyT21}'
print(f'saving likelihood results to {root}likelihood/{fname}')
np.savetxt(f'{root}likelihood/{fname}',np.vstack([a,l]))
