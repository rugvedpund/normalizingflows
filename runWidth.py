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
parser.add_argument('--galcut', type=float, required= False)
parser.add_argument('--noPCA', action='store_true')
parser.add_argument('--append')
parser.add_argument('--combineSigma', type=float, required=False)
parser.add_argument('--noise', type=float, default=0.0, required=False)
parser.add_argument('--noiseSeed', type=int, default=0, required=False)
parser.add_argument('--noisyT21', action='store_true')
parser.add_argument('--DA_factor', type=float, required=True)
parser.add_argument('--subsampleSigma', type=float, required=True)
parser.add_argument('--vs', type=str, required=True)
parser.add_argument('--gainFluctuationLevel', type=float, required=False)
parser.add_argument('--freqFluctuationLevel',type=float, required=False)

args=parser.parse_args()

for arg in vars(args):
    print(arg, getattr(args, arg))

root='/home/rugved/Files/LuSEE/ml/'
fg=fitsio.read('/home/rugved/Files/LuSEE/ml/200.fits')


fname=f'GIS_ulsa_nside128_sigma{args.sigma}_subsample{args.subsample_factor}_galcut{args.galcut}_noPCA{args.noPCA}_chromaticBeam{args.chromatic}_combineSigma{args.combineSigma}_noise{args.noise}_seed{args.noiseSeed}_subsampleSigma{args.subsampleSigma}'
if args.gainFluctuationLevel is not None: fname+=f'_gainFluctuation{args.gainFluctuationLevel}'
if args.append: fname+=args.append
print(f'now doing {fname}')

flow=nf.FlowAnalyzer(nocuda=False,loadPath=root+fname)
print('setting fg...')
flow.set_fg(fg=fg,sigma=args.sigma,chromatic=args.chromatic,galcut=args.galcut,noPCA=args.noPCA,
            subsample=args.subsample_factor,noise_K=args.noise, noiseSeed=args.noiseSeed, 
            combineSigma=args.combineSigma, subsampleSigma=args.subsampleSigma, gainFluctuationLevel=args.gainFluctuationLevel)
fgshape=flow.fgmeansdata.shape[0]

print('setting t21...')
freqs=np.arange(1,51)

t21=lusee.mono_sky_models.T_DarkAges_Scaled(freqs,nu_rms=14,nu_min=16.4,A=0.04)
flow.set_t21(t21, include_noise=args.noisyT21,overwrite=True)

npoints=100
vs_dict={'1DWidth':np.linspace(10.0,20.0,num=npoints),
         '1DNuMin':np.linspace(10.0,20.0,num=npoints),
         '1DAmplitude':np.linspace(max(0,args.DA_factor-2.0),args.DA_factor+2.0,num=npoints)}

vs=vs_dict[args.vs]
if args.freqFluctuationLevel is None: args.freqFluctuationLevel=0.0

l=np.zeros((npoints,fgshape))
for iv,v in enumerate(vs):
    if args.vs=='1DWidth':
        t21_vs=lusee.mono_sky_models.T_DarkAges_Scaled(freqs,nu_rms=v,A=0.04)
    elif args.vs=='1DNuMin':
        t21_vs=lusee.mono_sky_models.T_DarkAges_Scaled(freqs,nu_min=v,A=0.04)
    elif args.vs=='1DAmplitude':
        t21_vs=v*lusee.mono_sky_models.T_DarkAges_Scaled(freqs,A=0.04)
    else:
        print('use "1DWidth" or "1DNuMin"')
    t21_vsdata=flow.set_t21(t21_vs, include_noise=args.noisyT21, overwrite=False)
    print(f'{t21_vsdata.shape=}, {flow.t21data.shape=}, {flow.fgmeansdata.shape=}')
    l[iv,:]=flow.fgmeansdata+args.DA_factor*(flow.t21data*(1+args.freqFluctuationLevel*np.cos(6*np.pi/50*freqs)) - t21_vsdata)

out=flow._likelihood(l).cpu().numpy()
fname+=f'_noisyT21{args.noisyT21}_DA_factor{args.DA_factor}_subsampleSigma{args.subsampleSigma}_vs{args.vs}_freqFluctuationLevel{args.freqFluctuationLevel}'
print(f'saving likelihood results to {root}likelihood/{fname}')
np.savetxt(f'{root}likelihood/{fname}',np.vstack([vs,out]))
