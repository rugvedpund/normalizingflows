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
parser.add_argument('--plot', type=str, default='Width', required=True) # 1dAmplitude 1dWidth 1dNuMin WvA NvA WvN
parser.add_argument('--DA_factor', type=float, required=True)
parser.add_argument('--subsampleSigma', type=float, required=True)
parser.add_argument('--gainFluctuationLevel', type=float, required=False)
parser.add_argument('--gFdebug', type=int, required=False)
parser.add_argument('--retrain', action='store_true')
parser.add_argument('--freqFluctuationLevel',type=float, required=False)
args=parser.parse_args()

for arg in vars(args):
    print(arg, getattr(args, arg))

root='/home/rugved/Files/LuSEE/ml/'
fg=fitsio.read('/home/rugved/Files/LuSEE/ml/200.fits')

#try loading
fname=f'/home/rugved/Files/LuSEE/ml/GIS_ulsa_nside128_sigma{args.sigma}_subsample{args.subsample_factor}_galcut{args.galcut}_noPCA{args.noPCA}_chromaticBeam{args.chromatic}_combineSigma{args.combineSigma}_noise{args.noise}_seed{args.noiseSeed}_subsampleSigma{args.subsampleSigma}'
if args.gainFluctuationLevel is not None: fname+=f'_gainFluctuation{args.gainFluctuationLevel}_gFdebug{args.gFdebug}'
if args.append: fname+=args.append
flow=nf.FlowAnalyzer(nocuda=False,loadPath=fname)
flow.set_fg(fg=fg,sigma=args.sigma,chromatic=args.chromatic,galcut=args.galcut,noPCA=args.noPCA,
        subsample=args.subsample_factor,noise_K=args.noise, noiseSeed=args.noiseSeed, 
        combineSigma=args.combineSigma, subsampleSigma=args.subsampleSigma, gainFluctuationLevel=args.gainFluctuationLevel, gFdebug=args.gFdebug)
print('setting t21...')
freqs=np.arange(1,51)
t21=lusee.mono_sky_models.T_DarkAges_Scaled(freqs,nu_rms=14,nu_min=16.4,A=0.04)
flow.set_t21(t21, include_noise=args.noisyT21,overwrite=True)
flowt21data_fF=(1+args.freqFluctuationLevel*np.cos(6*np.pi/50*freqs))*flow.t21data

if args.retrain: flow.train(flow.train_data, flow.validate_data, nocuda=False, savePath=fname,retrain=True)

npoints=100
a=np.linspace(max(0,args.DA_factor-2.0),args.DA_factor+2.0,num=npoints)
width=np.linspace(10.0,20.0,num=npoints) #default is 14 MHz
nu_min=np.linspace(10.0,20.0, num=npoints) #default is 16.4 MHz

#prepare t21_vs of shape (nfreq,npoints) for 1d
if args.plot[:2]=='1d':
    x=a
    t21_vs=np.zeros((*flow.t21data.shape,npoints))
    for ixx,xx in enumerate(x):
        t21_vs[:,ixx]=xx*lusee.mono_sky_models.T_DarkAges_Scaled(freqs,nu_rms=14,nu_min=16.4,A=0.04)
        
    t21_vsdata=flow.proj_t21(t21_vs,include_noise=args.noisyT21)
#or prepare t21_vs of shape (nfreq.npoints**2) for 2d 
else:
    x=a if args.plot in ['WvA', 'NvA'] else nu_min
    y=width if args.plot in ['WvA','WvN'] else nu_min #plot this "vs" amplitude
    t21_vs=np.zeros((*flow.t21data.shape,npoints,npoints))
    for iyy,yy in enumerate(y):
        for ixx,xx in enumerate(x):
            if args.plot=='WvA': 
                t21_vs[:,iyy,ixx]=xx*lusee.mono_sky_models.T_DarkAges_Scaled(freqs,nu_rms=yy,nu_min=16.4,A=0.04)
            elif args.plot=='NvA': 
                t21_vs[:,iyy,ixx]=xx*lusee.mono_sky_models.T_DarkAges_Scaled(freqs,nu_rms=14,nu_min=yy,A=0.04)
            elif args.plot=='WvN': 
                t21_vs[:,iyy,ixx]=lusee.mono_sky_models.T_DarkAges_Scaled(freqs,nu_rms=yy,nu_min=xx,A=0.04)
            else: 
                print('args.plot argument invalid')
    t21_vsdata=flow.proj_t21(t21_vs.reshape((*flow.t21data.shape,-1)),include_noise=args.noisyT21)

    
#now calculate likelihood    
l=(flow.fgmeansdata[:,None]+args.DA_factor*flowt21data_fF[:,None]-t21_vsdata).T
likelihood=flow._likelihood(l).cpu().numpy()
if args.plot[:2]!='1d': likelihood=likelihood.reshape(npoints,npoints)
print(likelihood.shape)

#save
lname=f'/home/rugved/Files/LuSEE/ml/newlikelihoods/GIS_ulsa_nside128_sigma{args.sigma}_galcut{args.galcut}_chromaticBeam{args.chromatic}_combineSigma{args.combineSigma}_noise{args.noise}_seed{args.noiseSeed}_subsampleSigma{args.subsampleSigma}'
if args.gainFluctuationLevel is not None: lname+=f'_gainFluctuation{args.gainFluctuationLevel}_gFdebug{args.gFdebug}'
if args.append: lname+=args.append
lname+=f'_noisyT21{args.noisyT21}_vs{args.plot}_DAfactor{args.DA_factor}_freqFluctuationLevel{args.freqFluctuationLevel}'
print(f'saving 2Dlikelihood results to {lname}')
np.savetxt(lname,likelihood)