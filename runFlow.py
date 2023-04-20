import time
import torch
from sinf import GIS
import matplotlib.pyplot as plt
import numpy as np
import fitsio
import lusee
import matplotlib
import healpy as hp
import NormalizingFlow as nf
import argparse

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
parser.add_argument('--subsampleSigma', type=float, required=True)
parser.add_argument('--gainFluctuationLevel', type=float, required=False)
parser.add_argument('--gFdebug', type=int, required=False)
args=parser.parse_args()

for arg in vars(args):
    print(arg, getattr(args, arg))

if __name__=='__main__':
    time_start=time.time()

    #load ulsa map
    fg=fitsio.read('/home/rugved/Files/LuSEE/ml/200.fits')
    print(fg.shape)
    
    
    fname=f'/home/rugved/Files/LuSEE/ml/GIS_ulsa_nside128_sigma{args.sigma}_subsample{args.subsample_factor}_galcut{args.galcut}_noPCA{args.noPCA}_chromaticBeam{args.chromatic}_combineSigma{args.combineSigma}_noise{args.noise}_seed{args.noiseSeed}_subsampleSigma{args.subsampleSigma}'
    if args.gainFluctuationLevel is not None: fname+=f'_gainFluctuation{args.gainFluctuationLevel}_gFdebug{args.gFdebug}'
    if args.append: fname+=args.append
    flow=nf.FlowAnalyzer(nocuda=False,loadPath=fname)
    flow.set_fg(fg=fg,sigma=args.sigma,chromatic=args.chromatic,galcut=args.galcut,noPCA=args.noPCA,
            subsample=args.subsample_factor,noise_K=args.noise, noiseSeed=args.noiseSeed, 
            combineSigma=args.combineSigma, subsampleSigma=args.subsampleSigma, gainFluctuationLevel=args.gainFluctuationLevel, gFdebug=args.gFdebug)
    flow.train(flow.train_data, flow.validate_data, nocuda=False, savePath=fname,retrain=True)

    time_end=time.time()-time_start
    print('Time taken to complete:', time_end//60,'mins')

    
