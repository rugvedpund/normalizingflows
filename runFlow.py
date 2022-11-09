import torch
from sinf import GIS
import matplotlib.pyplot as plt
import numpy as np
import fitsio
import lusee
import matplotlib
import healpy as hp
from lusee.NormalizingFlow import *

fg=fitsio.read('~/LuSEE/ml/200.fits')
print(fg.shape)

def smoothAndTrain(fg,sigma,subsample_factor=100,one_over_f=False, n=None):
    fgsmooth=np.zeros_like(fg)
    nfreq,ndata=fg.shape
    for f in range(nfreq):
        sigma_f=sigma*(10.0/(f+1)) if one_over_f else sigma
        fgsmooth[f,:]=hp.sphtfunc.smoothing(fg[f,:],sigma=np.deg2rad(sigma_f))
    data=get_projected_data(fg)
    train_data=data[::subsample_factor,:]
    validate_data=data[1::subsample_factor,:]
    print(f'nfreq={nfreq},ndata={ndata}')
    print(f'one_over_f={one_over_f}')
    print(f'train_data.shape={train_data.shape}')
    print(f'validate_data.shape={validate_data.shape}')
    
    fname=f'/hpcgpfs01/scratch/rugvedpund/LuSEE/ml/GIS_ulsa_nside128_sigma{sigma}_subsample{subsample_factor}_freqScaling{one_over_f}'
    if n is not None: fname+=f'_{n}'
    nf=NormalizingFlow(nocuda=False,loadPath=fname)
    nf.train(train_data,validate_data,nocuda=False,savePath=fname)
   

smoothAndTrain(fg,sigma=8.0, subsample_factor=10, one_over_f=False)
