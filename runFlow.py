import time
import torch
from sinf import GIS
import matplotlib.pyplot as plt
import numpy as np
import fitsio
import lusee
import matplotlib
import healpy as hp
from NormalizingFlow import *


def sigma2nside(sigma):
    #done by hand
    sigma2nside_map={0.5:64, 1.0:32, 2.0:16, 4.0:8, 8.0:4}
    try:
        nside=sigma2nside_map[sigma]
    except KeyError:
        print('sigmas must be either 0.5, 1.0, 2.0, 4.0, or 8.0')
        raise NotImplementedError
    return sigma2nside_map[sigma]

def generate_noise(fg,noise_K,subsample,sigma=None,seed=0):
    nfreq,ndata=fg.shape
    np.random.seed(seed)
    if subsample is not None:
        npix=ndata/subsample
    else:
        assert sigma is not None
        nside=sigma2nside(sigma)
        npix=hp.nside2npix(nside)

    noise_sigma=noise_K*np.sqrt(npix)
    noise=np.random.normal(0,noise_sigma,(nfreq,ndata))
    return noise
    
def smooth(fg,sigma,one_over_f):
    print(f'smoothing with {sigma} deg, and chromatic {one_over_f}')
    fgsmooth=np.zeros_like(fg)
    nfreq,ndata=fg.shape
    for f in range(nfreq):
        sigma_f=sigma*(10.0/(f+1)) if one_over_f else sigma
        fgsmooth[f,:]=hp.sphtfunc.smoothing(fg[f,:],sigma=np.deg2rad(sigma_f))
    return fgsmooth

def galaxy_cut(fg,b_min):
    print(f'doing galcut for b_min={b_min} deg')
    _,npix=fg.shape
    nside=np.sqrt(npix/12).astype(int)
    col_min = np.pi/2 - np.deg2rad(b_min)
    col_max = np.pi/2 + np.deg2rad(b_min)
    mask_pix = hp.query_strip(nside, col_min, col_max, inclusive=False)
    cutfg = np.delete(fg, mask_pix, axis=1)
    return cutfg

def get_data_PCA(data, vector=None):
    """
    data should be of shape (nfreqs,ndata)
    vector should be of shape (nfreqs)
    out is of shape (ndata,nfreqs) - because torch needs it that way
    """
    print('doing PCA')
    assert data.shape[0]<data.shape[1]
    # if vector is not None: assert vector.shape==data.shape[0]
    cov=np.cov(data)
    eva,eve=np.linalg.eig(cov)
    s=np.argsort(np.abs(eva))[::-1] 
    eva,eve=np.real(eva[s]),np.real(eve[:,s])
    proj_data=eve.T@(data-data.mean(axis=1)[:,None]) #subtract mean and project
    rms=np.sqrt(proj_data.var(axis=1))
    out=(proj_data/rms[:,None]).T #divide by rms, transpose for pytorch
    out=np.random.permutation(out)
    if vector is None: 
        return out
    else:
        return out, (eve.T@vector)/rms

def get_data_noPCA(data):
    print('not doing PCA')
    assert data.shape[0]<data.shape[1]
    rms=np.sqrt(data.var(axis=1))
    out=((data - data.mean(axis=1)[:,None])/rms[:,None]).T
    return np.random.permutation(out)

def better_subsample(fgsmooth, sigma, galcut, vector=None):
    print(f'doing smart sampling for sigma={sigma} using nside={sigma2nside(sigma)}')
    
    #subsample fgsmooth
    nside=sigma2nside(sigma)
    ipix=np.arange(hp.nside2npix(nside))
    theta,phi=hp.pix2ang(nside,ipix)
    ipix_128=hp.ang2pix(128,theta,phi)
    gal=hp.query_strip(nside,np.pi/2-np.deg2rad(galcut),np.pi/2+np.deg2rad(galcut),
               inclusive=False)
    galpix_128=hp.ang2pix(*(128,*hp.pix2ang(nside,gal)))
    maskpix=np.setdiff1d(ipix_128,galpix_128)
    print(fgsmooth.shape)
    fg_subsample=fgsmooth[:,maskpix].copy()
    
    #galcut and do PCA
    print('doing internal PCA...')
    fgsmooth_cut=galaxy_cut(fgsmooth,galcut) 
    cov=np.cov(fgsmooth_cut)
    eva,eve=np.linalg.eig(cov)
    s=np.argsort(np.abs(eva))[::-1] 
    eva,eve=np.real(eva[s]),np.real(eve[:,s])
    
    #project subsampled fg
    proj_fg=eve.T@(fg_subsample-fg_subsample.mean(axis=1)[:,None])
    rms=np.sqrt(proj_fg.var(axis=1))
    out=(proj_fg/rms[:,None]).T #divide by rms, transpose for pytorch
    out=np.random.permutation(out)
    # print(f'{fgsmooth.shape=}, {fg_subsample.shape=}, {proj_fg.shape=}, {out.shape=}')
    if vector is not None:
        return out, (eve.T@vector)/rms
    else:
        return out

def train(data, subsample_factor, sigma, galcut, fname):
    print('training...')
    assert data.shape[0]>data.shape[1]
    if subsample_factor is not None:
        assert subsample_factor>=2
        train_data=data[::subsample_factor,:]
        validate_data=data[1::subsample_factor,:]
    else:
        ndata,nfreq=data.shape
        ntrain=int(0.8*ndata)
        train_data=data[:ntrain,:]
        validate_data=data[ntrain:,:]
        print(f'{data.shape=}, {ntrain=}, {train_data.shape=}, {validate_data.shape=}')
    nf=NormalizingFlow(nocuda=False, loadPath=fname)
    nf.train(train_data, validate_data, nocuda=False, savePath=fname)


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
args=parser.parse_args()

for arg in vars(args):
    print(arg, getattr(args, arg))

if __name__=='__main__':
    time_start=time.time()

    #load ulsa map
    fg=fitsio.read('/home/rugved/Files/LuSEE/ml/200.fits')
    print(fg.shape)
    
    #add noise
    noise=generate_noise(fg,args.noise,subsample=args.subsample_factor, sigma=args.sigma ,seed=args.noiseSeed)
    fg_noisy=fg+noise

    
    #smooth with chromatic/achromatic beam
    fgsmooth=smooth(fg_noisy, sigma=args.sigma, one_over_f=args.chromatic)
    
    if args.galcut is not None:
        fgsmooth_cut=galaxy_cut(fgsmooth,args.galcut)
    else:
        fgsmooth_cut=fgsmooth
    
    if args.subsample_factor is not None:
        #do PCA or not
        if args.noPCA: 
            data=get_data_noPCA(fgsmooth_cut)
        else:
            data=get_data_PCA(fgsmooth_cut)

        #combine with another sigma by concatenating
        if args.combineSigma is not None:
            fgsmooth2=smooth(fg,args.combineSigma,one_over_f=args.chromatic)

            if args.galcut is not None:
                fgsmooth_cut2=galaxy_cut(fgsmooth2,args.galcut)
            else:
                fgsmooth_cut2=fgsmooth2
            print(fgsmooth_cut.shape, fgsmooth_cut2.shape)
            #do PCA or not
            if args.noPCA: 
                data=get_data_noPCA(np.vstack([fgsmooth_cut,fgsmooth_cut2]))
            else:
                data=get_data_PCA(np.vstack([fgsmooth_cut,fgsmooth_cut2]))

            print(f'Also combining with sigma {args.combineSigma} to get data.shape {data.shape}')
    
    else:
        #better_sampling_routine: galcut, PCA and randomize:
        data=better_subsample(fgsmooth,args.sigma,args.galcut)   

    #train
    fname=f'/home/rugved/Files/LuSEE/ml/GIS_ulsa_nside128_sigma{args.sigma}_subsample{args.subsample_factor}_galcut{args.galcut}_noPCA{args.noPCA}_chromaticBeam{args.chromatic}_combineSigma{args.combineSigma}_noise{args.noise}_seed{args.noiseSeed}'
    if args.append: fname+=args.append
    print(fname)
    train(data, args.subsample_factor, args.sigma, args.galcut, fname)
        
    time_end=time.time()-time_start
    print('Time taken to complete:', time_end//60,'mins')
