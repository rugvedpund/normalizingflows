import torch
from sinf import GIS
import matplotlib.pyplot as plt
import numpy as np
import lusee
import matplotlib
import healpy as hp
import fitsio
import os
import corner

root=os.environ['NF_WORKDIR'] #specify path to save/load models and likelihood results

class NormalizingFlow:
    def __init__(self,nocuda,loadPath=''):
        self.device=torch.device('cpu') if nocuda else torch.device('cuda')
        try: 
            self.model=torch.load(loadPath).to(self.device)
            self.nlayer=len(self.model.layer)
            print('model loaded from ',loadPath)
        except FileNotFoundError:
            print('no file found, need to train')
        self.precompute_data_after=dict()
    def train(self,train_data,validate_data,nocuda,savePath,retrain,alpha=None,delta_logp=np.inf,
              verbose=False):
        """
        data must be of shape (nsamples, ndim)
        """
        self.nsamples,self.ndim=train_data.shape
        print('now training model with nsamples', self.nsamples,' of ndim',self.ndim)
        self.train_data=self._toTensor(train_data)
        self.validate_data=self._toTensor(validate_data)
        if retrain: 
            print('retraining...')
            self.model=None
        self.model=GIS.GIS(self.train_data.clone(),self.validate_data.clone(),nocuda=nocuda,
                           alpha=alpha,delta_logp=delta_logp,verbose=verbose)
        self.nlayer=len(self.model.layer)
        print('Total number of iterations: ', len(self.model.layer))
        torch.save(self.model,savePath)
        print('model saved at', savePath)
    def _toTensor(self,nparray):
        return torch.from_numpy(nparray).float().to(self.device)
    def _likelihood(self,data,end=None):
        data=self._toTensor(data)
        return self.model.evaluate_density(data,end=end)

class ForegroundModel():
    def __init__(self,args):
        self.fgFITS=args.fgFITS
        self.sigma=args.sigma
        self.chromatic=args.chromatic
        self.galcut=args.galcut
        self.freqs=args.freqs
        self.combineSigma=args.combineSigma
        self.subsampleSigma=args.subsampleSigma
        self.gainFluctuationLevel=args.gainFluctuationLevel
        self.nPCA=args.nPCA
        self.diffCombineSigma=args.diffCombineSigma
        self.avgAdjacentFreqBins=args.avgAdjacentFreqBins
        self._set_fg() 
    
    def load_fgFITS(self):
        self.fg=fitsio.read(f'{root}{self.fgFITS}')
        self.nfreqsFITS,self.npixFITS=self.fg.shape
        self.nfreqs=len(self.freqs)
        if self.nfreqsFITS!=self.nfreqs:
            print('fg shape and freqs do not match')
            print(f'given fg freqs {self.freqsFITS}, given freqs {self.freqs}')

class SignalModel():
    pass

class 

class FlowAnalyzerV3(NormalizingFlow):
    def __init__()