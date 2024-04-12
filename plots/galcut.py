import numpy as np
import NormalizingFlow as nf
import matplotlib.pyplot as plt

args=nf.Args()
args.fgFITS, args.freqs, args.chromatic = 'ulsa.fits', '1 51', False
# args.fgFITS, args.freqs, args.chromatic ='gsm16.fits', '51 101', False
args.noiseSeed=0

fig,ax=plt.subplots(1,3,figsize=(15,6))
for ics,cs in enumerate( ['','4','4 6'] ):
    for gc in [0.1,10.0,20.0,30.0,40.0]:
        args.SNRpp=1e12
        args.combineSigma=cs
        args.galcut=gc
        s,ll=nf.get_samplesAndLikelihood(args,plot='A')
        ax[ics].plot(s[:,0],nf.exp(ll),label=f'{gc:.1f} deg')
        # ax[ics].set_xscale('log')
        ax[ics].set_xlim(0.01,2)
        ax[ics].set_title(f'Combine Sigma {cs}')
        ax[ics].set_xlabel('Amplitude [mK]')
        ax[ics].set_ylabel('Likelihood')
        ax[ics].legend()
plt.show()
