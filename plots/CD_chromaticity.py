import NormalizingFlow as nf
import corner
import matplotlib.pyplot as plt
import numpy as np


args=nf.Args()
args.fgFITS, args.freqs ='ulsa.fits', '1 51'
# args.fgFITS, args.freqs ='gsm16.fits', '51 101'
args.SNRpp=1e9
args.combineSigma='4 6'
args.appendLik='_cubegame'

colors={'True':'darkgray','False':'C3'}
truths=[40,14,16.4] if args.fgFITS=='ulsa.fits' else [130,20,67.5]
labels={'True':'Chromatic','False':'Achromatic'}
binsize=50
# ranges=[(1,80),(10,30),(10,30)] if args.fgFITS=='ulsa.fits' else [(10,1000),(10,50),(50,80)]
ranges=[(1,400),(10,30),(10,30)] if args.fgFITS=='ulsa.fits' else [(100,200),(15,30),(60,70)]
fg='Dark Ages' if args.fgFITS=='ulsa.fits' else 'Cosmic Dawn'

args.chromatic='False'
s,ll=nf.get_samplesAndLikelihood(args,plot='all')
s[:,0]*=truths[0]
fig=corner.corner(s,weights=nf.exp(ll),
              labels=[r'$A$',r'$\nu_{\rm rms}$',r'$\nu_{\rm min}$'],
              plot_datapoints=True,levels=(1-np.exp(-0.5),1-np.exp(-2)),
              bins=binsize,hist_kwargs={'density':True},
              color=colors[args.chromatic],range=ranges)
args.chromatic='True'
s,ll=nf.get_samplesAndLikelihood(args,plot='all')
s[:,0]*=truths[0]
corner.corner(s,weights=nf.exp(ll),
              plot_datapoints=True,levels=(1-np.exp(-0.5),1-np.exp(-2)),
              bins=binsize,hist_kwargs={'density':True},
              color=colors[args.chromatic],
              fig=fig,range=ranges)
plt.plot([],[],c='k',ls='--',label='Truth')
corner.overplot_lines(fig,truths,color='k',ls='--',lw=1)
for cs in ['True','False']:
    plt.plot([],[],c=colors[cs],label=labels[cs],lw=5)
plt.legend(bbox_to_anchor=(0,2.5),loc='center left',borderaxespad=0)
plt.suptitle(f'{fg} Signal Constraints for SNR{args.SNRpp:.0e}')
plt.show()
