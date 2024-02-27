# %%
import numpy as np
import NormalizingFlow as nf
import matplotlib.pyplot as plt
import lusee
import pandas as pd
import corner

# %%

def loadsamplesandlls(fname):
    f=np.loadtxt(fname,unpack=True)
    return f[:-1].T,f[-1]

def plot3D(s,ll,**kwargs):
    fig=corner.corner(s,weights=nf.exp(ll),bins=50,range=[(0.8,1.2),(12,16),(15.6,17)],levels=[1-np.exp(-0.5),1-np.exp(-2)],**kwargs)
    corner.overplot_lines(fig,[1,14,16.4],color='k',ls='--',lw=1)
    return fig
    
def plot3DfromArgs(args,old=False,**kwargs):
    cosmicdawn=args.fgFITS=='ulsa.fits'
    truths=[1,14,16.4] if cosmicdawn else [1,20,67.5]
    s,ll=nf.get_samplesAndLikelihood(args,plot='all',old=old)
    fig=corner.corner(s,weights=nf.exp(ll),
                    labels=[r'$A$',r'$\nu_{\rm rms}$',r'$\nu_{\rm min}$'],hist_kwargs={'density':True},levels=(1-np.exp(-0.5),1-np.exp(-2)),
                    show_titles=True,**kwargs)
    corner.overplot_lines(fig,truths,color='k',ls='--',lw=1)
    return fig

def plot3x1D(args,old=False):
    cosmicdawn=args.fgFITS=='ulsa.fits'
    truths=[1,14,16.4] if cosmicdawn else [1,20,67.5]
    limits=[0.001,0.5,0.9999]
    fig,ax=plt.subplots(1,3,figsize=(10,4))
    for ivs,vs in enumerate( ['A','W','N'] ):
        s,ll=nf.get_samplesAndLikelihood(args,plot=vs,old=old)
        ax[ivs].plot(s[:,0],nf.exp(ll))
        ax[ivs].set_xscale('log')
        ax[ivs].axvline(truths[ivs],c='k',ls='--',label='Truth')
        quantiles=corner.core.quantile(s[:,0],limits,weights=nf.exp(ll))
        for q in quantiles:
            ax[ivs].axvline(q,c='k',alpha=0.5,lw=0.5)
        ax[ivs].set_title(f'{vs} \n [{quantiles[0]:.2f},{quantiles[1]:.2f},{quantiles[-1]:.2f}]')
    return fig,ax

# %%
args=nf.Args()
args.SNRpp=1e24
args.fgFITS='gsm16.fits'
args.freqs='51 101'
args.chromatic=False
cd=nf.FlowAnalyzerV2(nf.get_fname(args))
cd.set_fg(args)
cdt21=lusee.MonoSkyModels.T_CosmicDawn_Scaled(cd.freqs,nu_rms=20,nu_min=67.5,A=0.130)
cd.set_t21(cdt21,include_noise=True)
args.fgFITS='ulsa.fits'
args.freqs='1 51'
da=nf.FlowAnalyzerV2(nf.get_fname(args))
da.set_fg(args)
dat21=lusee.MonoSkyModels.T_DarkAges_Scaled(da.freqs,nu_rms=14,nu_min=16.4,A=0.04)
da.set_t21(dat21,include_noise=True)
# %%
plt.figure(figsize=(5,5))
cfgmean,cfg,ct21='C0','darkcyan','C1'
plt.plot(da.freqs,da.fgss.mean(axis=1),label='ULSA',c=cfgmean)
violinplot=plt.violinplot(da.fgss.T,da.freqs,widths=0.7,showmeans=False,
                          showextrema=False)
for pc in violinplot['bodies']:
    pc.set_facecolor(cfg)
    pc.set_alpha(0.7) 
plt.plot(da.freqs,nf.T_CMB(da.freqs),label='CMB',c='C2')
plt.plot(da.freqs,np.abs( dat21 ),label='Dark Ages',c=ct21)
plt.yscale('log')
plt.ylim(1e-4,1e9)
plt.ylabel('Temperature [K]')
plt.xlabel('Frequency [MHz]')
plt.title('Foregrounds in Real Frequency Basis')
plt.grid()
plt.legend()
# plt.savefig('plots/ulsa_freq.pdf',dpi=300,bbox_inches='tight')


# %%

plt.figure(figsize=(5,5))
cfgmean,cfg,ct21='C0','darkcyan','C1'
plt.plot(cd.freqs,cd.fgss.mean(axis=1),label='GSM16',c=cfgmean)
violinplot=plt.violinplot(cd.fgss.T,cd.freqs,widths=0.7,showmeans=False,
                          showextrema=False)
for pc in violinplot['bodies']:
    pc.set_facecolor(cfg)
    pc.set_alpha(0.7) 
plt.plot(cd.freqs,nf.T_CMB(cd.freqs),label='CMB',c='C2')
plt.plot(cd.freqs,np.abs( cdt21 ),label='Cosmic Dawn',c=ct21)
plt.yscale('log')
plt.ylim(2e-3,3e5)
plt.ylabel('Temperature [K]')
plt.xlabel('Frequency [MHz]')
plt.title('Foregrounds in Real Frequency Basis')
plt.grid()
plt.legend()
# plt.savefig('plots/gsm_freq.pdf',dpi=300,bbox_inches='tight')


# %%

plt.figure(figsize=(5,5))
cfgmean,cfg,ct21,ctmb='C0','C5','C1','C2'
plt.plot(da.freqs,np.abs(da.eve.T@da.fgss.mean(axis=1)),label='Foreground Mean',c=cfgmean)
plt.plot(da.freqs,np.abs(da.eve.T@da.t21),label='Dark Ages',c=ct21)
plt.plot(da.freqs,np.abs(da.eve.T@nf.T_CMB(da.freqs)),label='CMB',c=ctmb)
plt.plot(da.freqs,da.rms,label='Eigenmode RMS',c=cfg)
plt.yscale('log')
plt.legend()
plt.title('Decomposition to PCA Eigenmodes')
plt.xlabel('Eigenmode Number')
plt.ylabel('Temperature [K]')
plt.grid()
# plt.savefig('plots/ulsa_eigenmodes.pdf',dpi=300,bbox_inches='tight')

# %%

plt.figure(figsize=(5,5))
cfgmean,cfg,ct21,ctmb='C0','C5','C1','C2'
plt.plot(np.arange(1,51),np.abs(cd.eve.T@cd.fgss.mean(axis=1)),label='Foreground Mean',c=cfgmean)
plt.plot(np.arange(1,51),np.abs(cd.eve.T@cd.t21),label='Cosmic Dawn',c=ct21)
plt.plot(np.arange(1,51),np.abs(cd.eve.T@nf.T_CMB(cd.freqs)),label='CMB',c=ctmb)
plt.plot(np.arange(1,51),cd.rms,label='Eigenmode RMS',c=cfg)
plt.yscale('log')
plt.legend()
plt.title('Decomposition to PCA Eigenmodes')
plt.xlabel('Eigenmode Number')
plt.ylabel('Temperature [K]')
plt.grid()
# plt.savefig('plots/gsm_eigenmodes.pdf',dpi=300,bbox_inches='tight')

# %%

cfgmean,cfg,ct21,ctmb='C0','C0','C1','C2'
plt.figure(figsize=(10,3))
negatives=np.where(da.t21data<0)
t21data=da.t21data.copy()
t21data[negatives]*=-1
plt.plot(da.freqs,t21data,c=ct21,label='Dark Ages')
fgmeansdata=da.fgmeansdata.copy()
fgmeansdata[negatives]*=-1
plt.plot(da.freqs,fgmeansdata,c=cfgmean,label='Foreground Mean')
cmbdata=(da.eve.T@nf.T_CMB(da.freqs))/da.rms/10
cmbdata[negatives]*=-1
plt.plot(da.freqs,cmbdata,c=ctmb,label='CMB/10.0')
# pfgdata=da.pfg/da.rms[:,None]
# print(pfgdata.shape)
violinplot=plt.violinplot(da.data,da.freqs,showextrema=False,showmedians=True,
               widths=0.7,bw_method=0.05)
for pc in violinplot['bodies']:
    pc.set_facecolor(cfg)
    pc.set_alpha(0.7)
plt.legend()
plt.ylim(-20,20)
plt.ylabel('Fluctuation Amplitude')
plt.xlabel('Eigenmode Number')
plt.title('ULSA Foreground Distribution in PCA Basis - Achromatic Beam')
plt.grid()
# plt.savefig('plots/ulsa_distribution.pdf',dpi=300,bbox_inches='tight')

# %%

cfgmean,cfg,ct21,ctmb='C0','C0','C1','C2'
plt.figure(figsize=(10,3))
negatives=np.where(cd.t21data<0)
t21data=cd.t21data.copy()
t21data[negatives]*=-1
plt.plot(np.arange(1,51),t21data,c=ct21,label='Cosmic Dawn')
fgmeansdata=cd.fgmeansdata.copy()
fgmeansdata[negatives]*=-1
plt.plot(np.arange(1,51),fgmeansdata,c=cfgmean,label='Foreground Mean')
cmbdata=(cd.eve.T@nf.T_CMB(cd.freqs))/cd.rms/10
cmbdata[negatives]*=-1
plt.plot(np.arange(1,51),cmbdata,c=ctmb,label='CMB/10.0')
violinplot=plt.violinplot(cd.data,np.arange(1,51),showextrema=False,
                          showmedians=True,widths=0.7,bw_method=0.05)
for pc in violinplot['bodies']:
    pc.set_facecolor(cfg)
    pc.set_alpha(0.7)
plt.legend()
# plt.ylim(-60,60)
plt.ylabel('Fluctuation Amplitude')
plt.xlabel('Eigenmode Number')
plt.title('Foreground Distribution in PCA Basis - Achromatic Beam')
plt.grid()
# plt.savefig('plots/gsm_distribution.pdf',dpi=300,bbox_inches='tight')


# %%

args=nf.Args()
args.fgFITS, args.freqs, args.chromatic ='gsm16.fits', '51 101', True
# args.fgFITS, args.freqs, args.chromatic ='ulsa.fits', '1 51', True

# snrpps=[1e4,1e5,1e6]
snrpps=[1e5,1e6,1e7]
# snrpps=[1e7,1e8,1e9]
# snrpps=[1e8,1e9,1e10]
# snrpps=[1e9,1e10,1e11]

seeds=[0,1,2,3,4,5,6,7,8,9]
combineSigmas=['','4','4 6']
colors={'':'dimgray','4':'C2','4 6':'C3'}
lstyles=[':','-.','-']
labels={'':r'$2^\circ$','4':r'$2^\circ+4^\circ$',
        '4 6':r'$2^\circ+4^\circ+6^\circ$'}
alphas={'':0.05,'4':0.1,'4 6':0.3}
chromatic='Chromatic' if args.chromatic else 'Achromatic'
fg='Dark Ages' if args.fgFITS=='ulsa.fits' else 'Cosmic Dawn'
afactor=40 if args.fgFITS=='ulsa.fits' else 130

rawdata=list()
for snrpp in snrpps:
    for seed in seeds:
        for cs in combineSigmas:
            args.SNRpp=snrpp
            args.noiseSeed=seed
            args.combineSigma=cs
            s,ll=nf.get_samplesAndLikelihood(args,plot='A')
            rawdata.append({'snrpp':snrpp,'seed':seed,'cs':cs,'s':s.reshape(-1),'l':nf.exp( ll )})
df=pd.DataFrame(rawdata).set_index(['snrpp','seed','cs'])
pltdata=pd.DataFrame(columns=['s','med','std','medlow','medhigh'],
                     index=pd.MultiIndex.from_product([snrpps,combineSigmas],
                                                      names=['snrpp','cs']))
#create pltdata with medians and stds
for snrpp in snrpps:
    for cs in combineSigmas:
        pltdata['s'][snrpp,cs]=np.median([df['s'][snrpp,seed,cs] for seed in seeds],axis=0)
        pltdata['med'][snrpp,cs]=np.median([df['l'][snrpp,seed,cs] for seed in seeds],axis=0)
        pltdata['std'][snrpp,cs]=np.std([df['l'][snrpp,seed,cs] for seed in seeds],axis=0)
        pltdata['medlow'][snrpp,cs]=np.clip(pltdata['med'][snrpp,cs]-pltdata['std'][snrpp,cs],0,None)
        pltdata['medhigh'][snrpp,cs]=np.clip(pltdata['med'][snrpp,cs]+pltdata['std'][snrpp,cs],0,None)

#plot
for snrpp in snrpps:
    for cs in combineSigmas:
        plt.plot(afactor*pltdata['s'][snrpp,cs],pltdata['med'][snrpp,cs],c=colors[cs],
                 ls=lstyles[snrpps.index(snrpp)])
        plt.fill_between(afactor*pltdata['s'][snrpp,cs],pltdata['medlow'][snrpp,cs],
                         pltdata['medhigh'][snrpp,cs], color=colors[cs],alpha=alphas[cs])

#clean legend and axes
for snrpp in snrpps:
   plt.plot([],[],c='k',ls=lstyles[snrpps.index(snrpp)],label=rf'SNR={snrpp:.0e}')
for cs in combineSigmas:
    plt.plot([],[],c=colors[cs],label=labels[cs])
plt.axvline(afactor,color='k',ls='--',label='Truth')
plt.ylabel('Likelihood')
plt.xlabel('Amplitude [mK]')
plt.ylim(0,1.15)
plt.xlim(0.5,4e3)
plt.xscale('log')
plt.title(f'{fg} Signal Amplitude Likelihood - {chromatic} Beam')
plt.legend()
# plt.savefig(f'plots/{fg}_{chromatic}_likelihood.pdf',dpi=300,bbox_inches='tight')

# %%

# simple 3D plot

args=nf.Args()
args.fgFITS, args.freqs, args.chromatic ='ulsa.fits', '1 51', 'False'
# args.fgFITS, args.freqs, args.chromatic ='gsm16.fits', '51 101', 'False'
# args.SNRpp=1e24
args.noise, args.append = 0.0, '_whynoradio'
args.combineSigma='4 6'
args.noiseSeed=2

truths=[1,14,16.4] if args.fgFITS=='ulsa.fits' else [1,20,67.5]
binsize=30
fg='Dark Ages' if args.fgFITS=='ulsa.fits' else 'Cosmic Dawn'

# for ivs,vs in enumerate( ['A','W','N'] ):
#     plt.figure(figsize=(10,5))
#     plt.subplot(1,3,ivs+1)
#     s,ll=nf.get_samplesAndLikelihood(args,plot=vs)
#     s[:,0]*=truths[0] if vs=='A' else 1
#     plt.plot(s[:,0],nf.exp(ll))
#     plt.xscale('log')
#     plt.title(f'{fg} Signal Constraints for SNRpp={args.SNRpp:.0e}')
# plt.show()



s,ll=nf.get_samplesAndLikelihood(args,plot='all')
fig=corner.corner(s,weights=nf.exp(ll),
                labels=[r'$A$',r'$\nu_{\rm rms}$',r'$\nu_{\rm min}$'],
                bins=binsize,hist_kwargs={'density':True},
                plot_datapoints=False,levels=(1-np.exp(-0.5),1-np.exp(-2)))
corner.overplot_lines(fig,truths,color='k',ls='--',lw=1)


# %%

# combineSigmas 3D plot

args=nf.Args()
args.fgFITS, args.freqs, args.chromatic ='ulsa.fits', '1 51', False
# args.fgFITS, args.freqs, args.chromatic ='gsm16.fits', '51 101', False
args.SNRpp=1e12
args.appendLik='_walkers'

colors={'':'gray','4':'C2','4 6':'C3'}
truths=[40,14,16.4] if args.fgFITS=='ulsa.fits' else [130,20,67.5]
labels={'':r'$2^\circ$','4':r'$2^\circ+4^\circ$',
        '4 6':r'$2^\circ+4^\circ+6^\circ$'}
binsize=50
ranges=[(30,60),(13.5,16),(15,18)] if args.fgFITS=='ulsa.fits' else [(110,150),(18,22),(65,70)]
fg='Dark Ages' if args.fgFITS=='ulsa.fits' else 'Cosmic Dawn'

args.combineSigma=''
s,ll=nf.get_samplesAndLikelihood(args,plot='all')
s[:,0]*=truths[0]
fig=corner.corner(s,weights=nf.exp(ll),
              labels=[r'$A$',r'$\nu_{\rm rms}$',r'$\nu_{\rm min}$'],
              plot_datapoints=False,levels=(1-np.exp(-0.5),1-np.exp(-2)),
              bins=binsize,hist_kwargs={'density':True},
              color=colors[args.combineSigma],range=ranges)
args.combineSigma='4'
s,ll=nf.get_samplesAndLikelihood(args,plot='all')
s[:,0]*=truths[0]
corner.corner(s,weights=nf.exp(ll),
              plot_datapoints=False,levels=(1-np.exp(-0.5),1-np.exp(-2)),
              bins=binsize,hist_kwargs={'density':True},
              color=colors[args.combineSigma],
              fig=fig,range=ranges)
args.combineSigma='4 6'
s,ll=nf.get_samplesAndLikelihood(args,plot='all')
s[:,0]*=truths[0]
corner.corner(s,weights=nf.exp(ll),
              plot_datapoints=False,levels=(1-np.exp(-0.5),1-np.exp(-2)),
              bins=binsize,hist_kwargs={'density':True},
              color=colors[args.combineSigma],
              fig=fig,range=ranges)
corner.overplot_lines(fig,truths,color='k',ls='--',lw=1)
plt.plot([],[],c='k',ls='--',label='Truth')
for cs in ['','4','4 6']:
    plt.plot([],[],c=colors[cs],label=labels[cs],lw=5)
plt.legend(bbox_to_anchor=(0,2.5),loc='center left',borderaxespad=0)
plt.suptitle(f'{fg} Signal Constraints for SNR={args.SNRpp:.0e}')
# plt.savefig(f'plots/{fg}_combineSigmas.pdf',dpi=300,bbox_inches='tight')


# %%

# chromatic vs achromatic 3D plot

args=nf.Args()
# args.fgFITS, args.freqs ='ulsa.fits', '1 51'
args.fgFITS, args.freqs ='gsm16.fits', '51 101'
args.SNRpp=1e24
args.combineSigma=''
args.appendLik='_walkers'

colors={'True':'darkgray','False':'C3'}
truths=[40,14,16.4] if args.fgFITS=='ulsa.fits' else [130,20,67.5]
labels={'True':'Chromatic','False':'Achromatic'}
binsize=40
ranges=[(1,80),(10,50),(10,50)] if args.fgFITS=='ulsa.fits' else [(10,200),(10,50),(20,100)]
fg='Dark Ages' if args.fgFITS=='ulsa.fits' else 'Cosmic Dawn'

args.chromatic='False'
s,ll=nf.get_samplesAndLikelihood(args,plot='all')
s[:,0]*=truths[0]
fig=corner.corner(s,weights=nf.exp(ll),
              labels=[r'$A$',r'$\nu_{\rm rms}$',r'$\nu_{\rm min}$'],
              plot_datapoints=False,levels=(1-np.exp(-0.5),1-np.exp(-2)),
              bins=binsize,hist_kwargs={'density':True},
              color=colors[args.chromatic],range=ranges)
args.chromatic='False'
s,ll=nf.get_samplesAndLikelihood(args,plot='all')
s[:,0]*=truths[0]
corner.corner(s,weights=nf.exp(ll),
              plot_datapoints=False,levels=(1-np.exp(-0.5),1-np.exp(-2)),
              bins=binsize,hist_kwargs={'density':True},
              color=colors[args.chromatic],
              fig=fig,range=ranges)
plt.plot([],[],c='k',ls='--',label='Truth')
corner.overplot_lines(fig,truths,color='k',ls='--',lw=1)
for cs in ['True','False']:
    plt.plot([],[],c=colors[cs],label=labels[cs],lw=5)
plt.legend(bbox_to_anchor=(0,2.5),loc='center left',borderaxespad=0)
plt.suptitle(f'{fg} Signal Constraints for SNR{args.SNRpp:.0e}')

# %%

args=nf.Args()
# args.fgFITS, args.freqs ='ulsa.fits', '1 51'
args.fgFITS, args.freqs ='gsm16.fits', '51 101'
args.combineSigma=''
args.appendLik='_walkers'
args.noiseSeed=0
args.chromatic=False

truths=[40,14,16.4] if args.fgFITS=='ulsa.fits' else [130,20,67.5]
labels={'amp':r'$A$', 'width':r'$\nu_{\rm rms}$', 'numin':r'$\nu_{\rm min}$'}
fg='Dark Ages' if args.fgFITS=='ulsa.fits' else 'Cosmic Dawn'
chromatic='Chromatic' if args.chromatic=='True' else 'Achromatic'
sigmaLabels={2:'$2^\circ$',4:'$4^\circ$',6:'$6^\circ$'}
combineSigmalabels={'':'$2^\circ$','4':'$2^\circ+4^\circ$','4 6':'$2^\circ+4^\circ+6^\circ$'}
ylims=(2e7,5e14) if args.fgFITS=='ulsa.fits' else (2e5,5e14)

fig,ax=plt.subplots(1,3,figsize=(10,8))
snrpps=[1e4,1e5,1e6]
snrpplabels={1e4:r'$10^4$',1e5:r'$10^5$',1e6:r'$10^6$',1e7:r'$10^7$',
                1e8:r'$10^8$',1e9:r'$10^9$',1e10:r'$10^{10}$',1e11:r'$10^{11}$',
                1e12:r'$10^{12}$',1e13:r'$10^{13}$',1e24:r'$10^{24}$'}
#for noisy
for snrpp in snrpps:
    for ics,cs in enumerate( ['','4','4 6'] ):
        args.SNRpp=snrpp
        args.combineSigma=cs
        try:
            s,ll=nf.get_samplesAndLikelihood(args,plot='all')
        except FileNotFoundError:
            print('not found, continuing')
            continue
        constraints=nf.get_constraints(s,ll)
        constraints['amp']*=truths[0]
        constraints['amp+']*=truths[0]
        constraints['amp-']*=truths[0]
        for i,p in enumerate(['amp','width','numin']):
            ax[i].errorbar(constraints[p],args.SNRpp,
                        xerr=[[constraints[p+'-']],[constraints[p+'+']]],
                        fmt='o',capsize=5,c='C'+str(ics))

#for noiseless
args.SNRpp=1e24
s,ll=nf.get_samplesAndLikelihood(args,plot='all')
constraints=nf.get_constraints(s,ll)
constraints['amp']*=truths[0]
constraints['amp+']*=truths[0]
constraints['amp-']*=truths[0]
for i,p in enumerate(['amp','width','numin']):
    for ics,cs in enumerate( ['','4','4 6'] ):
        ax[i].axhline(snrpps[-1]*1e2,c='k',alpha=0.1)
        ax[i].errorbar(constraints[p],snrpps[-1]*1e2,
                    xerr=[[constraints[p+'-']],[constraints[p+'+']]],
                    fmt='o',capsize=5,c='C'+str(ics))
        ax[i].set_yscale('log')
        ax[i].axvline(truths[i],c='gray',ls='--',label='Truth')
        ax[i].set_xlabel(labels[p])

ax[0].set_yticks(snrpps+[snrpps[-1]*1e2],
                 labels=[snrpplabels[snrpp] for snrpp in snrpps]+['Noiseless'],
                 minor=False)
# for i in range(3): ax[i].set_ylim(ylims)
ax[0].set_ylabel(r'SNR$_{\rm pp}$')
ax[1].get_yaxis().set_visible(False)
ax[2].get_yaxis().set_visible(False)
fig.suptitle(f'{fg} Signal Constraints for {chromatic} Beams')

# %%

# explore galcut

args=nf.Args()
args.fgFITS, args.freqs, args.chromatic = 'ulsa.fits', '1 51', False
# args.fgFITS, args.freqs ='gsm16.fits', '51 101', False

args.SNRpp=1e12
args.combineSigma='4 6'
args.noiseSeed=1
args.galcut=40.0

plot3x1D(args)
print(f'{fg} SNRpp {args.SNRpp:.0e},combineSigma {args.combineSigma},galcut {args.galcut} deg,noiseSeed {args.noiseSeed}')
plot3D(args)

# %%

# explore zeronoise

args=nf.Args()
args.fgFITS, args.freqs, args.chromatic = 'ulsa.fits', '1 51', False
# args.fgFITS, args.freqs ='gsm16.fits', '51 101', False
args.SNRpp,args.append,old = 1e24, '_walkers', False
# args.appendLik='_walkers2'
# args.noise,old = 0.0, False
# args.SNRpp,args.append, old = 1e24, '_oldmodel', False
args.combineSigma='4 6'
# args.appendLik='_tS1'
# args.append='_oldmodel'
# args.noise, args.append, old = 0.0, '_oldcode', True
# args.noise, args.append, old = 0.0, '_oldmodel', False
# args.noise, old = 0.0, False
args.noiseSeed=0
# args.torchSeed=1

# plot3x1D(args,old=False)
fg='Dark Ages' if args.fgFITS=='ulsa.fits' else 'Cosmic Dawn'
noise = f'SNRpp {args.SNRpp:.0e}' if args.SNRpp is not None else f'noise {args.noise:.0e}'
# print(f'{fg} {noise},combineSigma {args.combineSigma},noiseSeed {args.noiseSeed}')
plot3D(args,old=old,bins=40,range=[(0.8,1.2),(12,16),(15.6,17)],plot_datapoints=False)
plt.show()
# %%

# explore galcut

args=nf.Args()
args.fgFITS, args.freqs, args.chromatic = 'ulsa.fits', '1 51', False
# args.fgFITS, args.freqs ='gsm16.fits', '51 101', False
args.noiseSeed=1

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
# %%

# hysteresis check

sbothnew,llbothnew=loadsamplesandlls('tests/both_samplesandlls_newmodel_newcodeN0')
sbothold,llbothold=loadsamplesandlls('tests/both_samplesandlls_oldcode')
sold,llold=loadsamplesandlls('tests/old_samplesandlls')
soldbig,lloldbig=loadsamplesandlls('tests/old_samplesandlls_bigcube')
snew,llnew=loadsamplesandlls('tests/new_samplesandlls_oldcode')
plot3D(sbothnew,llbothnew,plot_datapoints=True);plt.show()
plot3D(sbothold,llbothold,plot_datapoints=True);plt.show()
plot3D(sold,llold,plot_datapoints=True);plt.show()
plot3D(soldbig,lloldbig,plot_datapoints=True);plt.show()
# plot3D(snew,llnew,plot_datapoints=False);plt.show()



# %%

args=nf.Args()
args.fgFITS, args.freqs, args.chromatic = 'ulsa.fits', '1 51', False
# args.appendLik='_bigcube'
args.combineSigma='4 6'
args.noise, args.append, old = 0.00001, '_oldmodel', False
range=[(0.7,1.5),(12.95,15.05),(15.35,17.45)]
# args.noise, old = 0.00001, True
# range=[(0.9,1.1),(13.95,14.05),(16.35,16.45)]
args.noiseSeed=2
plot3DfromArgs(args,old=old,bins=50,range=range,plot_datapoints=False); plt.show()

# %%

# %%
