# %%
import numpy as np
import NormalizingFlow as nf
import matplotlib.pyplot as plt
import lusee

# %%
args=nf.Args()
args.SNRpp=1e24
args.fgFITS='gsm16.fits'
args.freqs='51 101'
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
