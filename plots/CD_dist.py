import NormalizingFlow as nf
import matplotlib.pyplot as plt
import lusee
import numpy as np

args=nf.Args()
args.SNRpp=1e24
args.fgFITS='gsm16.fits'
args.freqs='51 101'
args.chromatic=False
cd=nf.FlowAnalyzerV2(nf.get_fname(args))
cd.set_fg(args)
cdt21=lusee.MonoSkyModels.T_CosmicDawn_Scaled(cd.freqs,nu_rms=20,nu_min=67.5,A=0.130)
cd.set_t21(cdt21)
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
plt.title('GSM16 Foreground Distribution in PCA Basis - Achromatic Beam')
plt.grid()
plt.savefig('gsm_distribution.pdf',dpi=300,bbox_inches='tight')
plt.show()
