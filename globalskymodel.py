import numpy as np
import healpy as hp

from pygdsm import GlobalSkyModel16, GlobalSkyModel, HaslamSkyModel, LowFrequencySkyModel

nfreqs=101
nside_out=128
freqs=np.linspace(10,110,num=nfreqs)

print('doing lfsm')
dmap=np.zeros((nfreqs,hp.nside2npix(nside_out)))
gsm=LowFrequencySkyModel(data)
for fidx,f in enumerate(freqs):
    gsm.generate(f)
    print('generated for freq: ',f)
    dmap[fidx,:]=hp.ud_grade(gsm.generated_map_data,nside_out=nside_out)

print(dmap.shape)
print('writing map')
np.savetxt('lfsm.csv',dmap,header=' '.join([str(f) for f in freqs])) 

print('doing gsm')
dmap=np.zeros((nfreqs,hp.nside2npix(nside_out)))
gsm=GlobalSkyModel(data)
for fidx,f in enumerate(freqs):
    gsm.generate(f)
    print('generated for freq: ',f)
    dmap[fidx,:]=hp.ud_grade(gsm.generated_map_data,nside_out=nside_out)

print(dmap.shape)
print('writing map')
np.savetxt('gsm.csv',dmap,header=' '.join([str(f) for f in freqs])) 

print('doing gsm16')
dmap=np.zeros((nfreqs,hp.nside2npix(nside_out)))
gsm=GlobalSkyModel16(data)
for fidx,f in enumerate(freqs):
    gsm.generate(f)
    print('generated for freq: ',f)
    dmap[fidx,:]=hp.ud_grade(gsm.generated_map_data,nside_out=nside_out)

print(dmap.shape)
print('writing map')
np.savetxt('gsm16.csv',dmap,header=' '.join([str(f) for f in freqs])) 