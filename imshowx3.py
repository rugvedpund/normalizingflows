import numpy as np
import NormalizingFlow as nf
import matplotlib.pyplot as plt
import parser

##---------------------------------------------------------------------------##
# argparser

argparser = parser.create_parser()
args = argparser.parse_args()

# must set this
args.noisyT21 = True
args.diffCombineSigma = True

parser.prettyprint(args)

##---------------------------------------------------------------------------##
# setup corner plot params

cosmicdawn = True if args.fgFITS == "gsm16.fits" else False
truths = [1, 20, 67.5] if cosmicdawn else [1, 14, 16.4]
fg = "CD" if cosmicdawn else "DA"
labels = {
    "": "$2^\circ$",
    "4": "$2^\circ+4^\circ$",
    "4 6": "$2^\circ+4^\circ+6^\circ$",
}
axlabels = {'a':''r"$A$", 'w': r"$\nu_{\rm rms}$ [MHz]", 'n': r"$\nu_{\rm min}$ [MHz]"}
##---------------------------------------------------------------------------##

s, ll = nf.get_samplesAndLikelihood(args, plot="all")
print("Total number of samples: ", s.shape)
print("Max likelihood:", max(ll))

npoints = [61,60,59]
likelihoods = np.exp(ll - max(ll)).reshape(*npoints)

limits = dict()
mins = np.min(s, axis=0)
maxs = np.max(s, axis=0)
limits['amin'], limits['wmin'], limits['nmin'] = mins
limits['amax'], limits['wmax'], limits['nmax'] = maxs
limits['a'] = np.linspace(mins[0], maxs[0], npoints[0])
limits['w'] = np.linspace(mins[1], maxs[1], npoints[1])
limits['n'] = np.linspace(mins[2], maxs[2], npoints[2])
limits['da'] = (limits['a'][1] - limits['a'][0]) / 2
limits['dw'] = (limits['w'][1] - limits['w'][0]) / 2
limits['dn'] = (limits['n'][1] - limits['n'][0]) / 2

extent = dict()
extent['WvA'] = [limits['amin']-limits['da'], limits['amax']+limits['da'], limits['wmin']-limits['dw'], limits['wmax']+limits['dw']]
extent['NvA'] = [limits['amin']-limits['da'], limits['amax']+limits['da'], limits['nmin']-limits['dn'], limits['nmax']+limits['dn']]
extent['NvW'] = [limits['wmin']-limits['dw'], limits['wmax']+limits['dw'], limits['nmin']-limits['dn'], limits['nmax']+limits['dn']]

fig, ax = plt.subplots(2, 3, figsize=(15, 10))
ax[0,0].plot(limits['a'],likelihoods.sum(axis=(1,2)))

ax[0,1].plot(limits['w'],likelihoods.sum(axis=(0,2)))

ax[0,2].plot(limits['n'],likelihoods.sum(axis=(0,1)))


ax[1,0].imshow(likelihoods.sum(axis=2), extent=extent['WvA'], origin='lower', aspect='auto')
ax[1,0].set_xlabel(axlabels['a'])
ax[1,0].set_ylabel(axlabels['w'])
ax[1,0].plot(truths[0], truths[1], 'ro', label='Truth')

ax[1,1].imshow(likelihoods.sum(axis=1), extent=extent['NvA'], origin='lower', aspect='auto')
ax[1,1].set_xlabel(axlabels['a'])
ax[1,1].set_ylabel(axlabels['n'])
ax[1,1].plot(truths[0], truths[2], 'ro', label='Truth')

ax[1,2].imshow(likelihoods.sum(axis=0), extent=extent['NvW'], origin='lower', aspect='auto')
ax[1,2].set_xlabel(axlabels['w'])
ax[1,2].set_ylabel(axlabels['n'])
ax[1,2].plot(truths[1], truths[2], 'ro', label='Truth')

plt.show()
breakpoint()
