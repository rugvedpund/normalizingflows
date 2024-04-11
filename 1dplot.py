import numpy as np
import NormalizingFlow as nf
import corner
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

s,ll = nf.get_samplesAndLikelihood(args, plot="A")
q16,q50,q84,q95 = corner.core.quantile(s[:,0],[0.16,0.5,0.84,0.95],weights=nf.exp(ll))
qm,qp = q50-q16,q84-q50
title=rf"${q50:.2f}^{{+{qp:.2f}}}_{{-{qm:.2f}}}, {q95:.2f} 95\% CL$"

print("Total number of samples: ", s.shape)
print("Max likelihood:", max(ll))

plt.plot(s[:,0],nf.exp(ll))
plt.axvline(1.0,color='k',alpha=0.5,lw=0.5)
plt.xscale('log')
plt.xlim(0.01,100)
plt.figtext(0.75,0.5,parser.argstostring(args),bbox=dict(facecolor='gray', alpha=0.5),fontsize="xx-small",ma='center')
plt.title(title)
plt.show()

