import numpy as np
import fitsio
import NormalizingFlow as nf
import lusee
import parser
import os
import matplotlib.pyplot as plt
import corner

##---------------------------------------------------------------------------##
# argparser block

parser = parser.create_parser()
args = parser.parse_args()

# must have --noisyT21 and --diffCombineSigma!!!
args.noisyT21 = True
args.diffCombineSigma = True

args.appendLik = "_cube"

args.print()

##---------------------------------------------------------------------------##

# set seed
print("setting noise and torch seeds...")
np.random.seed(args.noiseSeed)
torch.manual_seed(args.torchSeed)
torch.cuda.manual_seed_all(args.torchSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda")
torch.set_default_tensor_type("torch.cuda.FloatTensor")

# try loading
fname = nf.get_fname(args)
print(f"loading flow from {fname}")
flow = nf.FlowAnalyzerV2(nocuda=False, loadPath=fname)
flow.set_fg(args)

print("setting t21...")
tcmb = nf.T_CMB(flow.freqs)
flow.set_t21(tcmb, include_noise=args.noisyT21)
if args.retrain:
    flow.train(
        flow.train_data, flow.validate_data, nocuda=False, savePath=fname, retrain=True
    )

##---------------------------------------------------------------------------##
# 1D
npoints = 1000
kwargs = {
    "amin": 0.1,
    "amax": 10000.0,
    "wmin": 11.0,
    "wmax": 17.0,
    "nmin": 15.0,
    "nmax": 18.0,
    "logspace": True,
}

vs = "A"
print(f"getting 1d likelihood for {vs}...")
samples1d, t21_vs1d = nf.get_t21vs1d(npoints, vs, cmb=True, **kwargs)
print(t21_vs1d)
t21vsdata1d = flow.proj_t21(t21_vs1d, include_noise=True)
likelihood1d = flow.get_likelihood(
    t21vsdata1d, args.freqFluctuationLevel, args.DA_factor, debugfF=False
)
lname = nf.get_lname(args, plot=vs)
print(f"saving 1d likelihood results to {lname}")
np.savetxt(
    lname,
    np.column_stack([samples1d, likelihood1d]),
    header="amp,width,numin,loglikelihood",
)

# s,ll=nf.get_samplesAndLikelihood(args,plot='A')
# quantiles=corner.core.quantile(s[:,0],[0.32,0.5,0.68,0.95],weights=nf.exp(ll))
# dq=quantiles[2]-quantiles[0]
# maxll=s[np.argmax(ll),0]
# for x in quantiles:
#     plt.axvline(2.718*x,c='k',alpha=0.5,lw=0.5)
# # plt.axvline(40,color='k')
# plt.plot(2.718*s,nf.exp(ll))
# plt.xscale('log')
# # plt.xlim(3e3,5e3)
# plt.title(f'Amplitude {2.718*quantiles[1]:.3f} +/-{2.718*dq:.3f} K, <{2.718*quantiles[3]:.3f} K 95% CL, max likelihood {2.718*maxll:.3f} K')
# plt.show()
