# Example:
# python runCubeGame.py --noise 0.0 --combineSigma '4 6' --freqs '1 51' --fgFITS 'ulsa.fits'

import cubesampler
import numpy as np
import torch
import NormalizingFlow as nf
import lusee
import parser
import corner
import matplotlib.pyplot as plt

##---------------------------------------------------------------------------##
# argparser block

argparser = parser.create_parser()
args = argparser.parse_args()

# must have --noisyT21 and --diffCombineSigma!!!
args.noisyT21 = True
args.diffCombineSigma = True

if args.appendLik == "":
    args.appendLik = "_cube"

parser.prettyprint(args)

##---------------------------------------------------------------------------##
# check if exists

# import exists
# if exists.exists(args):
#     cont=input("Likelihood already exists! Do you want to continue? (y/n)")
#     if cont != "y":
#         exit()

##---------------------------------------------------------------------------##

fname = nf.get_fname(args)

print(f"loading flow from {fname}")
flow = nf.FlowAnalyzerV2(nocuda=False, loadPath=fname)
flow.set_fg(args)

if args.fgFITS == "ulsa.fits":
    print("using DA model")
    t21 = lusee.MonoSkyModels.T_DarkAges_Scaled(
        flow.freqs, nu_rms=14, nu_min=16.4, A=0.04
    )
    cosmicdawn = False
elif args.fgFITS == "gsm16.fits":
    print("using CD model")
    t21 = lusee.MonoSkyModels.T_CosmicDawn_Scaled(
        flow.freqs, nu_rms=20, nu_min=67.5, A=0.130
    )
    cosmicdawn = True
flow.set_t21(t21)
if args.retrain:
    flow.train(
        flow.train_data, flow.validate_data, nocuda=False, savePath=fname, retrain=True
    )

##---------------------------------------------------------------------------##
# main sampler block

if cosmicdawn:
    priorlow, priorhigh = [0.01, 10, 50], [1000, 40, 90]
else:
    priorlow, priorhigh = [0.01, 10, 10], [1000, 30, 30]


def like(x):
    return flow.get_likelihoodFromSamplesGAME(x, priorlow=priorlow, priorhigh=priorhigh)


truth = [1., 20., 67.5] if cosmicdawn else [1., 14., 16.4]
amin, amax = 0.01, 100
npoints = 10000
amp = np.logspace(np.log(amin), np.log(amax), npoints, base=10)
width = truth[1] * np.ones(npoints)
numin = truth[2] * np.ones(npoints)
samples = np.vstack([amp, width, numin]).T

loglikelihoods = like(samples)

# save

lname = nf.get_lname(args, plot="A")
print(f"saving likelihood results to {lname}")
np.savetxt(
    lname,
    np.column_stack([samples, loglikelihoods]),
    header="amp,width,numin,loglikelihood",
)
