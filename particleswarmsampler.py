import pyswarms
import numpy as np
import torch
import NormalizingFlow as nf
import lusee
import parser
import corner
import matplotlib.pyplot as plt

# Example:
# python runPSO.py --noise 0.0 --combineSigma '4 6' --freqs '1 51' --fgFITS 'ulsa.fits'

##---------------------------------------------------------------------------##
# argparser block

argparser = parser.create_parser()
args = argparser.parse_args()

# must have --noisyT21 and --diffCombineSigma!!!
args.noisyT21 = True
args.diffCombineSigma = True

if args.appendLik == "":
    args.appendLik = "_PSO"

parser.prettyprint(args)

##---------------------------------------------------------------------------##

# set seed
print(f"setting noise seed {args.noiseSeed} and torch seed {args.torchSeed}")
np.random.seed(args.noiseSeed)
torch.manual_seed(args.torchSeed)
torch.cuda.manual_seed_all(args.torchSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda")
torch.set_default_tensor_type("torch.cuda.FloatTensor")


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
    priorlow,priorhigh = [0.01,10,50],[10,40,90]
    start=[(0.01, 10), (10, 40), (50, 90)]
else:
    priorlow,priorhigh = [0.01,10,10],[10,30,30]
    start=[(0.01, 10), (1, 30), (1, 30)]

def like(x):
    return flow.get_likelihoodFromSamplesGAME(
        x, priorlow=priorlow, priorhigh=priorhigh
    )

# c1: local preference, c2: global preference
options = {'c1': 1.0, 'c2': 1.0, 'w':1.0}
bounds = ([0.01,1,1],[10,40,40])
optimizer = pyswarms.single.GlobalBestPSO(n_particles=10000, dimensions=3, options=options, bounds=bounds)
cost, pos = optimizer.optimize(like, iters=50)

##---------------------------------------------------------------------------##
breakpoint()
