#!/bin/bash

noiseseed='0 1 2 3 4 5 6 7 8 9'

for s in $noiseseed
do


python runAll.py 2.0 --galcut 20 --noise 0.0  --noiseSeed $s --noisyT21 --plot 'all' --DA_factor 1.0 --subsampleSigma 2.0  --gainFluctuationLevel 0.0 --gFdebug 0 --combineSigma '4'

# --retrain | grep 'logp:' > logs/cS_seed$s
wait


done

echo 