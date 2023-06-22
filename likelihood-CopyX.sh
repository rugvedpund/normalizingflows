#!/bin/bash
noiseseed='1 2 3 4 5 6 7 8 9'
noise='0.0 0.000001 0.00001 0.0001 0.001 0.01 0.1'

for n in $noise
do

for seed in $noiseseed
do
python runAll.py 2.0 --noise $n --plot 'all' --combineSigma '4 6' --diffCombineSigma --append "_diffCombineSigmav2True" --noisyT21 --noiseSeed $seed --retrain
wait
python runAll.py 2.0 --noise $n --plot 'all' --combineSigma '4' --diffCombineSigma --append "_diffCombineSigmav2True" --noisyT21 --noiseSeed $seed --retrain
wait
done
done

echo 