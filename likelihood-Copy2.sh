#!/bin/bash

noise='0.0 0.00001 0.001 0.1'
noiseseed='0 1 2 3'

for n in $noise
do

for ns in $noiseseed
do


# python runAll.py 2.0 --noise $n --plot 'all' --noisyT21 --noiseSeed $ns --avgAdjacentFreqBins --append '_avgFreqBinsTrue' --retrain

# python runAll.py 2.0 --noise $n --plot 'all' --noisyT21 --noiseSeed $ns --combineSigma '4' --diffCombineSigma --avgAdjacentFreqBins --append '_avgFreqBinsTrue' --retrain

python runAll.py 2.0 --noise $n --plot 'all' --noisyT21 --noiseSeed $ns --combineSigma '4 6' --diffCombineSigma --avgAdjacentFreqBins --append '_avgFreqBinsTrue' --retrain 

wait

# python runAll.py 2.0 --noise $n --plot 'all' --noisyT21 --noiseSeed 1 --chromatic --retrain &
# python runAll.py 2.0 --noise $n --plot 'all' --noisyT21 --noiseSeed 2 --chromatic --retrain &
# python runAll.py 2.0 --noise $n --plot 'all' --noisyT21 --noiseSeed 3 --chromatic --retrain 

wait

done
done

echo 