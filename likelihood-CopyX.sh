#!/bin/bash

# noise='0.0 0.00001 0.001 0.1'
noiseseed='0 1 2'
# sigmas='4 6'

# for s in $sigmas
# do

# for n in $noise
# do

for ns in $noiseseed
do


python runAll.py 2.0 --combineSigma '4 6' --noise 0.0 --plot 'all' --noisyT21 --noiseSeed $ns --append '_SVD' --gainFluctuationLevel 0.01 --gFdebug 3 --retrain &
python runAll.py 2.0 --combineSigma '4 6' --noise 0.0 --plot 'all' --noisyT21 --noiseSeed $ns --append '_SVD' --gainFluctuationLevel 0.05 --gFdebug 3 --retrain
wait

done

# done
# done

echo 


# appendId='0 1 2 3 4'

# for a in $appendId
# do

# python runAll.py 2.0 --noise 0.1 --plot 'all' --noisyT21 --noiseSeed 0 --append "_${a}" --retrain &
# python runAll.py 2.0 --noise 0.1 --plot 'all' --noisyT21 --noiseSeed 1 --append "_${a}" --retrain &
# python runAll.py 2.0 --noise 0.1 --plot 'all' --noisyT21 --noiseSeed 2 --append "_${a}" --retrain 
# wait

# done

# echo 





# noise='0.0 0.00001 0.001 0.1'

# # for n in $noise
# # do

# # python runAll.py 2.0 --noise $n --plot 'all' --noisyT21 --noiseSeed 0 --combineSigma '4 6' --diffCombineSigma --append '_diffCombineSigmav2True' &
# # python runAll.py 2.0 --noise $n --plot 'all' --noisyT21 --noiseSeed 1 --combineSigma '4 6' --diffCombineSigma --append '_diffCombineSigmav2True'

# # wait

# python runAll.py 2.0 --noise 0.001 --plot 'all' --noisyT21 --noiseSeed 2 --combineSigma '4 6' --diffCombineSigma --append '_diffCombineSigmav2True' --retrain
# wait
# python runAll.py 2.0 --noise 0.001 --plot 'all' --noisyT21 --noiseSeed 3 --combineSigma '4 6' --diffCombineSigma --append '_diffCombineSigmav2True' --retrain

# # wait

# # python runAll.py 2.0 --noise $n --plot 'all' --noisyT21 --noiseSeed 0 --combineSigma '4' --diffCombineSigma --append '_diffCombineSigmav2True' &
# # python runAll.py 2.0 --noise $n --plot 'all' --noisyT21 --noiseSeed 1 --combineSigma '4' --diffCombineSigma --append '_diffCombineSigmav2True'

# # wait

# # python runAll.py 2.0 --noise $n --plot 'all' --noisyT21 --noiseSeed 2 --combineSigma '4' --diffCombineSigma --append '_diffCombineSigmav2True' &
# # python runAll.py 2.0 --noise $n --plot 'all' --noisyT21 --noiseSeed 3 --combineSigma '4' --diffCombineSigma --append '_diffCombineSigmav2True'

# # wait

# # done

# echo 