#!/bin/bash

cd ..

for j in $(seq 0 4)
do
    for i in $(seq 0 19)
    do
        python3 -m  experiments.mdps_learning_numerical $(($j*20 + $i)) &
    done
    wait
done