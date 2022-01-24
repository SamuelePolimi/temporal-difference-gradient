#!/bin/bash

cd ..

for j in $(seq 0 4)
do
    for i in $(seq 0 9)
    do
        python3 -m  experiments.mdps_learning $(($j*10 + $i)) &
    done
    wait
done