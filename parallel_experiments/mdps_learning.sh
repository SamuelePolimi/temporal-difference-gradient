#!/bin/bash

cd ..

for j in $(seq 0 4)
do
    for i in $(seq 0 29)
    do
        python3 -m  experiments.mdps_learning $(($j*30 + $i)) &
    done
    wait
done