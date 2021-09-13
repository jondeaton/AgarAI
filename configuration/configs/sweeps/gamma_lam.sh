#!/usr/bin/env bash


for gamma in 0.999 0.1 0.75 0.5 0.9 0.2 0.99; do
  for lam in 0.99  0.1 0.9 0.5  0.01; do

    config_name="g${gamma}-l${lam}"
    config_path="configuration/configs/sweeps/${config_name}.textproto"
    echo ${config_name}

    cp configuration/configs/sweeps/gamma_lam.textproto ${config_path}

    sed -i "s/gamma:REPLACE/gamma:${gamma}/g" ${config_path}
    sed -i "s/gae{lam:REPLACE}/gae{lam:${lam}}/g" ${config_path}

    python -m ppo.train --config="sweeps/${config_name}"
  done
done


