#!/usr/bin/env bash

for eps in 0.2 0.05 0.1 0.3 0.4 0.5 0.6 0.75 0.9; do

  config_name="eps${eps}"
  config_path="configuration/configs/sweeps/${config_name}.textproto"
  echo ${config_name}

  cp configuration/configs/sweeps/ppo_eps.textproto ${config_path}
  sed -i "s/ppo { clip_epsilon: REPLACE }/ppo { clip_epsilon: ${eps} }/g" ${config_path}

  python -m ppo.train --config="sweeps/${config_name}"
done


