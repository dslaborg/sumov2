#!/bin/bash

model_path=output/final.ckpt

# quick check of the model on moda test data
#python bin/eval.py -i $model_path -t -e final

# MODA
python scripts/paper/eval_moda.py -m $model_path

# DREAMS
# recordings with C3 channel
ie_file_names=(excerpt1 excerpt3)

for ie_file_name in "${ie_file_names[@]}"
do
    ie_path=~/data/dreams/DatabaseSpindles/${ie_file_name}.edf
    is_path=~/data/dreams/DatabaseSpindles/Hypnogram_${ie_file_name}.txt
    python scripts/paper/eval_dreams.py -ie "$ie_path" -is "$is_path" -m $model_path -c C3-A1
done

# recordings with CZ channel
ie_file_names=(excerpt2 excerpt4 excerpt5 excerpt6 excerpt7 excerpt8)

for ie_file_name in "${ie_file_names[@]}"
do
    ie_path=~/data/dreams/DatabaseSpindles/${ie_file_name}.edf
    is_path=~/data/dreams/DatabaseSpindles/Hypnogram_${ie_file_name}.txt
    python scripts/paper/eval_dreams.py -ie "$ie_path" -is "$is_path" -m $model_path -c CZ-A1
done
