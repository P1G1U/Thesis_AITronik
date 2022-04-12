#!/bin/bash
MAX_beta=2
MAX_drop=2

declare -a beta=("0.4" "0.6")
declare -a drop=("0.4" "0.6")

for (( i=0; i<$MAX_beta; i++))
do
for (( j=0; j<$MAX_drop;j++))
do
((num=$i*$MAX_drop+$j))

echo model $num

python3.9 script_model.py ${beta[$i]} ${drop[$j]} results$num.csv

done
done
