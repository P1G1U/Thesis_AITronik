#!/bin/bash
LENGTH_path=630

for (( i=1; i<$LENGTH_path; i+=3))
do

python3.9 create_laser_inst.py datasetsanpiero_path.csv $i

python3.9 hierarchical_model.py sanPiero.png laser_inst_$i.csv

done
