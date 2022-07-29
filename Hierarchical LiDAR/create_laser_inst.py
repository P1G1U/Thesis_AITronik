import sys
import pandas as pd

with open("data/{}".format(sys.argv[1]), "r") as data:
    dataset = pd.read_csv(data, delimiter=";").copy()

inst = int(sys.argv[2])
laser_inst = []

added = False
x_real = None
y_real = None
rot_real = None

for i in dataset.values:
    if i[0] == inst:
        laser_inst.append(i[3])

        if not added:
            x_real = i[4]
            y_real = i[5]
            rot_real = i[6]

            added = True

with open('data/laser/laser_inst_{}.csv'.format(inst), 'w') as f:
    f.write("range\n")
    for item in laser_inst:
        f.write("{}\n".format(item))

with open('metrics/truth_path.csv'.format(inst), 'a+') as f:
    f.write("{};{};{}\n".format(x_real, y_real, rot_real))
