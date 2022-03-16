from os import listdir
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from math import log10
import pandas as pd
import plotly.express as px
from scipy.stats import chi2_contingency

def list_str_2_int(list_of_strings):
    return list(map(int, list_of_strings))

class Demon:
    def __init__(self, ints, turns_num) -> None:
        self.stamina_consumed = ints[0]
        self.turns_recovering = ints[1]
        self.stamina_recovered = ints[2]
        self.fragments_list = ints[4:]

        #cannot be longer than turns_num
        if len(self.fragments_list) > turns_num:
            self.fragments_list = self.fragments_list[:turns_num]

        self.total_fragments = sum(self.fragments_list)
        
        if self.total_fragments == 0:
            self.fragments_list = []
        else:
            for i in range(len(self.fragments_list) - 1, -1, -1):
                if self.fragments_list[i] != 0:
                    break
            del self.fragments_list[i + 1:]
        
        self.turns_fragments = len(self.fragments_list)
    

def parse_file(filename) -> Tuple[int,int,int,int,List[Demon]]:
    demons = []
    with open('inputs/'+filename,'r') as f:
        stamina_init, stamina_max, turns_num, demons_num  = list_str_2_int(f.readline().strip().split(' '))
        for line in f:
            demons.append(Demon(list_str_2_int(line.strip().split(' ')), turns_num))
        assert len(demons) == demons_num
    return stamina_init, stamina_max, turns_num, demons_num, demons

for filename in sorted(listdir('inputs')):
    if '00' in filename:
        continue
    print(filename)
    stamina_init, stamina_max, turns_num, demons_num, demons = parse_file(filename)
    print('\t',stamina_init, stamina_max, turns_num, demons_num)
    print('\t',
        max(d.stamina_consumed for d in demons),
        max(d.stamina_recovered for d in demons),
        max(d.turns_recovering for d in demons),
        max(d.turns_fragments for d in demons)
        )
    print('\t',
        min(d.stamina_consumed for d in demons),
        min(d.stamina_recovered for d in demons),
        min(d.turns_recovering for d in demons),
        min(d.turns_fragments for d in demons)
        )
    # print('\t',
    #     len([d for d in demons if d.fragments_list == []])/ demons_num
    #     )
    # turns = sorted([d.turns_recovering for d in demons])
    # if turns_num < len(turns):
    #     print('\t',turns[turns_num])
    # else:
    #     print('\tA',turns_num,len(turns))
    # print(len([d for d in demons if d.turns_fragments > turns_num]))

exit()
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     p = ax.scatter(
#         [d.stamina_consumed / stamina_max for d in demons],
#         [d.stamina_recovered / stamina_max for d in demons],
#         [d.turns_recovering / turns_num for d in demons],
#         c = [d.total_fragments for d in demons])
#     fig.colorbar(p)
#     ax.set_xlabel('stamina_consumed')
#     ax.set_ylabel('stamina_recovered')
#     ax.set_zlabel('turns_recovering')
#     plt.show()

filenames = [
    '01-the-cloud-abyss.txt',
    '02-iot-island-of-terror.txt',
    '04-the-desert-of-autonomous-machines.txt'
]

for filenum, filename in enumerate(filenames):
    stamina_init, stamina_max, turns_num, demons_num, demons = parse_file(filename)

    # ax = fig.add_subplot(1, 3, filenum+1, projection='3d')
    # p = ax.scatter(
    #     [d.stamina_consumed / stamina_max for d in demons],
    #     [d.stamina_recovered / stamina_max for d in demons],
    #     [d.turns_recovering / turns_num for d in demons],
    #     c = [d.total_fragments for d in demons])
    # #fig.colorbar(p)
    # ax.set_xlabel('stamina_consumed')
    # ax.set_ylabel('stamina_recovered')
    # ax.set_zlabel('turns_recovering')
    # ax.set_title(filename)

    df = pd.DataFrame({
        'stamina_consumed' : [d.stamina_consumed for d in demons],
        'stamina_recovered' : [d.stamina_recovered for d in demons],
        'turns_recovering' : [d.turns_recovering for d in demons],
    })

    fig = px.scatter(df, x = 'stamina_recovered', y = 'turns_recovering', color = 'stamina_consumed')
    fig.show()
    # obs = pd.crosstab(df['stamina_consumed'], df['stamina_recovered']).to_numpy()
    # chi2, p, dof, ex = chi2_contingency(obs)
    # print(p)



    # plt.subplot(3,3,filenum+1)
    # plt.hist([d.stamina_consumed for d in demons], bins = 25, range = (1,25), align='mid')
    # plt.subplot(3,3,3+filenum+1)
    # plt.hist([d.stamina_recovered for d in demons], bins = 40, range = (1,40), align='mid')
    # plt.subplot(3,3,6+filenum+1)
    # plt.hist([d.turns_recovering for d in demons], bins = 100, range = (1,100), align='mid')

