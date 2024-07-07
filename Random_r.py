from operator import itemgetter

import numpy as np
from sklearn.preprocessing import minmax_scale, MinMaxScaler
import random
from numpy import asarray
import ast
'''
QoS_List = [[0.2, 4500, 0.4], [0.2, 4500, 0.4],[0.0, 4500, 0.44],[0.0, 4500,0.44],[0.12, 4500, 0.74],[0.12, 4500, 0.74],[0.31,4500,0.85],[0.31,4500,0.85],[0,0,0]]  #packet loss less than 0.2, bandwidth at least 500, latecny less than 31]

final_packet_loss = {0: 0.2, 1: 0.2, 2: 0.0, 3: 0.0, 4: 0.12, 5: 0.12, 6: 0.31, 7: 0.31}
Path_Available_Bandwidth = {0: [4500], 1: [4500], 2: [4500], 3: [4500], 4: [4500], 5: [4500], 6: [4500], 7: [4500]}
final_latecny = {0: 0.4, 1: 0.4, 2: 0.44, 3: 0.44, 4: 0.74, 5: 0.74, 6: 0.85, 7: 0.85}
'''

final_latecny = {}
final_packet_loss = {}
Path_Available_Bandwidth = {}
QoS_List = []
x = 0
number_of_Qos = 6

sum_bandwidth = []


for x in range(number_of_Qos): #higherest +1
    random_final_packet_loss = random.uniform(0.0158, 0.0211)
    final_packet_loss[x] = random_final_packet_loss
    QoS_List.append([random_final_packet_loss])

for x in range(number_of_Qos):
    random_Available_Bandwidth = random.randint(50, 75)
    sum_bandwidth.append(random_Available_Bandwidth)
    Path_Available_Bandwidth[x] = [random_Available_Bandwidth]
    QoS_List[x].append(random_Available_Bandwidth)

for x in range(number_of_Qos):
    random_final_latecny = random.uniform(0.006, 0.03)
    final_latecny[x] = random_final_latecny
    QoS_List[x].append(random_final_latecny)

sum_bandwidth_v = sum(sum_bandwidth)
'''


final_packet_loss = {0: 0.0, 1: 0.0158, 2: 0.0132}
Path_Available_Bandwidth = {0: [300], 1: [200], 2: [150]}
final_latecny = {0: 0.007, 1: 0.003, 2: 0.01}
for key in final_latecny:
    final_latecny[key] *= 2
'''
print(final_latecny)

Path_Available_Bandwidth = str(Path_Available_Bandwidth).replace("[", "").replace("]", "")
Path_Available_Bandwidth = eval(Path_Available_Bandwidth)

path_matrix_n = {}
path_matrix = {}
State_List = [[]]

for (key1, value1), (key2, value2), (key3, value3) in zip(final_packet_loss.items(), Path_Available_Bandwidth.items(),
                                                          final_latecny.items()):
    path_matrix[key1] = [value1, value2, value3]
    # print(key1)]

print('Path_matrix: ', path_matrix)
for x_id, y_id in path_matrix.items():
    State_List[0].append(y_id)

print('State_List: ', State_List)

State_List[0] = sorted(State_List[0], key=itemgetter(1), reverse=True)

print('State_List_sorted: ', State_List)


def normal(var):
    list = []

    for item in var.values():
        list.append(item)

    list = minmax_scale(list)

    for x in var:
        var[x] = list[x]


normal(final_packet_loss)
normal(Path_Available_Bandwidth)
normal(final_latecny)

for (key1, value1), (key2, value2), (key3, value3) in zip(final_packet_loss.items(), Path_Available_Bandwidth.items(),
                                                          final_latecny.items()):
    path_matrix_n[key1] = [value1, value2, value3]
    # print(key1)

# print('x',path_matrix_n)


# print('final_packet_loss: ', final_packet_loss)
# print('Path_Available_Bandwidth: ', Path_Available_Bandwidth)
# print('final_latecny: ',final_latecny)
print('path_matrix_n: ', path_matrix_n)

State_List_n = [[]]

for x_id, y_id in path_matrix_n.items():
    State_List_n[0].append(y_id)

print('Normalized_value： ', State_List_n)


def normal_r(State_List, State_List_n):
    temp_n = []
    for item in State_List[0]:
        temp_n.append(item[1])

    temp_n = minmax_scale(temp_n)
    # print('::', temp_n)
    for idx, item in enumerate(temp_n):
        State_List_n[0][idx][1] = item
        #print('idx: ', idx, 'item:', item, 'State_List:', State_List[0][idx])
        #print('item:', item)
    print('Normailized_QoS_list = ', State_List_n)
    print('QoS_List =', State_List[0])
    return State_List_n


normal_r(State_List, State_List_n)

print('QoS_List =',QoS_List)

#print(sum_bandwidth_v)



