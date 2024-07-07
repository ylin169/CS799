import torch
import random
import pickle

from sklearn.preprocessing import minmax_scale, MinMaxScaler
'''
State_List_org_1 = [
    [[0.1840277777777778, 9999.996768482733, 0.36617302894592285], [0.0, 9999.996529543754, 0.4310758113861084]]]

State_List_org_2 = [
    [[0.1840277777777778, 1.996768482733, 2.36617302894592285], [0.0, 1.996529543754, 2.4310758113861084]]]


# packet loss, bandwidth, latency
QoS_List = [[0.2, 5000, 0.4], [0.2, 6000, 0.9],[0.2, 2000, 0.5],[0.1, 2000, 0.5]]


State_List = [[[0.9172714078374455, 1.0, 0.0], [0.0, 0.0, 0.13016185107243616]]]
Normailized_QoS_list = [[[0.0, 0.75, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.19999999999999996], [1.0, 0.0, 0.8754636764526367]]]

for x in State_List[0]:
    x.append(Normailized_QoS_list[0][0][0])
    x.append(Normailized_QoS_list[0][0][1])
    x.append(Normailized_QoS_list[0][0][2])
    #x.pop(2)
    #x.pop(2)
    #x.pop(2)



print('b:', State_List)

for x in State_List[0]:
    x.pop(3)
    x.pop(3)
    x.pop(3)

print('a:', State_List)

input_data = torch.randn(3, 10)
print('input_data', input_data)
'''
final_latecny = {}
final_packet_loss = {}
Path_Available_Bandwidth = {}
QoS_List = []
#final_packet_loss = {0: 0.2, 1: 0.2, 2: 0.0, 3: 0.0, 4: 0.12, 5: 0.12, 6: 0.31, 7: 0.31}
#Path_Available_Bandwidth = {0: [4500], 1: [4500], 2: [4500], 3: [4500], 4: [4500], 5: [4500], 6: [4500], 7: [4500]}
#final_latecny = {0: 0.4, 1: 0.4, 2: 0.44, 3: 0.44, 4: 0.74, 5: 0.74, 6: 0.85, 7: 0.85}
#QoS_List = [[0.2, 4500, 0.4], [0.2, 4500, 0.4],[0.0, 4500, 0.44],[0.0, 4500,0.44],[0.12, 4500, 0.74],[0.12, 4500, 0.74],[0.31,4500,0.85],[0.31,4500,0.85],[0,0,0]]  #packet loss less than 0.2, bandwidth at least 500, latecny less than 31]
final_latecny = {}
final_packet_loss = {}
Path_Available_Bandwidth = {}
QoS_List = []
x = 0
sum_bandwidth = []



for x in range(12):
    random_final_packet_loss = random.randint(1, 99)/100
    final_packet_loss[x] = random_final_packet_loss
    QoS_List.append([random_final_packet_loss])

for x in range(12):
    random_Available_Bandwidth = random.randint(100, 3000)
    sum_bandwidth.append(random_Available_Bandwidth)
    Path_Available_Bandwidth[x] = [random_Available_Bandwidth]
    QoS_List[x].append(random_Available_Bandwidth)
for x in range(12):
    random_final_latecny = random.randint(1, 99)/100
    final_latecny[x] = random_final_latecny
    QoS_List[x].append(random_final_latecny)

sum_bandwidth_v = sum(sum_bandwidth)

print(final_packet_loss)
print(Path_Available_Bandwidth)
print(final_latecny)

print(QoS_List)

print(sum_bandwidth_v)

# State_List_O = [[[0.1840277777777778, 9999.996768482733, 0.36617302894592285], [0.0, 9999.996529543754, 0.4310758113861084]]]
State_List_O = [
    [[0.1840277777777778, 9999.996768482733, 0.36617302894592285], [0.0, 9999.996529543754, 0.4310758113861084],
     [0.11805555555555555, 9999.996529111084, 0.7347431182861328],
     [0.30208333333333337, 9999.996529543754, 0.8377318382263184]]]
# packet loss, bandwidth, latency
# State_List = [[[0.9172714078374455, 1.0, 0.0], [0.0, 0.0, 0.13016185107243616]]]
State_List = [[[0.9172714078374455, 1.0, 0.0], [0.0, 0.0, 0.13016185107243616],
               [0.6661828737300435, 0.5000000074505806, 0.7830560718473345], [1.0, 0.0, 1.0]]]

QoS_List = [[0.31, 8321, 0.99], [0.57, 6237, 0.98], [0.2, 6214, 0.99]]

Normailized_QoS_list = [[[0.29729729729729726, 1.0, 1.0], [0.9999999999999999, 0.010915994304698806, 0.0], [0.0, 0.0,
                                                                                                            1.0]]]  # QoS_List = [[0.2, 5000, 0.4],[0,5000,0.5]]  #packet loss less than 0.2, bandwidth at least 500, latecny less than 31]
# Normailized_QoS_list = [[[1.0, 1.0, 0.8],[0,1,1],[0,0,0]]] 


with open("state_list.txt", "rb") as f:
    final_packet_loss = pickle.load(f)
    free_bw = pickle.load(f)
    final_latecny = pickle.load(f)
    print('Loaded final_packet_loss: ', final_packet_loss)
    print('Loaded free_bw: ', free_bw)
    print('Loaded final_latecny: ', final_latecny)

import pickle


# Path to the uploaded file
file_path = 'state_list.txt'


# Function to read the pickle file and print its contents
def read_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        final_packet_loss = pickle.load(file)
        free_bw = pickle.load(file)
        final_latecny = pickle.load(file)

        return final_packet_loss, free_bw, final_latecny


# Read the contents of the pickle file
final_packet_loss, free_bw, final_latecny = read_pickle_file(file_path)

final_packet_loss, free_bw, final_latecny

print(final_packet_loss, free_bw, final_latecny)


def transform_state(final_packet_loss, free_bw, final_latecny):
    State_List_O = []

    keys = final_packet_loss.keys()

    for key in keys:
        packet_loss = round(final_packet_loss[key], 4)
        bandwidth = round(free_bw[key][0] if isinstance(free_bw[key], list) else free_bw[key], 2)
        latency = round(final_latecny[key], 5)
        State_List_O.append([packet_loss, bandwidth, latency])

    # Wrap the result in another list
    State_List_O = [State_List_O]

    return State_List_O


# Transform the data
State_List_O = transform_state(final_packet_loss, free_bw, final_latecny)
print('xx:', State_List_O)

Path_Available_Bandwidth = free_bw


Path_Available_Bandwidth = str(free_bw).replace("[", "").replace("]", "")

Path_Available_Bandwidth = eval(Path_Available_Bandwidth)

print('xxxxxxxx: ',Path_Available_Bandwidth)

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

State_List_n = [[]]

for x_id, y_id in path_matrix_n.items():
    State_List_n[0].append(y_id)

print('Normalized_value： ', State_List_n)