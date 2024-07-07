import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import random
from random import sample
from torch import IntTensor

from Normal_r import normal_r
from agent import Agent
from agent import ReplayMemory
import constants
import copy
import matplotlib.pyplot as plt
from chainer import serializers
from sklearn.preprocessing import minmax_scale, MinMaxScaler





# BUFFER_SIZE = 500000
EPSILON_START = 0.9
EPSILON_END = 0.2
EPSILON_DECAY =100
TARGET_UPDATE_FREQUENCY = 100
n_episode =1200
#was 50000
PATH = "state_dict_model1.pth"
PATH2 = "state_dict_model2.pth"
State_List_org_1 = [
    [[0.1840277777777778, 9999.996768482733, 0.36617302894592285], [0.0, 9999.996529543754, 0.4310758113861084]]]
AVG_REWARD = []
STATS_EVERY = 100
eval_interval = 10
loss_buffer = []
step_buffer = []
success_counters = 0
r_next = 0
step_buffer_c = []
episode_i_c = 0
episode_no_neg_r = 0
avg_loss_buff = []


#State_List_O = [[[0.1840277777777778, 9999.996768482733, 0.36617302894592285], [0.0, 9999.996529543754, 0.4310758113861084]]]
file_path = 'state_list.txt'
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

State_List = [[]]

for x_id, y_id in path_matrix_n.items():
    State_List[0].append(y_id)

print('Normalized_value： ', State_List)


'''
State_List_O = [[[0.0169, 100, 0.02], [0, 200, 0.0135], [0.0126, 250, 0.00634]]]
# packet loss, bandwidth, latency
#State_List = [[[0.9172714078374455, 1.0, 0.0], [0.0, 0.0, 0.13016185107243616]]]
State_List = [[[1.0, 0.0, 1.0], [0.0, 0.6666666666666667, 0.5241581259150805], [0.7455621301775149, 1.0, 0.0]]]
'''




path = [[[1, 2, 3, 4], [1, 5, 6, 4], [1, 7, 8, 4]]]
path_pair = []
#,[0,0,0]
Normailized_QoS_list =  [[[1.0, 1.0, 0.9862004425251647], [0.0, 0.8214285714285716, 1.0], [1.0, 0.5, 0.431637961740035], [0.0, 0.1785714285714286, 0.5995426018744925], [1.0, 0.03571428571428581, 0.5745535257180476], [1.0, 0.0, 0.0],[0,0,0]]]
QoS_List = [[0.0126, 78, 0.01128765439451277], [0, 73, 0.015764225061579963], [0, 64, 0.013971549219721267], [0.0126, 55, 0.013859683854402217], [0.0126, 51, 0.015702450367369607], [0.0126, 50, 0.013219912232830882],[0,0,0]]
'''
Normailized_QoS_list =  [[[0.752577319587629, 0.9948979591836734, 0.0], [0.010309278350515462, 0.9489795918367346, 0.25], [0.37113402061855677, 0.9285714285714285, 1.0], [0.37113402061855677, 0.04081632653061218, 0.0], [0.8247422680412371, 0.7091836734693876, 0.5], [0.4639175257731959, 0.3928571428571428, 0.25], [0.4226804123711341, 0.7602040816326529, 0.25], [0.2989690721649485, 0.933673469387755, 0.25], [0.2371134020618557, 0.9999999999999999, 0.5], [0.5979381443298969, 0.22448979591836726, 0.5], [0.4536082474226805, 0.2806122448979591, 0.0], [0.6907216494845361, 0.34183673469387743, 1.0], [0.4123711340206186, 0.5561224489795918, 1.0], [0.0, 0.3826530612244897, 0.75], [0.31958762886597947, 0.29591836734693866, 0.75], [0.49484536082474234, 0.061224489795918324, 0.5], [0.309278350515464, 0.20408163265306112, 0.0], [0.4536082474226805, 0.3979591836734693, 0.75], [0.5257731958762887, 0.6428571428571427, 0.75], [0.16494845360824742, 0.6377551020408162, 0.0], [0.1443298969072165, 0.780612244897959, 0.0], [0.13402061855670103, 0.6530612244897959, 0.75], [0.7422680412371134, 0.683673469387755, 0.5], [0.24742268041237117, 0.9438775510204079, 0.75], [0.3608247422680413, 0.9540816326530611, 0.25], [0.041237113402061855, 0.520408163265306, 0.0], [1.0000000000000002, 0.22448979591836726, 0.25], [0.7835051546391754, 0.6989795918367346, 0.25], [0.24742268041237117, 0.14795918367346927, 0.0], [0.7010309278350515, 0.47448979591836726, 0.25], [0.07216494845360824, 0.1020408163265305, 0.75], [0.31958762886597947, 0.9999999999999999, 1.0], [0.5567010309278352, 0.13265306122448972, 0.0], [0.8350515463917526, 0.8061224489795916, 0.5], [0.21649484536082478, 0.5816326530612245, 0.5], [0.09278350515463918, 0.4285714285714285, 1.0], [0.49484536082474234, 0.7244897959183673, 0.5], [0.37113402061855677, 0.7653061224489796, 0.75], [0.02061855670103093, 0.0, 1.0], [0.8247422680412371, 0.7346938775510202, 0.25], [0.5257731958762887, 0.770408163265306, 0.0], [0.13402061855670103, 0.7142857142857141, 0.25], [0.8350515463917526, 0.5816326530612245, 0.75], [0.8762886597938145, 0.13265306122448972, 0.5], [0.9587628865979382, 0.0, 0.5], [0.26804123711340216, 0.9744897959183673, 1.0], [0.154639175257732, 0.9489795918367346, 0.25], [0.010309278350515462, 0.8214285714285713, 0.25], [0.845360824742268, 0.3928571428571428, 1.0], [0.8144329896907218, 0.5153061224489796, 0.0],[0,0,0],[0,0,0]]]
QoS_List = [[0.75, 295, 0.95], [0.03, 286, 0.96], [0.38, 282, 0.99], [0.38, 108, 0.95], [0.82, 239, 0.97], [0.47, 177, 0.96], [0.43, 249, 0.96], [0.31, 283, 0.96], [0.25, 296, 0.97], [0.6, 144, 0.97], [0.46, 155, 0.95], [0.69, 167, 0.99], [0.42, 209, 0.99], [0.02, 175, 0.98], [0.33, 158, 0.98], [0.5, 112, 0.97], [0.32, 140, 0.95], [0.46, 178, 0.98], [0.53, 226, 0.98], [0.18, 225, 0.95], [0.16, 253, 0.95], [0.15, 228, 0.98], [0.74, 234, 0.97], [0.26, 285, 0.98], [0.37, 287, 0.96], [0.06, 202, 0.95], [0.99, 144, 0.96], [0.78, 237, 0.96], [0.26, 129, 0.95], [0.7, 193, 0.96], [0.09, 120, 0.98], [0.33, 296, 0.99], [0.56, 126, 0.95], [0.83, 258, 0.97], [0.23, 214, 0.97], [0.11, 184, 0.99], [0.5, 242, 0.97], [0.38, 250, 0.98], [0.04, 100, 0.99], [0.82, 244, 0.96], [0.53, 251, 0.95], [0.15, 240, 0.96], [0.83, 214, 0.98], [0.87, 126, 0.97], [0.95, 100, 0.97], [0.28, 291, 0.99], [0.17, 286, 0.96], [0.03, 261, 0.96], [0.84, 177, 0.99], [0.81, 201, 0.95],[0,0,0],[0,0,0]]
'''

'''
QoS_List = [[0.2, 4500, 0.4], [0.2, 4500, 0.4],[0.0, 4500, 0.44],[0.0, 4500,0.44],[0.12, 4500, 0.74],[0.12, 4500, 0.74],[0,0,0]]  #packet loss less than 0.2, bandwidth at least 500, latecny less than 31]
Normailized_QoS_list = [[[0.6451612903225807, 0.0, 0.0], [0.6451612903225807, 0.0, 0.0],
                         [0.0, 0.0, 0.0888888888888889], [0.0, 0.0, 0.0888888888888889],
                         [0.3870967741935484, 0.0, 0.7555555555555556], [0.3870967741935484, 0.0, 0.7555555555555556],
                         [0,0,0]]]




Normailized_QoS_list =  [[[0.9042553191489362, 1.0, 0.75], [0.4680851063829787, 0.9913294797687862, 1.0], [0.6808510638297873, 0.9884393063583816, 0.5], [0.14893617021276598, 0.9783236994219653, 0.25], [0.5212765957446809, 0.976878612716763, 1.0], [0.10638297872340427, 0.9320809248554913, 0.25], [0.5212765957446809, 0.9291907514450868, 0.5], [1.0, 0.9234104046242775, 0.0], [0.2659574468085107, 0.8959537572254335, 1.0], [0.648936170212766, 0.869942196531792, 0.5], [0.5957446808510638, 0.8395953757225434, 0.75], [0.9680851063829786, 0.8049132947976878, 0.5], [0.4680851063829787, 0.7933526011560694, 0.75], [0.5319148936170213, 0.7615606936416186, 0.25], [0.13829787234042554, 0.7528901734104045, 0.0], [0.09574468085106383, 0.7066473988439306, 0.0], [0.6702127659574468, 0.6358381502890174, 0.25], [0.4680851063829787, 0.5346820809248554, 0.5], [0.09574468085106383, 0.5317919075144508, 0.25], [0.8617021276595744, 0.4812138728323699, 0.5], [0.6382978723404256, 0.47687861271676296, 0.25], [0.6382978723404256, 0.4696531791907514, 0.25], [0.3191489361702128, 0.44364161849710976, 0.5], [1.0, 0.42196531791907516, 0.75], [0.978723404255319, 0.40462427745664736, 1.0], [0.1276595744680851, 0.3973988439306358, 0.75], [0.5638297872340426, 0.39450867052023125, 0.5], [0.6063829787234042, 0.3916184971098266, 0.5], [0.8829787234042553, 0.37138728323699416, 0.0], [0.6382978723404256, 0.3410404624277456, 0.5], [0.648936170212766, 0.32803468208092484, 0.0], [0.6382978723404256, 0.31069364161849705, 0.0], [0.9893617021276595, 0.30635838150289013, 0.5], [0.13829787234042554, 0.29479768786127164, 1.0], [0.9255319148936171, 0.28468208092485553, 0.25], [0.1276595744680851, 0.2817919075144509, 0.0], [0.9574468085106383, 0.2817919075144509, 0.0], [0.40425531914893614, 0.24566473988439302, 0.25], [0.22340425531914893, 0.17919075144508673, 0.75], [0.22340425531914893, 0.157514450867052, 0.25], [0.3723404255319149, 0.157514450867052, 0.0], [0.3404255319148936, 0.11705202312138724, 0.25], [0.648936170212766, 0.10549132947976875, 0.75], [0.0, 0.10549132947976875, 0.75], [0.4680851063829787, 0.05924855491329478, 0.75], [0.4680851063829787, 0.052023121387283267, 0.0], [0.5744680851063829, 0.041907514450867045, 0.75], [0.946808510638298, 0.034682080924855474, 0.0], [0.6170212765957447, 0.033236994219653204, 0.75], [0.1276595744680851, 0.0, 0.75], [0,0,0], [0,0,0]]]
QoS_List = [[0.13, 998, 0.96], [0.64, 992, 0.97], [0.6, 990, 0.97], [0.9, 983, 0.96], [0.52, 982, 0.97], [0.47, 951, 0.99], [0.15, 949, 0.98], [0.15, 945, 0.95], [0.97, 926, 0.95], [0.47, 908, 0.98], [0.38, 887, 0.95], [0.47, 863, 0.97], [0.94, 855, 0.97], [0.59, 833, 0.98], [0.66, 827, 0.96], [0.64, 795, 0.98], [0.15, 746, 0.98], [0.86, 676, 0.95], [0.57, 674, 0.98], [0.97, 639, 0.98], [0.63, 636, 0.96], [0.53, 631, 0.96], [0.63, 613, 0.95], [0.28, 598, 0.99], [0.63, 586, 0.96], [0.12, 581, 0.95], [0.56, 579, 0.97], [0.88, 577, 0.98], [0.92, 563, 0.95], [0.33, 542, 0.97], [0.47, 533, 0.95], [0.84, 521, 0.97], [0.67, 518, 0.97], [0.93, 510, 0.95], [0.95, 503, 0.99], [0.52, 501, 0.99], [0.35, 501, 0.96], [0.16, 476, 0.99], [0.17, 430, 0.96], [0.63, 415, 0.97], [0.03, 415, 0.98], [0.47, 387, 0.98], [0.64, 379, 0.95], [0.41, 379, 0.96], [0.24, 347, 0.96], [0.96, 342, 0.97], [0.16, 335, 0.95], [0.61, 330, 0.98], [0.24, 329, 0.98], [0.12, 306, 0.96], [0,0,0], [0,0,0]]
'''
'''
QoS_List = [[0.94, 800, 0.96], [0.35, 793, 0.97], [0.65, 782, 0.95], [0.69, 777, 0.97], [0.8, 775, 0.96], [0.76, 773, 0.95], [0.25, 769, 0.95], [0.77, 760, 0.96], [0.56, 759, 0.98], [0.19, 749, 0.96], [0.9, 746, 0.98], [0.15, 735, 0.97], [0.06, 735, 0.96], [0.16, 731, 0.98], [0.85, 711, 0.99], [0.3, 705, 0.99], [0.12, 699, 0.99], [0.96, 697, 0.95], [0.8, 696, 0.96], [0.47, 695, 0.97], [0.57, 695, 0.97], [0.15, 686, 0.97], [0.78, 677, 0.99], [0.25, 662, 0.97], [0.27, 660, 0.97], [0.72, 659, 0.97], [0.82, 656, 0.99], [0.75, 645, 0.95], [0.65, 634, 0.97], [0.84, 633, 0.96], [0.76, 632, 0.96], [0.16, 631, 0.97], [0.22, 621, 0.97], [0.08, 620, 0.96], [0.87, 618, 0.97], [0.31, 617, 0.97], [0.38, 610, 0.97], [0.06, 593, 0.97], [0.68, 575, 0.98], [0.99, 575, 0.97], [0.83, 572, 0.99], [0.78, 555, 0.96], [0.32, 550, 0.96], [0.49, 548, 0.98], [0.43, 529, 0.96], [0.04, 528, 0.99], [0.51, 525, 0.96], [0.64, 519, 0.99], [0.06, 514, 0.98], [0.18, 514, 0.95], [0,0,0], [0,0,0]]
Normailized_QoS_list =  [[[0.021052631578947358, 0.9999999999999998, 0.75], [0.6736842105263158, 0.9755244755244756, 0.75], [0.8736842105263157, 0.9370629370629369, 0.5], [0.8526315789473683, 0.9195804195804194, 1.0], [0.9999999999999999, 0.9125874125874123, 0.5], [0.22105263157894736, 0.9055944055944056, 0.0], [0.7578947368421052, 0.8916083916083914, 0.0], [0.22105263157894736, 0.8601398601398602, 0.5], [0.631578947368421, 0.8566433566433564, 1.0], [0.0, 0.8216783216783214, 1.0], [0.9473684210526314, 0.811188811188811, 0.25], [0.7578947368421052, 0.7727272727272727, 0.25], [0.9052631578947368, 0.7727272727272727, 0.75], [0.08421052631578946, 0.7587412587412585, 1.0], [0.2842105263157894, 0.6888111888111885, 0.5], [0.8210526315789473, 0.6678321678321677, 1.0], [0.9684210526315787, 0.6468531468531469, 0.0], [0.24210526315789474, 0.6398601398601398, 0.5], [0.8421052631578947, 0.6363636363636365, 0.25], [0.6842105263157894, 0.6328671328671327, 0.5], [0.3578947368421052, 0.6328671328671327, 0.5], [0.15789473684210525, 0.6013986013986015, 0.25], [0.7789473684210526, 0.5699300699300698, 1.0], [0.12631578947368421, 0.5174825174825173, 0.5], [0.18947368421052632, 0.5104895104895102, 0.5], [0.7789473684210526, 0.5069930069930069, 0.25], [0.27368421052631575, 0.49650349650349646, 1.0], [0.7999999999999999, 0.45804195804195813, 0.25], [0.49473684210526314, 0.41958041958041936, 0.25], [0.11578947368421053, 0.41608391608391604, 0.5], [0.41052631578947363, 0.41258741258741227, 0.25], [0.5473684210526316, 0.40909090909090895, 0.75], [0.14736842105263157, 0.37412587412587395, 0.0], [0.7473684210526316, 0.3706293706293706, 0.0], [0.29473684210526313, 0.36363636363636354, 0.25], [0.11578947368421053, 0.3601398601398602, 0.5], [0.45263157894736833, 0.3356643356643356, 0.5], [0.6421052631578947, 0.27622377622377603, 0.5], [0.021052631578947358, 0.21328671328671311, 0.5], [0.4736842105263157, 0.21328671328671311, 0.75], [0.32631578947368417, 0.2027972027972027, 0.5], [0.042105263157894736, 0.14335664335664333, 0.25], [0.7157894736842104, 0.12587412587412583, 0.5], [0.5578947368421051, 0.11888111888111874, 0.5], [0.6421052631578947, 0.05244755244755228, 0.0], [0.7684210526315789, 0.04895104895104896, 0.25], [0.7999999999999999, 0.038461538461538325, 0.25], [0.831578947368421, 0.01748251748251728, 1.0], [0.021052631578947358, 0.0, 0.25], [0.12631578947368421, 0.0, 0.75], [0,0,0], [0,0,0]]]

'''



n_State_list = len(State_List)
list_a = []
list_r = []
Number = 0

if Number == 0:
    agent = Agent(idx=0,
                  n_input=9,
                  n_output=1,
                  mode='train')
else:
    print('test_Mark')


for sublist in path[0]:
    # Create sublists containing consecutive pairs of elements
    pairs = [[sublist[i], sublist[i+1]] for i in range(len(sublist)-1)]
    path_pair.append(pairs)


#n_time_step = len(QoS_List)
n_time_step = 5

aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}


gamma = 0.999
REWARD_BUFFER = np.empty(shape=n_episode) # reserve a memery space to store the reward record
episode_BUFFER_AVG = []
episode_BUFFER = []
epsilon_buffer=[]
REWARD_BUFF_AVG = []
step_i_buffer = []
n_State_list_c = 0
iterations_loss = 0
w_p, w_b, w_l = 0.0001, 1.2, 0.0001




def Reward_Cal(a, s, step_i, next_State_List_O):
    #print('Ea: ', a, ' Es: ', s)
    #QoS = minmax_scale(QoS_List)
    print('Selected_Path_State: ', a, success_counters)

    # packet loss, bandwidth, latency
    #State_List_O_index = State_List_O[0]
    #print('HHHHHH: ',State_List_O_index)
    #State_List_O_index_action = State_List_O_index[a]
    #print('QQQQQQQQQQQ: ', State_List_O_index_action)
    #print('TTTTTTTTTTTT: ', State_List_O_index_action[0])
    #packet_loss_reward_0 = State_List_O[0][a][0] - QoS_List[0]
    #step_i = step_i - 1
    #print('OOOOOOOOOOOOOOO: ', step_i, State_List_O[0][a][1])


    if (State_List_O[0][a][0] - QoS_List[step_i][0]) > 0:
        packet_loss_reward = -2000
    else:
        packet_loss_reward = (1 - (State_List_O[0][a][0] - QoS_List[step_i][0])) * w_p

    if State_List_O[0][a][1] <= 0:
        bandwidth_reward = -2000
    else:
        bandwidth_utilization = ((((State_List_O[0][a][1]) / constants.State_List_org_2[0][a][1])))
        #print('bandwidth_utilization: ',bandwidth_utilization)
        if bandwidth_utilization >=100:
            bandwidth_reward = -2000
        else:
            #bandwidth_reward = 1/bandwidth_utilization * 100
            bandwidth_reward = bandwidth_utilization



    if (State_List_O[0][a][2] - QoS_List[step_i][2]) >0:
        latency_reward = -1000
    else:
        latency_reward = (1 - (State_List_O[0][a][2] - QoS_List[step_i][2])) * w_l




    sum_variable = [packet_loss_reward, bandwidth_reward, latency_reward]
    r = (sum(sum_variable))/100
    ##
    ##print('oooooooooooooooooo: ', step_i)
    if QoS_List[step_i +1] == [0,0,0]:
        r = 1
        r_next = 1
    else:
        r_next = cal_next_r(next_State_List_O, QoS_List, step_i)



    print('packet_loss_reward: ', packet_loss_reward, 'bandwidth_reward: ', bandwidth_reward, 'latency_reward: ',latency_reward, 'r:', r, 'step_i: ',step_i, 'r_next: ', r_next)


    #print('BBBBB, %f, %f, %f ' %(packet_loss_reward, bandwidth_reward, latency_reward))

    #r = sum(State_List[0][a]) - sum(QoS)
    #print('GGGGGGG:', bandwidth_utilization)
    #State_List_O[0][a][1] = State_List_O[0][a][1] - QoS_List[1]

    #print('GGGG: ', State_List[0][a], State_List_O[0][a], r)
    #print('Qos_G: ', constants.State_List_org_2[0][a][1], State_List_O[0][a][1])
    #print('State_List: ', State_List_O)
    #print('State_List_n: ', State_List)


    #print('RRRRR: ', State_List_O)
    #print('XXXXXXX: ','Action:', a, State_List_O[0])
    #print('GGGGGGG: ', Test_s)
    #print('GGGGGGG: ', r_next)



    if r < 0:
        r = -0.1
        return r, r_next

    if r_next == -0.0004:
        r = -0.1
        return r, r_next

    else:
        return r, r_next
#need to investigate if followping method is still required

def cal_next_r(next_State_List_O, Q_L, i):
    subtracted_r = 0
    #print('Q_L: ', Q_L, 'i:', i, 'len(QoS_List): ', len(QoS_List), 'Q_L[i+1]: ', Q_L[i+1])
    if i == len(QoS_List)-2:
        return 1
    else:
        for x in next_State_List_O[0]:

            #print(x, Q_L[i+1])
            array1 = np.array(x)
            array2 = np.array(Q_L[i+1])
            subtracted_array = np.subtract(array1, array2)
            subtracted = list(subtracted_array)
            #print('GGGGGGGG: ', subtracted)
            if (subtracted[0] <=0 and subtracted[1] >=0 and subtracted[2] <=0):
                subtracted_r += 1
            else:
                subtracted_r += -1

        subtracted_r = subtracted_r/10000
        #print('GGGGGGGG: ', subtracted_r)
        return subtracted_r


        ##print('Ooooooooooooooooooo:', next_State_List_O[0][1])
        ##print('Ooooooooooooooooooo:', Q_L)


for episode_i in range(n_episode):
    # for episode_i in itertools.count():
    done = False
    episode_reward = 0
    QoS_reward_C = 0
    QoS_reward = 0
    step_i = 0
    step_y = 0
    while not done:

    #for step_y in range(n_time_step):
        #print('Step Y',step_y)
        step_y += 1
        #print('XXXXXXXXXXXXXXXX: ', step_i)
        #epsilon = np.interp(episode_i * (n_time_step) + step_y, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])  # interpolation
        '''
        if episode_i < 3000:
            
            a_episode = random.choices(['a','b'], [8,2])[0]
        elif episode_i > 1000:
            a_episode = random.choices(['a', 'b'], [4, 6])[0]
        '''
        #a_episode = random.choices(['a', 'b'], [2, 8])[0]
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * (success_counters) / EPSILON_DECAY)
        #print('aaaa:', a_episode)

        #print('range(n_time_step): ',range(n_time_step))

        random_sample = random.random()
        print( '***Random_sample', random_sample, 'epsilon: ', epsilon, 'episode_i:', episode_i, 'Step_y:',step_y, 'Step_i:',step_i)
        #print('State_List: ', State_List_O)
        #print('TTTTTTT: ', State_List)
        if random_sample <= epsilon: # Choose a random path from current state
        #if a_episode == 'a':
            #print('',)
            action_Value = sample(range(len(State_List[0])),1) #Random select the action from first of the List elements
            a = int(action_Value[0])
            #print('a: ',a )
            for x in State_List[0]:

                x.append(Normailized_QoS_list[0][step_i][0])
                x.append(Normailized_QoS_list[0][step_i][1])
                x.append(Normailized_QoS_list[0][step_i][2])
                x.append(Normailized_QoS_list[0][step_i+1][0])
                x.append(Normailized_QoS_list[0][step_i+1][1])
                x.append(Normailized_QoS_list[0][step_i+1][2])
            s = np.array(State_List[0])
            #print('a:', a)
            #print('s',s )
            #print('Current_State_List_0', State_List_O)
            #s_a = []

            #print('step_i:', step_i, 'State_list:', s)

            for x in State_List[0]:
                x.pop(3)
                x.pop(3)
                x.pop(3)
                x.pop(3)
                x.pop(3)
                x.pop(3)
            #update the environment
            #normal_r(State_List_O, State_List)
            for index, x in enumerate(path_pair):
                for sublist in x:
                    if sublist in path_pair[a]:
                        #print('yes', index)
                        State_List_O[0][index][1] = State_List_O[0][index][1] - QoS_List[step_i][1]
                        break
            #State_List_O[0][a][1] = State_List_O[0][a][1] - QoS_List[step_i][1]
            print('State_List_O: ', State_List_O)
            normal_r(State_List_O, State_List)

            #step_i+=1

            if step_i == len(QoS_List) -1:
                s_ = s
                #print('s_', s_)
                #print('Next_State_List_O: ', State_List_O)

            else:

                #Create next state S_
                for x in State_List[0]:
                    #print('xxx: ', step_i)
                    x.append(Normailized_QoS_list[0][step_i][0])
                    x.append(Normailized_QoS_list[0][step_i][1])
                    x.append(Normailized_QoS_list[0][step_i][2])
                    x.append(Normailized_QoS_list[0][step_i+1][0])
                    x.append(Normailized_QoS_list[0][step_i+1][1])
                    x.append(Normailized_QoS_list[0][step_i+1][2])

                s_ = np.array(State_List[0])
                for x in State_List[0]:
                    x.pop(3)
                    x.pop(3)
                    x.pop(3)
                    x.pop(3)
                    x.pop(3)
                    x.pop(3)
                #print('s_',s_)
                #print('Next_State_List_O: ', State_List_O)
                """
                if step_i == len(QoS_List)-1:
                    #print('Done:', step_i, len(QoS_List))
                    done = True
                else:
                    #print('Done:', step_i, len(QoS_List))
                    done = False
               """
            step_i += 1
        else:
            for x in State_List[0]:
                x.append(Normailized_QoS_list[0][step_i][0])
                x.append(Normailized_QoS_list[0][step_i][1])
                x.append(Normailized_QoS_list[0][step_i][2])
                x.append(Normailized_QoS_list[0][step_i+1][0])
                x.append(Normailized_QoS_list[0][step_i+1][1])
                x.append(Normailized_QoS_list[0][step_i+1][2])

            s = np.array(State_List[0])
            #print('Input_S: ', s)
            a_n = agent.online_net.act(s)

            #print('Output_N_Action: ', a_n)
            a_ToList = a_n.tolist()
            a = int(a_ToList[0][0])
            #print('N_A: ', a)
            #s_ = np.array(State_List[0])

            #prepare Update for next State:

            for x in State_List[0]:
                x.pop(3)
                x.pop(3)
                x.pop(3)
                x.pop(3)
                x.pop(3)
                x.pop(3)
            #print('GGG: ', State_List)

            #update the environment
            for index, x in enumerate(path_pair):
                for sublist in x:
                    if sublist in path_pair[a]:
                        #print('yes', index)
                        State_List_O[0][index][1] = State_List_O[0][index][1] - QoS_List[step_i][1]
                        break
            #State_List_O[0][a][1] = State_List_O[0][a][1] - QoS_List[step_i][1]
            normal_r(State_List_O, State_List)

            #print('TTT: ', State_List, 'step_i: ', step_i)
            #step_i += 1
            if step_i == len(QoS_List) - 1:
                s_ = s
                #print('s_', s_)
                #print('Next_State_List_O: ', State_List_O)
            else:
                #Create next state S_
                #print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
                for x in State_List[0]:
                    x.append(Normailized_QoS_list[0][step_i][0])
                    x.append(Normailized_QoS_list[0][step_i][1])
                    x.append(Normailized_QoS_list[0][step_i][2])
                    x.append(Normailized_QoS_list[0][step_i + 1][0])
                    x.append(Normailized_QoS_list[0][step_i + 1][1])
                    x.append(Normailized_QoS_list[0][step_i + 1][2])

                s_ = np.array(State_List[0])
                #print('AAA: ', s_)
                for x in State_List[0]:
                    x.pop(3)
                    x.pop(3)
                    x.pop(3)
                    x.pop(3)
                    x.pop(3)
                    x.pop(3)
                #print('QQQ: ', State_List)
                #print('PPP:', step_i, len(QoS_List))

                #print('S_: ', s_)
                # Calculate teh next state
                #print('State: ', S, type(S), ' ', 'action_value: ', a, ' ', 'State.Shape: ', S.shape, ' ', 'Next_state', Next_State )
            step_i += 1
            #Calculate the reward

        step_i = step_i - 1
        next_State_List_O = State_List_O

        r_, r_next = Reward_Cal(a, s, step_i, next_State_List_O)
        #print('State_List_O: ', State_List_O, 'action:', a, 'step_i:', step_i, 'Currnet_Stats: ',s, 'Next_state: ', s_)
        #print('State_List_O: ', State_List_O, 'action:', a, 'step_i:', step_i)
        #print('Normal_stat:', State_List)
        #print('DDDDDDDDDD: ', r_, episode_reward)
        #print('OOODone:', step_i, len(QoS_List))

        if r_ > 0:
            #print('xxxxx - episode_reward: ', episode_reward)
            #QoS_reward_C = 0
            episode_no_neg_r = r_
            if step_i == 0:
                QoS_reward = 0.0005 + (r_ + r_next) * 0.1
                #QoS_reward = -0.09 + (r_ + r_next)
                #episode_no_neg_r = QoS_reward
            else:
                #print('%%%%%%%%%%%%%%%%%%%%%%%%%%')
                QoS_reward = 0.0005 + (r_ + r_next) * 0.1
                #QoS_reward = -0.09 + (r_ + r_next)
                #QoS_reward = r_
                #episode_no_neg_r = QoS_reward
                #print('QoS_reward:',QoS_reward, 'action: ', a)
            step_i += 1
            print('xxxxx: ', step_i, len(QoS_List), n_episode, 'reward: ',QoS_reward, episode_reward, step_y, QoS_reward_C,'r_next:', r_next)
            if step_i == len(QoS_List)-1:

                #print('HOOOOO: ', 'State_List_O: ', State_List_O)
                #print('HOOOOO: ', 's: ', s)
                #print('HOOOOO: ', 's_: ', s_)
                QoS_reward = 1
                #QoS_reward = r_ * step_i®
                #QoS_reward = r_
                print('HOOOOO: ', step_i, len(QoS_List), n_episode, 'rewards:',QoS_reward, episode_reward, step_y, QoS_reward_C)
                step_buffer.append(0)
                success_counters += 1
                done = True
                #print('s:', s)
                #print('s_', s_)
                #print('Done', done)
                #s_ = s_.fill(0)
                agent.memo.add_memo((s, a, QoS_reward, done, s_))
            else:
                done = False
                agent.memo.add_memo((s, a, QoS_reward, done, s_))

        if r_ < 0:
            episode_no_neg_r = r_
            done = False
            if step_i == 0:
                QoS_reward = -0.75
                done = False
            else:
                QoS_reward = -0.75
                done = False


            QoS_reward_C += QoS_reward
            #print(":::::::::::",QoS_reward_C)
            if QoS_reward_C <= -3.75:
                QoS_reward = -0.75
                for index, x in enumerate(path_pair):
                    for sublist in x:
                        if sublist in path_pair[a]:
                            #print('yes', index)
                            State_List_O[0][index][1] = State_List_O[0][index][1] + QoS_List[step_i][1]
                            break
                #State_List_O[0][a][1] = State_List_O[0][a][1] + QoS_List[step_i][1]
                x = normal_r(State_List_O, State_List)
                #QoS_reward = -0.01
                done = True
                agent.memo.add_memo((s, a, QoS_reward, done, s_))
                #done = True

            else:
                #print('s_: ',s_)
                for index, x in enumerate(path_pair):
                    for sublist in x:
                        if sublist in path_pair[a]:
                            #print('yes', index)
                            State_List_O[0][index][1] = State_List_O[0][index][1] + QoS_List[step_i][1]
                            break
                #State_List_O[0][a][1] = State_List_O[0][a][1] + QoS_List[step_i][1]
                x = normal_r(State_List_O, State_List)
                print('Revert to Last Org_Stat: ', State_List_O)
                #print('Revert to Last Nor_Stat: ', x)
                #s_ = s_.fill(0)
                QoS_reward = -0.75
                step_i = step_i
                done = False

            #s_.fill(None)
            #print('Revert - Add ER: ', 's:', s , 'a:', a, 'QoS_reward:', QoS_reward, 's_:', s_)
                agent.memo.add_memo((s, a, QoS_reward, done, s_))


        episode_reward += QoS_reward
        QoS_reward = 0

        if done == True:  # if done - break
            # print('XX: ', n_time_step, done)

            r_ = 0
            step_i = 0
            QoS_reward = 0
            QoS_reward_C = 0
            State_List = copy.deepcopy(constants.original_State_List)  # clear Stats
            State_List_O = copy.deepcopy(constants.State_List_org_2)
            REWARD_BUFFER[episode_i] = episode_reward

            # episode_BUFFER.append(episode_i)

            # print('REWARD_BUFFER: ', REWARD_BUFFER)

            # print('YES DONEEEEEEEEEEEEEEEE', episode_reward, step_y)
            # n_time_step = 0
            break

        #print(' Action: ', a, 'Current_State: ', s, ' Next_state: ', s_, ' Reward: ', r_, ' episode_reward: ', episode_reward, 'Done: ', done)
        #batch_s, batch_a, batch_r, batch_done, batch_s_ = agent.memo.sample()

        print(':::::::::::::', agent.memo.count_size(), agent.memo.mem_size())
        count_m = agent.memo.mem_size()
        count_c = agent.memo.count_size()

        if count_c >= 1000:
            batch, weights, tree_idxs = agent.memo.sample()
            #print('batch',batch,'weight', weights,'Tree_idxs', tree_idxs)
            batch_s, batch_a, batch_r, batch_done, batch_s_ = batch
            #print('batch_s',batch_s, 'batch_a',batch_a, 'batch_r',batch_r, 'batch_done',batch_done,'batch_s_', batch_s_)
            target_q_values = agent.target_net(batch_s_)  # Input batched stats into Neural network
            # target_q_values = torch.unsqueeze(target_q_values, 0)

            #
            # print('@@@@@@@@@@@@Target_q_values: ', target_q_values, 'batch_s_: ', batch_s_)
            max_target_q_values = target_q_values.max(dim=1, keepdim=False)[0]  # Choose the maxium value of the output
            #print('%%%%%%%%%%%%%%% target_q_values: ', target_q_values, 'batch_r: ', batch_r, 'batch_s_', batch_s_)
            #print(max_target_q_values)
            # max_q_index = torch.argmax(q_values, dim = 1, keepdim=True)[0]
            batch_r_x = torch.unsqueeze(batch_r, 0)
            q_values = agent.online_net(batch_s)
            #print('GGGGGGG batch_s:', batch_s )
            #print('GGGGGGG q_values:', q_values)

            # print('Q_Values: ',q_values, 'batch_a_Shape: ', batch_a.size(), 'Batch_a: ', batch_a, 'Shape: ', q_values.size() )
            batch_a_x = torch.unsqueeze(batch_a, 0)
            #print('q_values: ', q_values)
            #print('batch_a_x: ', batch_a_x)
            # print('CCCCCCCCCCCCCCCCCC:',batch_a_x, 'batch_a: ', batch_a)
            a_q_values = torch.gather(input=q_values, dim=1, index=batch_a_x)
            # print('######################Action_Q_value: ',a_q_values, 'Action_Q_value_size: ', a_q_values.size(), 'targets: ', targets, 'targets_size: ' , targets.size())
            # print("OOOOOOOOOOOO:", batch_r, batch_r_x)
            #print('max_target_q_values: ',max_target_q_values,  'batch_r_x: ', batch_r_x)
            #if done == True:
                #print('target_q_values: ', target_q_values)
                #print('max_target_q_values :',max_target_q_values )
                #print('batch_r_x :', batch_r_x)
                #print('Done :', batch_done)

            targets =  batch_r_x + agent.GAMMA*(1 - batch_done) * max_target_q_values
            td_error = (abs(a_q_values - targets) ** 2)
            #print('targets: ', targets)
            #print('a_q_values: ', a_q_values)
            #print('batch_r_x: ', batch_r_x)
            #print('td_error.numpy():', td_error.detach().numpy())
            #print('Cccccc:', td_error)
            #print('a_q_values: ',a_q_values)
            #print('targets: ',targets)
            #print('weights: ', weights)
            #loss = nn.functional.smooth_l1_loss(a_q_values, targets) #lose function mse_loss / smooth_l1_loss
            #loss = nn.functional.smooth_l1_loss(a_q_values, targets)
            #
            loss = torch.mean((a_q_values - targets)**2 * weights)
            #step_i_buffer.append(step_i)
            #print('lossXXXXXX: ', loss, 'td_error.numpy():', td_error.detach().numpy(), 'tree_idx:', tree_idxs)
            agent.memo.update_priorities(tree_idxs, td_error.detach().numpy())

            #back propagation
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
            loss_int = loss.detach().numpy()
            loss_buffer.append(loss_int)
            iterations_loss += 1


            #print('PPPPPP: ', State_List_O)
            #print('yyyyyy: ', State_List_org)
            epsilon_buffer.append(epsilon)
            n_State_list_c += 1

        '''
        if done == True :  # if done - break
            #print('XX: ', n_time_step, done)

            r_ = 0
            step_i = 0
            QoS_reward = 0
            QoS_reward_C = 0
            State_List = copy.deepcopy(constants.original_State_List)  # clear Stats
            State_List_O = copy.deepcopy(constants.State_List_org_2)
            REWARD_BUFFER[episode_i] = episode_reward

            #episode_BUFFER.append(episode_i)

            #print('REWARD_BUFFER: ', REWARD_BUFFER)

            #print('YES DONEEEEEEEEEEEEEEEE', episode_reward, step_y)
            #n_time_step = 0
            break
        '''


    #if done == True:
    if episode_i %100== 0:
        #print('KKKKK: ', step_buffer)
        step_buffer_c.append(step_buffer.count(0))
        print('step_buffer_c: ', step_buffer_c)
        step_buffer.clear()

    if episode_i % TARGET_UPDATE_FREQUENCY == 0:
        agent.target_net.load_state_dict(agent.online_net.state_dict())
        #print("AAA")
        print("Episode: {}".format(episode_i))
        print("Avg. Reward: {}".format(np.mean(REWARD_BUFFER[:episode_i])))
        avg_loss_buff.append(np.mean(loss_buffer[:episode_i]))
        episode_BUFFER_AVG.append(episode_i)
        REWARD_BUFF_AVG.append(np.mean(REWARD_BUFFER[:episode_i]))
        episode_i_c = episode_i




        #AVG_REWARD.append(Test_reward)
        #print('YY:', AVG_REWARD)
        #print('xxxxxxxxxxxx')

        torch.save(agent.online_net.state_dict(), PATH)



print('Number Of Success: ', success_counters)
iterations = range(0, n_State_list_c)
n_episode_y = range(0, n_episode)
n_episode_c = n_episode//100
n_episode_i = range(0, n_episode_c)


iterations_loss_i = range(0, iterations_loss)


fig, ax = plt.subplots(2, 3)


ax[0, 0].plot(iterations, epsilon_buffer)
plt.ylabel('epsilon')
plt.xlabel('Episode #')


ax[0, 1].plot(episode_BUFFER_AVG, REWARD_BUFF_AVG)
plt.ylabel('AVG_Reward')
plt.xlabel('Episode #')

ax[1, 0].plot(n_episode_y, REWARD_BUFFER)
plt.ylabel('Reward')
plt.xlabel('Episode #')

ax[1, 1].plot(episode_BUFFER_AVG, avg_loss_buff)
plt.ylabel('Loss')
plt.xlabel('Episode #')

ax[1, 2].plot(n_episode_i, step_buffer_c)
plt.ylabel('Steps')
plt.xlabel('Episode #')

print(step_buffer_c)

plt.show()






