from sklearn.preprocessing import minmax_scale, MinMaxScaler

def normal_r(State_List, State_List_n):
    temp_n = []
    for item in State_List[0]:
        #print('Item:', item)
        temp_n.append(item[1])

    temp_n = minmax_scale(temp_n)
    #print('::', temp_n)
    for idx,  item in enumerate(temp_n):
        State_List_n[0][idx][1] = item
        #print('idx: ',idx, 'item:', item, 'State_List:', State_List[0][idx])
        #print('item:', item)
    #print('Normal_function', State_List_n)
    return State_List_n
    #print('""', State_List_n)self.all_s:  [[[0.91727141 1.         0.         0.4        0.33333333 0.        ]



