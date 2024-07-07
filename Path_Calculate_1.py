# Copyright (C) 2016 Nippon Telegraph and Telephone Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from ryu.lib import hub
from ryu.app import simple_switch_13
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu.topology import event, switches
from ryu.topology.api import get_switch, get_link
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.base import app_manager
from ryu.controller import mac_to_port
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.mac import haddr_to_bin
from ryu.lib.packet import packet
from ryu.lib.packet import arp
from ryu.lib.packet import ethernet
from ryu.lib.packet import ipv4
from ryu.lib.packet import ipv6
from ryu.lib.packet import ether_types
from ryu.lib import mac, ip
from ryu.topology.api import get_switch, get_link
from ryu.app.wsgi import ControllerBase
from ryu.topology import event

from collections import defaultdict
from operator import itemgetter
from ryu.lib import hub
import requests
from ryu.lib.packet import lldp

import os
import random
import time
import copy
from operator import attrgetter

import pickle

from decimal import *


class ProjectController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(ProjectController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.topology_api_app = self
        self.datapath_list = {}
        self.arp_table = {}
        self.switches = []
        self.hosts = {}
        self.multipath_group_ids = {}
        self.group_ids = []
        self.pairs = []
        self.new_pair_list = []
        self.adjacency = defaultdict(dict)
        self.bandwidths = defaultdict(lambda: defaultdict(lambda: DEFAULT_BW))
        self.datapath_Store = []

        self.datapaths = {}
        self.pathStore = []
        self.pathSorted = []
        self.flow_stats = {}
        self.port_stats = {}
        self.stats = {}
        self.sending_echo_request_interval = 0.1
        self.DELAY_DETECTING_INTERVAL = 5
        self.DELAY_SENDING_LLDP = 0.01
        self.echo_latency = {}
        #self.monitor_thread = hub.spawn(self._monitor) # path monitorning
        self.normal_port = []
        self.lldp_topo = {}
        self.monitor_thread = hub.spawn(self._SpawnLLDP)
        #self.monitor_thread_1 = hub.spawn(self._SpawnProbe)
        self.Path_Latency = {}
        self.Latency_pair = {}
        self.Joined_switches = []
        self.Joined_switch_Latency = []
        self.Latency_Path_Store = []
        self.datapath_stat = {}
        self.latency_pair_store = {}
        self.Copy_Latency_pair = {}
        self.final_latecny = {}
        self.counterSend = 0
        self.counterB = 0
        self.counterReceive = 0
        self.switch_id_list_send = {}
        self.switch_id_list_receive = {}
        self.Send_LLDP_Packet_Counter_List = []
        self.packet_loss_Store_switch_id = []
        self.path_loss_sort = {}
        self.final_packet_loss = {}
        self.final_latecny_pl = {}
        self.free_bandwidth = {}
        self.MONITOR_PERIOD = 1
        self.port_speed = {}
        self.port_features = {}
        self.switch_pair_port_list = {}
        self.free_bw = {}
        self.save_lldp_packet = []
        self.save_lldp_packet_sorted = []
        self.link_bw = [[1, 2, 200], [2, 3, 200], [3, 4, 200], [1, 5, 100], [5, 6, 100], [6, 4, 100], [1, 7, 250], [7, 8, 250], [8, 4, 250]]



    # once get the switch pair - flood lldp to get the port that is connected to each other - done
# question, can i send LLDP packet from switch it self? - done
#Design Ping Path
#Example pathSorted:  [[2], [4, 2], [4, 3, 2]]
# 1) prepend orignal into each
# 2) [[1, 2], [1, 4, 2], [1, 4, 3, 2]]
# 3) Break down to each pair? such like [[1,2]],
    # [[1,4], [4,2]],
    # [[1,4], [4,3], [3,2]]
##################################### Latency Start #####################################################################

    def _SpawnLLDP(self):

        while True:
            self.send_echo_request()
            if len(self.datapath_Store) > 1:
                for i in self.datapath_Store:
                    if i != None:
                       self.send_port_stats_request(i)
                       self.send_lldp_packet_sort()
                       #self.send_lldp_packet()
                       self.Latency_calculate()
                       self._path_free_bandwidth()
                       self._normallization()

            hub.sleep(self.DELAY_DETECTING_INTERVAL)


    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        #print('uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu')
        # If you hit this you might want to increase
        # the "miss_send_length" of your switch
        if ev.msg.msg_len < ev.msg.total_len:
            self.logger.debug("packet truncated: only %s of %s bytes",
                              ev.msg.msg_len, ev.msg.total_len)
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        pkt = packet.Packet(data=msg.data)
        pkt_ethernet = pkt.get_protocols(ethernet.ethernet)[0]
        if not pkt_ethernet:
            #print('oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo')
            return

        data_time = time.time()
        pkt_lldp = pkt.get_protocols(lldp.lldp)
        if pkt_lldp and len(pkt_lldp[0].tlvs) == 5:
            #print('mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm')
            #print('Switch: ', datapath.id , 'in_port: ', in_port, pkt_ethernet, 'pkt_lldp', pkt_lldp, 'data_time: ', data_time, 'TVL:',(pkt_lldp[0].tlvs[3].tlv_info))
            # Latency_test = eval((pkt_lldp[0].tlvs[3].tlv_info)) - data_time
            # print('Latency_test: ',Latency_test)

            # print(type(pkt_lldp[0].tlvs[0].chassis_id.decode("utf-8")))
            # pkt_lldp[0].tlvs[0].chassis_id.decode("utf-8") = Peer Send Switch ID that we have recieved
            # pkt_lldp[0].tlvs[1].port_id.decode("utf-8") = Peer Send Port ID that we have recieved
            Peer_Switch_ID = pkt_lldp[0].tlvs[0].chassis_id.decode("utf-8")
            Peer_Port_ID = pkt_lldp[0].tlvs[1].port_id.decode("utf-8")

            if pkt_lldp[0].tlvs[0].chassis_id.decode("utf-8") not in self.switch_id_list_receive:
                self.switch_id_list_receive[pkt_lldp[0].tlvs[0].chassis_id.decode("utf-8")] = {}
                if pkt_lldp[0].tlvs[1].port_id.decode("utf-8") not in self.switch_id_list_receive[
                    pkt_lldp[0].tlvs[0].chassis_id.decode("utf-8")].keys():
                    self.switch_id_list_receive[pkt_lldp[0].tlvs[0].chassis_id.decode("utf-8")][
                        pkt_lldp[0].tlvs[1].port_id.decode("utf-8")] = 0
                    # print('self.switch_id_list_Receive_A:', self.switch_id_list_receive)

            if pkt_lldp[0].tlvs[0].chassis_id.decode("utf-8") in self.switch_id_list_receive:
                if pkt_lldp[0].tlvs[1].port_id.decode("utf-8") not in self.switch_id_list_receive[
                    pkt_lldp[0].tlvs[0].chassis_id.decode("utf-8")].keys():
                    self.switch_id_list_receive[pkt_lldp[0].tlvs[0].chassis_id.decode("utf-8")][
                        pkt_lldp[0].tlvs[1].port_id.decode("utf-8")] = 0
                    # print('self.switch_id_list_Receive_B:', self.switch_id_list_receive)
                if pkt_lldp[0].tlvs[1].port_id.decode("utf-8") in self.switch_id_list_receive[
                    pkt_lldp[0].tlvs[0].chassis_id.decode("utf-8")].keys():
                    self.switch_id_list_receive[pkt_lldp[0].tlvs[0].chassis_id.decode("utf-8")][
                        pkt_lldp[0].tlvs[1].port_id.decode("utf-8")] += 1
                    # print('self.switch_id_list_Receive_C:', self.switch_id_list_receive)

            switch_pair = tuple((int(datapath.id), int(Peer_Switch_ID)))
            switch_pair_sort = tuple(sorted((int(datapath.id), int(Peer_Switch_ID))))
            port_pair = (int(in_port), int(Peer_Port_ID))

            if switch_pair not in self.switch_pair_port_list and switch_pair != switch_pair_sort:
                self.switch_pair_port_list[switch_pair] = port_pair
                #print('GGGGGGGGGGGGGGG: self.switch_pair_port_list: ', self.switch_pair_port_list)

            self.CalculateLatency(datapath, in_port, pkt_ethernet, pkt_lldp, data_time)
            self.CalculatePacketLoss_Sort(datapath.id, in_port, pkt_ethernet, Peer_Switch_ID, Peer_Port_ID)
            switch_pair = ()
            port_pair = ()

        # self.logger.info("packet in %s %s %s %s->%s %s %S", dpid, src, dst, in_port, out_port, ev.msg.data, Packet_P)

    def send_echo_request(self):
        '''
        if len(self.new_pair_list) >0:
            #print('wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww:', self.datapath_Store)
            x = self.new_pair_list[0][0]
            datapath = self.datapath_Store[x]  # switch 1,2,3 etc..
            ofproto = datapath.ofproto
            parser = datapath.ofproto_parser
            data_time = "%.12f" % time.time()
            byte_arr = bytearray(data_time.encode())
            echo_req = parser.OFPEchoRequest(datapath, data=byte_arr)
            datapath.send_msg(echo_req)
        else:
            #print('Noooooooooooooooooooooooo')
            return
        '''
        for i in range(len(self.new_pair_list)):
            ++i
            #hub.sleep(self.sending_echo_request_interval)
            for e in range(len(self.new_pair_list[i])):
                x = self.new_pair_list[i][e]
                datapath = self.datapath_Store[x] # switch 1,2,3 etc..
                ofproto = datapath.ofproto
                ++e
                parser = datapath.ofproto_parser
                data_time = "%.12f" % time.time()
                byte_arr = bytearray(data_time.encode())
                echo_req = parser.OFPEchoRequest(datapath, data=byte_arr)
                datapath.send_msg(echo_req)
                #print('data_time: ',data_time, 'byte_arr:',byte_arr)
                hub.sleep(self.sending_echo_request_interval)



    @set_ev_cls(ofp_event.EventOFPEchoReply, MAIN_DISPATCHER)
    def echo_reply_handler(self, ev):
        """
            Handle the echo reply msg, and get the latency of link.
        """
        now_timestamp = time.time()
        try:
            #print('eval(ev.msg.data): ',eval(ev.msg.data), 'now_timestamp: ', now_timestamp)
            latency = now_timestamp - eval(ev.msg.data)
            #print('latency: ',latency)
            self.echo_latency[ev.msg.datapath.id] = latency
            #print('######################################self.echo_latency: ', self.echo_latency)
        except:
            return

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        self.datapath_Store.append(datapath)
        if self.datapath_Store[0] != None:
            self.datapath_Store.insert(0, None)
        #print('datapath_Store:', self.datapath_Store)
        #print('DatapathAAAAAAAAAAAAAAAAAAA: ', datapath)
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        #self.send_port_stats_request(datapath)
        self.add_flow(datapath, 0, match, actions)
    """
    1) Send Port stats request by calling method - end_port_stats_request
    2) The calling method handled by port_stats_reply_handler
    
    """



    def send_port_stats_request(self, datapath):
        #print('datapathXXXXXXXX', datapath)
        """
        #datapathXXXXXXXX <ryu.controller.controller.Datapath object at 0x7f07bd7b2c70>
        #datapathXXXXXXXX <ryu.controller.controller.Datapath object at 0x7f07bd7b2550>
        #datapathXXXXXXXX <ryu.controller.controller.Datapath object at 0x7f07bd7b2a30>
        #datapathXXXXXXXX <ryu.controller.controller.Datapath object at 0x7f07bd746910>
        """

        ofp = datapath.ofproto
        ofp_parser = datapath.ofproto_parser
        req = ofp_parser.OFPPortDescStatsRequest(datapath, 0, ofp.OFPP_ANY)
        datapath.send_msg(req)

        req = ofp_parser.OFPPortStatsRequest(datapath, 0, ofp.OFPP_ANY)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPPortDescStatsReply, MAIN_DISPATCHER)
    def port_descStats_reply_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        # LLDP packet to controller
        match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_LLDP)
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER)]
        self.add_flow(datapath, 0, match, actions)
        dpid = msg.datapath.id

        if datapath.id not in self.Joined_switch_Latency:
            self.Joined_switch_Latency.append(datapath.id)

        #print('Joined_switch: ', self.Joined_switch_Latency)
        #print('Joined_switches: ', self.Joined_switches)

        for stat in ev.msg.body:
            if stat.port_no < ofproto.OFPP_MAX:
                self.normal_port.append(stat.port_no)
            if (datapath.id not in self.datapath_stat):
                self.datapath_stat[datapath.id] = []
            if [datapath, stat.port_no, stat.hw_addr] not in self.datapath_stat.get(datapath.id):
                self.datapath_stat[datapath.id].append([datapath, stat.port_no, stat.hw_addr])
            #print('DDDDDDDd: ', stat.port_no, ' ', stat.curr_speed, ' ', stat.max_speed)

            self.port_features.setdefault(dpid, {})
            port_feature = (stat.curr_speed)
            self.port_features[dpid][stat.port_no] = port_feature
            #print('self.port_features: ', self.port_features)
                # Need to Need Unique dtapath id into the Joined_switch and make sure the length == number of the switches

        if len(self.Joined_switch_Latency) == len(self.Joined_switches):
            for key in self.Latency_pair.keys():
                for key_value in self.Latency_pair[key]:
                    for key_pairs in key_value:
                        #print('key_pairs: ', key_pairs)
                        #print('GGGGGGGGGGGGGGGGGGGGGgself.datapath_stat', self.datapath_stat)
                        if key_pairs not in self.datapath_stat.keys():
                            return
                        else:
                            for stat_value in self.datapath_stat[key_pairs]:
                                #print('stat_value: ',stat_value[0], stat_value[1], stat_value[2])
                                #self.send_lldp_packet(datapath, stat.port_no, stat.hw_addr)
                                self.save_lldp_packet.append([stat_value[0], stat_value[1], stat_value[2]])
                                #self.send_lldp_packet(stat_value[0], stat_value[1], stat_value[2])
                    # print('normal_port', self.normal_port, 'ofproto.OFPP_MAX: ', ofproto.OFPP_MAX, 'port_no: ', stat.port_no) # normal_port [1, 2, 1, 2, 1, 2]

                    #print("Datapath %s, Datapath.id %s Stat.Port_no %s, stat.hw_addr %s" % (datapath, datapath.id, stat.port_no, stat.hw_addr))
        else:
            return

        if len(self.normal_port) == 2:
            #Port A to port B
            #print('Len == 2')
            match = parser.OFPMatch(in_port=self.normal_port[0])
            actions = [parser.OFPActionOutput(self.normal_port[1])]
            self.add_flow(datapath, 0 , match, actions)

            #Port B to port A
            match = parser.OFPMatch(in_port=self.normal_port[1])
            actions = [parser.OFPActionOutput(self.normal_port[0])]
            self.add_flow(datapath, 0, match, actions)

        self.normal_port = []

    def send_lldp_packet_sort(self):
        #print('ggggggggggggggggggggggggggggggggggggggggggggggggggggggg')
        for item in self.save_lldp_packet:
            #print('Item:', item)
            if item not in self.save_lldp_packet_sorted:
                self.save_lldp_packet_sorted.append(item)

        for items in self.save_lldp_packet_sorted:
                datapath = items[0]
                port_no = items[1]
                hw_addr = items[2]
                self.send_lldp_packet(datapath, port_no, hw_addr)
                hub.sleep(self.DELAY_SENDING_LLDP)







    def send_lldp_packet(self, datapath, port_no, hw_addr):
        #print('save_lldp_packet: ', self.save_lldp_packet)
        PORT_ID_STR = '!I'  # uint32_t

        #datapath = data[0]
        #port_no = data[1]
        #hw_addr = data[2]

        ofp = datapath.ofproto
        pkt = packet.Packet()
        data_time = "%.12f" % time.time()
        byte_arr = bytearray(data_time.encode())
        pkt.add_protocol(ethernet.ethernet(ethertype=ether_types.ETH_TYPE_LLDP, src=hw_addr, dst=lldp.LLDP_MAC_NEAREST_BRIDGE))
        tlv_chassis_id = lldp.ChassisID(subtype=lldp.ChassisID.SUB_LOCALLY_ASSIGNED,chassis_id=str(datapath.id).encode())
        tlv_port_id = lldp.PortID(subtype=lldp.PortID.SUB_LOCALLY_ASSIGNED, port_id=str(port_no).encode())
        tlv_ttl = lldp.TTL(ttl=10)
        tlv_system_description = lldp.SystemDescription(system_description=byte_arr)
        tlv_end = lldp.End()
        tlvs = (tlv_chassis_id, tlv_port_id, tlv_ttl, tlv_system_description, tlv_end)
        pkt.add_protocol(lldp.lldp(tlvs))
        pkt.serialize()
        data = pkt.data
        parser = datapath.ofproto_parser
        actions = [parser.OFPActionOutput(port=port_no)]
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=ofp.OFP_NO_BUFFER, in_port=ofp.OFPP_CONTROLLER,
                                  actions=actions, data=data)
        #print('TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT', out)
        datapath.send_msg(out)

        if datapath.id not in self.switch_id_list_send:
            self.switch_id_list_send[datapath.id] = {}
            #print('Switch ID: ', datapath.id, 'self.switch_id_list: ', self.switch_id_list_send, 'port_no: ', port_no)
            if port_no not in self.switch_id_list_send[datapath.id].keys():
                self.switch_id_list_send[datapath.id][port_no] = 0
                #print('self.switch_id_list_A:', self.switch_id_list_send)

        if datapath.id in self.switch_id_list_send:
            if port_no not in self.switch_id_list_send[datapath.id].keys():
                self.switch_id_list_send[datapath.id][port_no] = 0
                #print('self.switch_id_list_B:', self.switch_id_list_send)
            if port_no in self.switch_id_list_send[datapath.id].keys():
                self.switch_id_list_send[datapath.id][port_no] +=1
                #print('self.switch_id_list_C:', self.switch_id_list_send)





    def CalculateLatency(self, datapath, in_port, pkt_ethernet, pkt_lldp, data_time):
        pair_total_latecny = 0
        total_pair_total_latecny_avg = 0
        self.lldp_topo.setdefault(int(datapath.id), {})
        self.lldp_topo[datapath.id].setdefault(int(in_port),
                                               [pkt_lldp[0].tlvs[0].chassis_id, list(pkt_lldp[0].tlvs[1].port_id)])


        #print('self.normal_port', self.normal_port)

        if len(self.normal_port) != 0 or len(self.normal_port) == 0:

            if len(self.echo_latency) !=0:
                #print('oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo')
                #print('self.normal_port[0]: ', self.normal_port[0], self.normal_port[1])
                #src_controller_latency = self.echo_latency[self.normal_port[0]]
                #dst_controller_latency = self.echo_latency[self.normal_port[1]]
                #print('pkt_lldp[0].tlvs[3].tlv_info', pkt_lldp[0].tlvs[3].tlv_info, 'Destination Switch ID: ', datapath.id, 'Destination Time: ', data_time, 'Source Switch ID: ', pkt_lldp[0].tlvs[0].chassis_id.decode("utf-8"))
                latency_pair = sorted([int(pkt_lldp[0].tlvs[0].chassis_id.decode("utf-8")), datapath.id])

                #latency = eval((pkt_lldp[0].tlvs[3].tlv_info)) - data_time - dst_controller_latency - src_controller_latency
                latency = data_time - eval((pkt_lldp[0].tlvs[3].tlv_info)) - self.echo_latency[latency_pair[1]] - self.echo_latency[latency_pair[0]]
                #print('Send_Switch: ', int(pkt_lldp[0].tlvs[0].chassis_id.decode("utf-8")),'Send_time: ', eval((pkt_lldp[0].tlvs[3].tlv_info)), 'Received_switch: ', datapath.id, 'Receive Time: ',  data_time, 'Latency: ', latency, 'latency_pair:',latency_pair)
                #print('DDDDDDDDDDDDDDDDDDDDDDD: ', self.Latency_pair, 'Latency: ', latency, 'self.Copy_Latency_pair: ', self.Copy_Latency_pair)
                #print('self.Copy_Latency_pair.items(): ', self.Copy_Latency_pair.items())
                for idx, pairs in self.Copy_Latency_pair.items():
                    for pairs_path in pairs:
                        if [pairs_path[0], pairs_path[1]] == latency_pair and len(pairs_path) == 2:
                            pairs_path.append(latency)

                        if [pairs_path[1], pairs_path[0]] == latency_pair and len(pairs_path) == 2:
                            pairs_path.append(latency)

                        if [pairs_path[0], pairs_path[1]]  == latency_pair and len(pairs_path) == 3:
                            pairs_path[2] = latency

                        if [pairs_path[1], pairs_path[0]]  == latency_pair and len(pairs_path) == 3:
                            pairs_path[2] = latency

                #print('latency: ', latency, 'Latency_pair: ', latency_pair,)
                #print('************self.Copy_Latency_pair: ', self.Copy_Latency_pair)
                #print('latency_pair_store: ', self.latency_pair_store)

                self.normal_port = []
            else:
                print('Waiting for controller latency to be calculated')
        else:
            print('Waiting for switch to be added to the list')
        #
        #print(self.lldp_topo.get(datapath.id, {}))
        #print('GGGGGGGgg', self.lldp_topo.setdefault([pkt_lldp]))

    def Latency_calculate(self):
        latency = 0
        self.latency_pair_store = copy.deepcopy(self.Copy_Latency_pair)
        #print('##########latency_pair_store: ', self.latency_pair_store)
        for idx, item in self.latency_pair_store.items():
            for items in item:
                if len(items) == 3:
                    while len(items) != 1:
                        items.pop(0)
                    latency = latency + items[0]
                else:
                    print('waiting for LLDP')
            self.final_latecny[idx] = latency
            latency = 0
        #print('self.latency_pair_store', self.latency_pair_store )
        #print('self.final_latecny', self.final_latecny)
##################################### Latency End #####################################################################

##################################### Loss Start #####################################################################

    def CalculatePacketLoss_Sort(self, datapath, in_port, pkt_ethernet, Peer_Switch_ID, Peer_Port_ID):
        #print('self.switch_id_list_send:', self.switch_id_list_send)
        #print('self.switch_id_list_Receive:', self.switch_id_list_receive)
        #print('Dest_sw_ID: ', datapath, 'In_port: ', in_port, 'Org_Switch: ',Peer_Switch_ID, 'Org_Port_ID: ',Peer_Port_ID)
        p_l_s = {}

        d_s = datapath
        d_p = in_port
        t_p_l_p = []
        t_p_l_d = 0
        t_p_l_s = 0
        t_p_l_f = 0
        s_s = Peer_Switch_ID
        s_p = Peer_Port_ID
        sum_pl = 0


        for r_s_id in self.switch_id_list_receive:
            for s_s_id in self.switch_id_list_send:
                if str(r_s_id) == str(s_s_id):
                    for keys_r in self.switch_id_list_receive[r_s_id].keys():
                        for keys_s in self.switch_id_list_send[s_s_id].keys():
                            if str(keys_s) == str(keys_r):
                                #print('S_ID: ', s_s_id, 'S_port: ', keys_s, 'R_ID: ', r_s_id, 'R_port: ', keys_r)
                                #print('S_packet: ', self.switch_id_list_send[s_s_id][keys_s], 'R_packet: ', self.switch_id_list_receive[r_s_id][keys_r])
                                #print('S_packet: ', self.switch_id_list_send, 'R_packet: ',self.switch_id_list_receive)
                                p_s = self.switch_id_list_send[s_s_id][keys_s]
                                p_r = self.switch_id_list_receive[r_s_id][keys_r]
                                p_l = (p_s - p_r)/p_s
                                if r_s_id not in p_l_s:
                                    p_l_s[r_s_id] = {}
                                    if keys_r not in p_l_s[r_s_id].keys():
                                        p_l_s[r_s_id][keys_r] = p_l
                                if r_s_id in p_l_s:
                                    if keys_r not in p_l_s[r_s_id].keys():
                                        p_l_s[r_s_id][keys_r] = p_l
        #print('Switch: ', datapath.id , 'in_port: ', in_port, pkt_ethernet, 'pkt_lldp', pkt_lldp, 'data_time: ', data_time)

        for key_d in p_l_s:
            if str(d_s) == str(key_d):
                for keys_c in p_l_s[key_d].keys():
                    if str(d_p) == str(keys_c):
                        t_p_l_p.append(key_d)
                        t_p_l_d = p_l_s[key_d][keys_c]




        for key_s in p_l_s:
            if str(s_s) == str(key_s):
                for key_c in p_l_s[key_s].keys():
                    if str(s_p) == str(key_c):
                        t_p_l_p.append(key_s)
                        t_p_l_s = p_l_s[key_s][key_c]

        #print('CCCCC: ', t_p_l_p, t_p_l_d, t_p_l_s)

        if t_p_l_s == 0:
            t_p_l_f = t_p_l_d
        if t_p_l_d == 0:
            t_p_l_f = t_p_l_s

        if t_p_l_s and t_p_l_d != 0:
            t_p_l_f = (t_p_l_s + t_p_l_d)/2

        t_p_l_p = sorted(t_p_l_p)
        #print('ggggg: ', t_p_l_p, t_p_l_f)

        if len(t_p_l_p) != 1:

            if t_p_l_p not in self.path_loss_sort.values():
                self.path_loss_sort[t_p_l_f] = t_p_l_p
                #print('path_loss_sortxxxxx: ', self.path_loss_sort)




            if t_p_l_p in self.path_loss_sort.values():
                for key, value in list(self.path_loss_sort.items()):
                    #print('key: ', key, 't_p_l_f: ',  t_p_l_f)
                    if value == t_p_l_p:
                        if t_p_l_f != key:
                            self.path_loss_sort[t_p_l_f] = self.path_loss_sort[key]
                            #print('path_loss_sortYYYYYYYy: ', self.path_loss_sort)
                            del self.path_loss_sort[key]


        for idx, items in self.final_latecny_pl.items():

            for item in items:
                for idx_pl, items_pl in self.path_loss_sort.items():
                    if item == list(map(int, items_pl)):
                        sum_pl+=idx_pl
                        for idf, items_f in self.final_packet_loss.items():
                            if idx == idf:
                                sum_pl = round(sum_pl, 2)
                                #print('uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu', sum_pl, type(sum_pl))
                                self.final_packet_loss[idf] = sum_pl
            sum_pl = 0

        #print('path_loss_sort: ', self.path_loss_sort)
        #print('self.final_packet_loss: ', self.final_packet_loss)

        #DDDD = {0: [[1, 2], [2, 3]], 1: [[1, 4], [4, 3]], 2: [[1, 2], [2, 4], [4, 3]], 3: [[1, 4], [4, 2], [2, 3]]}
        #path_loss_sort = {0.24054783950617284: ['1', '4'], 0.12721836419753085: ['2', '4'],0.09109760802469136: ['1', '2'], 0.14978780864197533: ['3', '4'],0.2004243827160494: ['2', '3']}

##################################### Loss End #####################################################################

##################################### Normallization Function Start #####################################################################
    def _normallization(self):
        x_rounded_final_latecny = {key: round(value, 3) for key, value in self.final_latecny.items()}
        print('self.final_packet_loss: ', self.final_packet_loss)
        print('self.free_bw: ', self.free_bw)
        print('self.final_latecny', x_rounded_final_latecny)

        with open("state_list.txt", "wb") as f:
            #print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            pickle.dump(self.final_packet_loss, f)
            pickle.dump(self.free_bw, f)
            pickle.dump(self.final_latecny, f)
            f.close()



##################################### Normallization Function End #####################################################################

##################################### Add_Flow #####################################################################

##################################### Add_End #####################################################################



    @set_ev_cls(ofp_event.EventOFPStateChange,
                [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        #print('datapath:', datapath.id)
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.debug('register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        # print "Adding flow ", match, actions
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, command=ofproto.OFPFC_ADD, match=match, instructions=inst)
        datapath.send_msg(mod)


    """
    def _monitor(self):
        while True:
            for dp in self.datapaths.values():
                self.request_stats(dp)
            hub.sleep(5)

    def request_stats(self, datapath):
        self.logger.debug('send stats request: %016x', datapath.id)
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

 """
##################################### Bandwidth Start #####################################################################
    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        #self.flow_stats = {}
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        self.stats['port'] = {}
        self.stats['port'][dpid] = body
        self.flow_stats.setdefault(dpid, {})
        self.free_bandwidth.setdefault(dpid, {})

        for stat in sorted(body, key=attrgetter('port_no')):
            port_no = stat.port_no
            if port_no != ofproto_v1_3.OFPP_LOCAL:
                key = (dpid, port_no)

                value = (stat.tx_bytes, stat.rx_bytes, stat.rx_errors,
                         stat.duration_sec, stat.duration_nsec)

                self._save_stats(self.port_stats, key, value, 5)
                #print('Speed: ', self.port_stats)

                pre = 0
                period = self.MONITOR_PERIOD
                tmp = self.port_stats[key]
                #print('****tmp: ', tmp)
                if len(tmp) > 1:
                    pre = tmp[-2][0] + tmp[-2][1]
                    #print('pre: ', pre, tmp[-2][0], tmp[-2][1])
                    period = self._get_period(tmp[-1][3], tmp[-1][4], tmp[-2][3], tmp[-2][4])


                speed = self._get_speed(self.port_stats[key][-1][0] + self.port_stats[key][-1][1], pre, period)


                #print('############33 Speed: ', speed)

                self._save_stats(self.port_speed, key, speed, 5)
                self._save_freebandwidth(dpid, port_no, speed)
                #print('freebandwidth: ',self.free_bandwidth)

                #rint('Speed: ', self.port_speed)

    def _path_free_bandwidth(self):
        bw = self.free_bandwidth
        tmp = []
        dict = self.Latency_pair
        for keys in dict.keys():
            #print('keys:::::::::::;', keys)
            for values in dict[keys]:
                #print('values: ', values)
                for keys_sp, values_sp in self.switch_pair_port_list.items():
                    if keys_sp == values or sorted(keys_sp) == values:
                        #print('Keys_sp', keys_sp, 'values_sp', values_sp)
                        v1 = bw[keys_sp[0]][values_sp[0]]
                        v2 = bw[keys_sp[1]][values_sp[1]]
                        tmp.append(v1)
                        tmp.append(v2)
            #print('oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo',self.free_bandwidth)
            if len(tmp) != 0:
                self.free_bw[keys].append(min(tmp))
                tmp=[]
            if len(self.free_bw[keys]) > 1:
                self.free_bw[keys].pop(0)

        #print('Path Available Bandwidth: ', self.free_bw)

    def _save_stats(self, _dict, key, value, length):
        if key not in _dict:
            _dict[key] = []
        _dict[key].append(value)

        if len(_dict[key]) > length:
            _dict[key].pop(0)

    def _get_period (self, n_sec, n_nsec, p_sec, p_nsec):
        #print('DDDDDDDDDDDDDDDDDDDDDd: ', self._get_time(n_sec, n_nsec) - self._get_time(p_sec, p_nsec))
        return self._get_time(n_sec, n_nsec) - self._get_time(p_sec, p_nsec)

    def _get_time(self, sec, nsec):
        return sec + nsec / (10 ** 9)

    def _get_speed(self, now, pre, period):
        if period:
            return (now - pre) / (period)
        else:
            return 0

    def _save_freebandwidth(self, dpid, port_no, speed):
        temp_pair = [dpid, port_no]
        if temp_pair in [[item[0], item[1]] for item in self.link_bw]:
            for item in self.link_bw:
                if temp_pair == [item[0], item[1]]:
                    tmp = item[2]
                    port_state = tmp * (10**3)
        if temp_pair in [[item[1], item[0]] for item in self.link_bw]:
            for item in self.link_bw:
                if temp_pair == [item[1], item[0]]:
                    tmp = item[2]
                    port_state = tmp * (10**3)
        else:
            port_state = self.port_features.get(dpid).get(port_no)
        #dpid = 3
        #port = 1, 2, 3, 4, 5

        #print('port_state: ', port_state,  ' self.switch_pair_port_list: ',self.switch_pair_port_list)
        if port_state:
            capacity = port_state
            #print('CCCCCCCCCCCCCCCCCCCCCCCcc: ', speed, capacity,dpid)
            #print('UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU ', port_no, self.free_bandwidth)
            curr_bw = self._get_free_bw(capacity, speed)
            self.free_bandwidth[dpid].setdefault(port_no, None)
            self.free_bandwidth[dpid][port_no] = curr_bw
            #print('ooooooooooooooooooooooooooooo: ', self.free_bandwidth)
        else:
            self.logger.info('Fail in getting port state')

    def _get_free_bw(self, capacity, speed):
        return max(capacity / 10**3 - speed * 8 / 10**6, 0)
##################################### Bandwidth End #####################################################################


    """
    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        self.logger.info('datapath         '
                         'in-port  eth-dst           '
                         'out-port packets  bytes')
        self.logger.info('---------------- '
                         '-------- ----------------- '
                         '-------- -------- --------')
        for stat in sorted([flow for flow in body if flow.priority == 1],
                           key=lambda flow: (flow.match['in_port'],
                                             flow.match['eth_dst'])):
            self.logger.info('%016x %8x %17s %8x %8d %8d',
                             ev.msg.datapath.id,
                             stat.match['in_port'], stat.match['eth_dst'],
                             stat.instructions[0].actions[0].port,
                             stat.packet_count, stat.byte_count)
            print('+++++++++++++++++++++++++++++++++++++++++++++', self.flow_stats[dpid][key])

"""
    @set_ev_cls(event.EventSwitchEnter)
    def switch_enter_handler(self, ev):
        switch = ev.switch.dp
        ofp_parser = switch.ofproto_parser

        if switch.id not in self.switches:
            self.switches.append(switch.id)
            self.datapath_list[switch.id] = switch

            # Request port/link descriptions, useful for obtaining bandwidth
            req = ofp_parser.OFPPortDescStatsRequest(switch)
            switch.send_msg(req)

        switch_list = get_switch(self.topology_api_app, None)
        switches = [switch.dp.id for switch in switch_list]
        #print('switches: ', switches)
        links_list = get_link(self.topology_api_app, None)
        links = [(link.src.dpid, link.dst.dpid, {'port': link.src.port_no}) for link in links_list]
        #print('links:', links)

    @set_ev_cls(event.EventLinkAdd)
    def event_link_add_handler(self, ev):
        src_sw = ev.link.src.dpid
        dst_sw = ev.link.dst.dpid
        src_pn = ev.link.src.port_no
        dst_pn = ev.link.dst.port_no
        #print('src_sw: ',src_sw)
        #print('dst_sw: ',dst_sw)

        if (src_sw, dst_sw) not in self.pairs:
            self.pairs.append([src_sw, dst_sw])

        if (src_sw) not in self.Joined_switches:
            self.Joined_switches.append(src_sw)
        #print('pairs: ', self.pairs)
        #print('self.Joined_switches',self.Joined_switches)
        #Calculate the list of paths
        self.Sorting_pairs(self.pairs)
        #print('src_pn: ', src_pn)
        #print('dst_pn: ', dst_pn)

    def Sorting_pairs(self,pairs):
        pairs_list = []

        source = []
        destination = []
        A_Z_dict = {}
        NodeConnectList = []

        #print('pairsx: ', self.pairs)
        for i in self.pairs:
            pairs_Sorted_Element = sorted(i)
            pairs_list.append(pairs_Sorted_Element)

        pairs_Sorted = sorted(pairs_list)
        #print('pairs_Sorted', pairs_Sorted)
        # remove duplicate entry
        for Single_pairs in pairs_Sorted:
            if Single_pairs not in self.new_pair_list:
                self.new_pair_list.append(Single_pairs)
                print('new_pair_list', self.new_pair_list)

        for i in self.new_pair_list:
            source.append(i[0])
            destination.append(i[1])

        A = source + destination

        Z = destination + source

        A_key = sorted(set(A))
        Z.insert(0, None)
        A.insert(0, None)
        A_key.insert(0, None)
        #print('A_key', A_key)

        for i in range(len(A_key)):
            A_Z_dict[A_key[i]] = []

        for i in range(len(A)):
            if Z[i] not in A_Z_dict[A[i]]:
                A_Z_dict[A[i]].append(Z[i])

        for i in A_Z_dict.values():
            NodeConnectList.append(i)


        #print('self.source', source)
        #print('self.destination', destination)
        #print('NodeConnectList: ', NodeConnectList)
        #Start = Switch 1, end = Switch 2

        start = 1
        end = []
        end_switch = 4
        if start in source and (end_switch in destination or end_switch in source):
            end = [end_switch]
            #print('end[0]++++++++++++++==', NodeConnectList[end[0]])
            self.countPaths(start, end, A_Z_dict, NodeConnectList)
        else:
            print("No source or destination switch in the Topology")

    def countPaths(self, start, end, A_Z_dict, NodeConnectList):
        pathStore = []
        #print('A_Z_dict: ', A_Z_dict)
        #print('start: ', start, end[0], NodeConnectList, len(A_Z_dict), 'NodeConnectList: ', NodeConnectList[end[0]])
        visited = [False] * len(A_Z_dict)
        visited.insert(0, None)
        #print('visited: ',visited)
        pathCount = [0]
        path = []
        self.countPathsUtil(start, end, visited, pathCount, path, NodeConnectList, pathStore, start)
        pathnumber = [int(i) for i in pathCount]
        #print(pathnumber[0])
        pathSorted = sorted(pathStore, key=len)
        pathSorted = pathSorted[0:3]
        self.Latency_pair_calcalate(pathSorted)
        #print('#############################pathSorted: ', type(pathSorted))
        print('#############################pathSorted: ',pathSorted)


    def countPathsUtil(self, u, end, visited, pathCount, path, NodeConnectList, pathStore, start):

        visited[u] = True
        # print('NodeConnectList[0]: ', NodeConnectList[0])
        # print('NodeConnectList[1]: ', NodeConnectList[1])
        # print('U: ', u, end[0], 'Len: ', NodeConnectList[u])
        # print('WTF: ', u-1, end[0])
        if (u == end[0]):
            # print('WTF? ',u, 2)
            # print('xxxx: ', pathCount)
            pathCount[0] += 1
            print('Success: ', pathCount)
            pathx = path.copy()
            pathx.insert(0, 1)
            pathStore.append(pathx)
        else:
            i = 0
            #print('UUUUUUUUUUUU:', u)

            if len(NodeConnectList) > u:
                if len(visited) > u:
                    #print('NodeConnectList:', NodeConnectList, 'len(NodeConnectList): ', len(NodeConnectList), 'len(visited): ', len(visited), 'U:', u)
                    #print('NodeConnectList[U]:', NodeConnectList[u])
                    while i <= len(NodeConnectList[u])-1:
                        #print('len(NodeConnectList[u])', u, 'NodeConnectList[u][i]: ',NodeConnectList[u][i], 'Len(Visit): ', len(visited), 'Visited: ', visited)
                        if NodeConnectList[u][i] >= len(visited):
                            #print('laaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
                            return
                        if (not visited[NodeConnectList[u][i]]):
                            # print('NodeConnectList: ', visited[NodeConnectList[u][i]], 'U: ', u)'
                            #print('NodeConnectListxxxx: ', NodeConnectList[u][i])
                            path.append(NodeConnectList[u][i])
                            self.countPathsUtil(NodeConnectList[u][i], end, visited, pathCount, path, NodeConnectList, pathStore, start)
                            path.pop()
                        i += 1
            else:
                print('Waiting for Node')
        visited[u] = False
        return pathStore

    def Latency_pair_calcalate(self, pathSorted):
        i = 1
        pair = []
        for idx, x in enumerate(pathSorted):
            if len(x) == 2:
                self.Latency_pair[idx] = [x]
            else:
                self.Latency_pair[idx] = []
                #print('b: ', pathSorted)
                xdx = 0
                for xdx, z in enumerate(x):
                    # print('xdxx:', xdx)
                    if (xdx > 1):
                        pair.append(x[xdx - 1])
                        pair.append(z)
                        # print('xdaaaax:', x[xdx-1], z, pair)
                    else:
                        pair.append(z)
                    # print('pair: ', pair)
                    if len(pair) == 2:
                        self.Latency_pair[idx].append(pair)
                        print('Latency_pair: ', self.Latency_pair)
                        pair = []
        self.Copy_Latency_pair = copy.deepcopy(self.Latency_pair)
        self.final_latecny = copy.deepcopy(self.Latency_pair)
        for idx, item in self.final_latecny.items():
            item.clear()
        self.final_packet_loss = copy.deepcopy(self.final_latecny)
        for idx, item in self.final_packet_loss.items():
            item.clear()
        self.final_latecny_pl = copy.deepcopy(self.Latency_pair)
        for idx, items in self.final_latecny_pl.items():
            for item in items:
                self.final_latecny_pl[idx][items.index(item)] = sorted(self.final_latecny_pl[idx][items.index(item)])

        self.free_bw = copy.deepcopy(self.final_latecny)
        for idx, item in self.free_bw.items():
            item.clear()
        #print('*********************************************: ', self.final_latecny_pl)



