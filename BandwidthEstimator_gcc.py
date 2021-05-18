import collections
import numpy as np

# 过程中需要用到的一些常量
kMinNumDeltas = 60
threshold_gain_ = 4
kBurstIntervalMs = 5
kTrendlineWindowSize = 20  # 用于求解趋势斜率的样本个数，每个样本为包组的单向延时梯度
kTrendlineSmoothingCoeff = 0.9
kOverUsingTimeThreshold = 10
kMaxAdaptOffsetMs = 15.0
eta = 1.08  # increasing coeffience for AIMD
alpha = 0.85  # decreasing coeffience for AIMD
k_up_ = 0.0087
k_down_ = 0.039

class Estimator(object):
    def __init__(self):
        # ----- 包组时间相关 -----
        self.packets_list = []  # 记录当前时间间隔内收到的所有包
        self.packet_group = []
        self.first_group_complete_time = -1  # 第一个包组的完成时间（该包组最后一个包的接收时间）

        # ----- 延迟相关/计算trendline相关 -----
        self.acc_delay = 0
        self.smoothed_delay = 0
        self.acc_delay_list = collections.deque([])
        self.smoothed_delay_list = collections.deque([])

        # ----- 预测带宽相关 -----
        self.state = 'Hold'
        self.last_bandwidth_estimation = 1e6

        self.gamma1 = 12.5  # 检测过载的动态阈值
        self.num_of_deltas_ = 0  # delta的累计个数
        self.time_over_using = -1  # 记录over_using的时间
        self.prev_trend = 0.0  # 前一个trend
        self.overuse_counter = 0  # 对overuse状态计数
        self.last_update_ms = -1  # 上一次更新阈值的时间
        self.now_ms = -1  # 当前系统时间

    def report_states(self, stats: dict):
        '''
        将200ms内接收到包的包头信息都存储于packets_list中
        :param stats: a dict with the following items
        :return: 存储200ms内所有包包头信息的packets_list
        '''
        pkt = stats
        packet_info = PacketInfo()
        packet_info.payload_type = pkt["payload_type"]
        packet_info.ssrc = pkt["ssrc"]
        packet_info.sequence_number = pkt["sequence_number"]
        packet_info.send_timestamp = pkt["send_time_ms"]
        packet_info.receive_timestamp = pkt["arrival_time_ms"]
        packet_info.padding_length = pkt["padding_length"]
        packet_info.header_length = pkt["header_length"]
        packet_info.payload_size = pkt["payload_size"]
        packet_info.size = pkt["header_length"] + pkt["payload_size"] + pkt["padding_length"]
        packet_info.bandwidth_prediction = self.last_bandwidth_estimation
        self.now_ms = packet_info.receive_timestamp  # 以最后一个包的到达时间作为系统时间

        with open('debug.log','a+') as f:
            assert (isinstance(stats, dict))
            f.write(str(stats))
            f.write('\n')

        self.packets_list.append(packet_info)

    def get_estimated_bandwidth(self) -> int:
        '''
        计算估计带宽
        :return: 估计带宽 bandwidth_estimation
        '''
        if len(self.packets_list) == 0:  # 若该时间间隔内未收到包,则返回上一次带宽预测结果
            return self.last_bandwidth_estimation

        # 1. 分包组
        pkt_group_list = self.divide_packet_group()
        if len(pkt_group_list) < 2:  # 若仅有一个包组，返回上一次带宽预测结果
            return self.last_bandwidth_estimation

        # 2. 计算包组梯度
        send_time_delta_list, _, _, delay_gradient_list = self.compute_deltas_for_pkt_group(pkt_group_list)
        with open('debug.log', 'a+') as f:
            f.write("delay_gradient_list = "+str(delay_gradient_list)+"\n")

        # 3. 计算斜率
        trendline = self.trendline_filter(delay_gradient_list, pkt_group_list)
        if trendline == None:   # 当窗口中样本数不够时，返回上一次带宽预测结果
            return self.last_bandwidth_estimation

        # 4. 判断当前网络状态
        overuse_flag = self.overuse_detector(trendline, send_time_delta_list[-1])
        print("current overuse_flag : " + str(overuse_flag))
        # 5. 给出带宽调整方向
        state = self.state_transfer(overuse_flag)
        print("current state : "+str(state))
        # 6. 调整带宽
        bandwidth_estimation = self.rate_adaptation(state)
        self.last_bandwidth_estimation = bandwidth_estimation
        self.packets_list = []  # 清空packets_list

        with open("debug.log", 'a+') as f:
            f.write("Current BWE = " + str(int(bandwidth_estimation)) + '\n')
            f.write("=============================================================\n")
        return bandwidth_estimation

    def divide_packet_group(self):
        '''
        对接收到的包进行分组
        :return: 存有每个包组相关信息的pkt_group_list
        '''
        # todo:对乱序包和突发包的处理
        pkt_group_list = []
        first_send_time_in_group = self.packets_list[0].send_timestamp

        pkt_group = [self.packets_list[0]]
        for pkt in self.packets_list[1:]:
            if pkt.send_timestamp - first_send_time_in_group <= kBurstIntervalMs:
                pkt_group.append(pkt)
            else:
                pkt_group_list.append(PacketGroup(pkt_group))  # 填入前一个包组相关信息
                if self.first_group_complete_time == -1:
                    self.first_group_complete_time = pkt_group[-1].receive_timestamp
                first_send_time_in_group = pkt.send_timestamp
                pkt_group = [pkt]
        pkt_group_list.append(PacketGroup(pkt_group))
        with open('debug.log', 'a+') as f:
            f.write("num of groups = "+str(len(pkt_group_list))+'\n')

        return pkt_group_list

    def compute_deltas_for_pkt_group(self, pkt_group_list):
        '''
        计算包组时间差
        :param pkt_group_list: 存有每个包组相关信息的list
        :return: 发送时间差、接收时间差、包组大小差、延迟梯度list
        '''
        send_time_delta_list, arrival_time_delta_list, group_size_delta_list, delay_gradient_list = [], [], [], []
        for idx in range(1, len(pkt_group_list)):  # 遍历每个包组
            send_time_delta = pkt_group_list[idx].send_time_list[-1] - pkt_group_list[idx - 1].send_time_list[-1]
            arrival_time_delta = pkt_group_list[idx].arrival_time_list[-1] - pkt_group_list[idx - 1].arrival_time_list[
                -1]
            group_size_delta = pkt_group_list[idx].pkt_group_size - pkt_group_list[idx - 1].pkt_group_size
            delay = arrival_time_delta - send_time_delta
            self.num_of_deltas_ += 1
            send_time_delta_list.append(send_time_delta)
            arrival_time_delta_list.append(arrival_time_delta)
            group_size_delta_list.append(group_size_delta)
            delay_gradient_list.append(delay)

        return send_time_delta_list, arrival_time_delta_list, group_size_delta_list, delay_gradient_list

    def trendline_filter(self, delay_gradient_list, pkt_group_list):
        '''
        根据包组的延时梯度计算斜率因子，判断延时变化的趋势
        :param delay_gradient_list: 延迟梯度list
        :param pkt_group_list: 存有每个包组信息的list
        :return: 趋势斜率trendline
        '''
        for i, delay_gradient in enumerate(delay_gradient_list):
            accumulated_delay = self.acc_delay + delay_gradient
            smoothed_delay = kTrendlineSmoothingCoeff * self.smoothed_delay + (
                    1 - kTrendlineSmoothingCoeff) * accumulated_delay
            self.acc_delay = accumulated_delay
            self.smoothed_delay = smoothed_delay
            self.acc_delay_list.append(pkt_group_list[i + 1].complete_time - self.first_group_complete_time)
            self.smoothed_delay_list.append(smoothed_delay)
            if len(self.acc_delay_list) > kTrendlineWindowSize:
                self.acc_delay_list.popleft()
                self.smoothed_delay_list.popleft()
        if len(self.acc_delay_list) == kTrendlineWindowSize:
            avg_acc_delay = sum(self.acc_delay_list) / len(self.acc_delay_list)
            avg_smoothed_delay = sum(self.smoothed_delay_list) / len(self.smoothed_delay_list)

            # 通过线性拟合求解延时梯度的变化趋势：
            numerator = 0
            denominator = 0
            for i in range(kTrendlineWindowSize):
                numerator += (self.acc_delay_list[i] - avg_acc_delay) * (
                        self.smoothed_delay_list[i] - avg_smoothed_delay)
                denominator += (self.acc_delay_list[i] - avg_acc_delay) * (self.acc_delay_list[i] - avg_acc_delay)
            trendline = numerator / denominator
        else:
            trendline = None
        return trendline

    def overuse_detector(self, trendline, ts_delta):
        '''
        根据滤波器计算的趋势斜率，判断当前是否处于过载状态
        :param trendline: 趋势斜率
        :param ts_delta: 发送时间间隔
        :return: overuseflag - 'OVERUSE'  # 'NORMAL', 'UNDERUSE'
        '''
        now_ms = self.now_ms
        overuse_flag = 'NORMAL'
        if self.num_of_deltas_ < 2:
            return overuse_flag

        modified_trend = trendline * min(self.num_of_deltas_, kMinNumDeltas) * threshold_gain_
        print("modified_trend = "+str(modified_trend))

        if modified_trend > self.gamma1:
            if self.time_over_using == -1:
                self.time_over_using = ts_delta / 2
            else:
                self.time_over_using += ts_delta
            self.overuse_counter += 1
            if self.time_over_using > kOverUsingTimeThreshold and self.overuse_counter > 1:
                if trendline > self.prev_trend:
                    self.time_over_using = 0
                    self.overuse_counter = 0
                    overuse_flag = 'OVERUSE'
            elif modified_trend < -self.gamma1:
                self.time_over_using = -1
                self.overuse_counter = 0
                overuse_flag = 'UNDERUSE'
        else:
            self.time_over_using = -1
            self.overuse_counter = 0
            overuse_flag = 'NORMAL'

        self.prev_trend = trendline
        self.update_threthold(modified_trend, now_ms)  # 更新判断过载的阈值
        return overuse_flag

    def update_threthold(self, modified_trend, now_ms):
        '''
        更新判断过载的阈值
        :param modified_trend: 修正后的趋势
        :param now_ms: 当前系统时间
        :return: 无
        '''
        if self.last_update_ms == -1:
            self.last_update_ms = now_ms
        if abs(modified_trend) > self.gamma1 + kMaxAdaptOffsetMs:
            self.last_update_ms = now_ms
            return

        if abs(modified_trend) < self.gamma1:
            k = k_down_
        else:
            k = k_up_
        kMaxTimeDeltaMs = 100
        time_delta_ms = min(now_ms - self.last_update_ms, kMaxTimeDeltaMs)
        self.gamma1 = k * (abs(modified_trend) - self.gamma1) * time_delta_ms
        if (self.gamma1 < 6):
            self.gamma1 = 6
        elif (self.gamma1 > 600):
            self.gamma1 = 600
        self.last_update_ms = now_ms

    def state_transfer(self, overuse_flag):
        '''
        更新发送码率调整方向
        :param overuse_flag: 网络状态
        :return: 新的调整方向
        '''
        newstate = None
        if self.state == 'Decrease' and overuse_flag == 'OVERUSE':
            newstate = 'Decrease'
        elif self.state == 'Decrease' and (overuse_flag == 'NORMAL' or overuse_flag == 'UNDERUSE'):
            newstate = 'Hold'
        elif self.state == 'Hold' and overuse_flag == 'OVERUSE':
            newstate = 'Decrease'
        elif self.state == 'Hold' and overuse_flag == 'NORMAL':
            newstate = 'Increase'
        elif self.state == 'Hold' and overuse_flag == 'UNDERUSE':
            newstate = 'Hold'
        elif self.state == 'Increase' and overuse_flag == 'OVERUSE':
            newstate = 'Decrease'
        elif self.state == 'Increase' and overuse_flag == 'NORMAL':
            newstate = 'Increase'
        elif self.state == 'Increase' and overuse_flag == 'UNDERUSE':
            newstate = 'Hold'
        else:
            print('Wrong state!')
        self.state = newstate
        return newstate

    def rate_adaptation(self, state):
        '''
        根据当前状态（hold, increase, decrease），决定最后的码率
        :param state: （hold, increase, decrease）
        :return: 估计码率
        '''
        bandwidth_estimation = self.last_bandwidth_estimation
        if state == 'Increase':
            bandwidth_estimation = eta * self.last_bandwidth_estimation
        elif state == 'Decrease':
            bandwidth_estimation = alpha * self.last_bandwidth_estimation
        elif state == 'Hold':
            bandwidth_estimation = self.last_bandwidth_estimation
        else:
            print('Wrong State!')
        return bandwidth_estimation


class PacketInfo:
    def __init__(self):
        self.payload_type = None
        self.sequence_number = None  # int
        self.send_timestamp = None  # int, ms
        self.ssrc = None  # int
        self.padding_length = None  # int, B
        self.header_length = None  # int, B
        self.receive_timestamp = None  # int, ms
        self.payload_size = None  # int, B
        self.bandwidth_prediction = None  # int, bps


# 定义包组的类，记录一个包组的相关信息
class PacketGroup:
    def __init__(self, pkt_group):
        self.pkts = pkt_group
        self.arrival_time_list = [pkt.receive_timestamp for pkt in pkt_group]
        self.send_time_list = [pkt.send_timestamp for pkt in pkt_group]
        self.pkt_group_size = sum([pkt.size for pkt in pkt_group])
        self.pkt_num_in_group = len(pkt_group)
        self.complete_time = self.arrival_time_list[-1]
        self.transfer_duration = self.arrival_time_list[-1] - self.arrival_time_list[0]  # todo：当包组只有一个包时这样计算会有问题
