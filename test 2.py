from alphartc_gym import gym

import os
import glob
from alphartc_gym.utils.packet_info import PacketInfo
from alphartc_gym.utils.packet_record import PacketRecord

def test():
    bwe = 1000
    total_stats = []
    trace_files = os.path.join(os.path.dirname(__file__),"traces","*.json")
    for trace_file in glob.glob(trace_files):
        g = gym.Gym("test_gym")
        g.reset(trace_path=trace_file,report_interval_ms=200,duration_time_ms=0)
        while True:
            stats, done = g.step(bwe)     # use the estimated bandwidth
            #todo: get bwe estimated by model

            if not done:
                total_stats += stats
            else:
                break
        #assert(total_stats):
        for stats in total_stats:
            assert(isinstance(stats,dict))