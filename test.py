from alphartc_gym import gym

import os
import glob
# from gym.alphartc_gym.utils.packet_info import PacketInfo
# from gym.alphartc_gym.utils.packet_record import PacketRecord
from BandwidthEstimator_gcc import Estimator

def test():
    f = open('test.log','a+')
    estimator = Estimator()
    bwe = 1000
    total_stats = []
    trace_files = os.path.join(os.path.dirname(__file__), "traces", "*.json")
    for trace_file in glob.glob(trace_files):
        g = gym.Gym("test_gym")
        g.reset(trace_path=trace_file,report_interval_ms=200,duration_time_ms=0)
        while True:
            stats, done = g.step(bwe)     # use the estimated bandwidth
            #todo: get bwe estimated by model
            for stat in stats:
                estimator.report_states(stat)
            bwe = estimator.get_estimated_bandwidth()
            f.write('Current BWE: ' + str(bwe))
            f.write('\n')
            if not done:
                total_stats += stats
            else:
                break
        assert(total_stats)
        f.write('====================States Info====================\n')
        for stats in total_stats:
            assert(isinstance(stats,dict))
            f.write(str(stats))
            f.write('\n')
        f.close()


if __name__ == '__main__':
    test()