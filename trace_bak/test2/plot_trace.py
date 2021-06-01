import numpy as np
import matplotlib.pyplot as plt
import json

def plot_bandwidth(duration,capacity):
    len_duration = len(duration)
    x_duration = []
    y = 0.0
    for i in range(0,len_duration + 1):
        x_duration.append(sum(duration[:i]))
    x = np.linspace(0,x_duration[-1],len_duration)
    for i in range(0,len_duration):
        y += np.asarray([float(1) if (j >= x_duration[i] and j < x_duration[i+1]) else float(0) for j in x]) * capacity[i]
    plt.xlabel('duration(ms)')
    plt.ylabel('capacity(kbps)')
    # plt.title('4G_500kbps.json')
    plt.plot(x,y)
    plt.show()

def plot_rtt(duration,rtt):

    pass

def plot_loss(duration,loss):

    pass

def plot_all(trace_file):
    duration = []
    capacity = []
    rtt = []
    loss = []
    with open(trace_file,'r',encoding='utf-8') as f:
        traces = json.load(f)
        uplink_info = traces["uplink"]["trace_pattern"]
        for i in range(len(uplink_info)):
            duration.append(uplink_info[i]["duration"])
            capacity.append(uplink_info[i]["capacity"])
            # if(uplink_info[i].has_key("rtt")):
            #     rtt.append(uplink_info[i]["rtt"])
            # if(uplink_info[i].has_key("loss")):
            #     loss.append(uplink_info[i]["loss"])
        plot_bandwidth(duration,capacity)
        # plot_rtt(duration,rtt)
        # plot_loss(duration,loss)

if __name__ == "__main__":
    plot_all('norway_bus_1.json')