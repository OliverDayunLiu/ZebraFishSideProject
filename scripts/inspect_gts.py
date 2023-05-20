import numpy as np
import sys, os
import pandas
import matplotlib.pyplot as plt


def plot_graph(xml_path, label, color):
    full_xml = xml_path
    heartrates = []
    xml = pandas.read_excel(full_xml)
    atrium_values = xml['Atrium'].tolist()
    atrium_values = np.array(atrium_values)
    atrium_avg_value = np.average(atrium_values)
    frame_nums = np.arange(1, len(atrium_values) + 1)
    for i in range(0, len(frame_nums)):
        heartrates.append(float(atrium_values[i] - atrium_avg_value) / atrium_avg_value)
    plt.plot(frame_nums, heartrates, label=label, color=color)

def main():
    plot_graph(os.path.join('../data/raw_video/10mM MDMA #8.xlsx'), 'MDMA8', 'green')
    plot_graph(os.path.join('../data/raw_video/10mM methylone #5.xlsx'), 'methylone5', 'blue')
    plot_graph(os.path.join('../data/raw_video/control #21.xlsx'), 'control21', 'red')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()