import numpy as np
import pandas

def check_fake(values, i):
    fake_interval = 3
    left = max(i - fake_interval, 0)
    right = min(i + fake_interval, len(values))
    fake = False
    for j in range(left, right + 1):
        if values[j] > values[i]:
            fake = True
            break
    return fake

def get_peaks(time, values):
    time_array = []
    for i in range(1, len(time)-1):
        if values[i] > values[i-1] and values[i] > values[i+1]:
            if check_fake(values, i) is False:
                time_array.append(time[i])
        elif values[i] == values[i-1] and values[i] > values[i+1]:
            last_number = values[i-1]
            for j in range(i-1, -1, -1):
                if values[j] != last_number:
                    if values[j] < last_number:
                        if check_fake(values, i) is False:
                            time_array.append(time[i])
                            break
                    break
    return time_array

def calculate_peak_intervals(file):
        xml = pandas.read_excel(file)
        atrium_values = xml['Atrium'].tolist()
        ventricle_values = xml['Ventricle'].tolist()
        time_values = xml['time(s)'].tolist()

        atrium_peaks = get_peaks(time_values, atrium_values)
        ventricle_peaks = get_peaks(time_values, ventricle_values)

        print(atrium_peaks)
        print(ventricle_peaks)

        """
        atrium_time_differences = []
        ventricle_time_differences = []
        for i in range(1, len(atrium_peaks)):
            atrium_time_differences.append(atrium_peaks[i] - atrium_peaks[i-1])
        for i in range(1, len(ventricle_peaks)):
            ventricle_time_differences.append(ventricle_peaks[i] - ventricle_peaks[i-1])
        #print(atrium_time_differences)
        #print(ventricle_time_differences)
        """

calculate_peak_intervals('10mM_butylone_1.xlsx')