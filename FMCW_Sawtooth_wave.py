# Workshop 4 FMCW Sawtooth wave (Made: Jauaries).

import math
import pandas as pd
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

from scipy.constants import c
from scipy import fftpack
from ipywidgets import interact, widgets
from functools import partial


B = 150e+6
T = 1e-3
d = 10
d1 = 0
d2 = 10

sweep_time = 1e-3
samples_per_sweep = 256
sampling_rate = (1/sweep_time) * samples_per_sweep

# From calculating and observing the first plot
f_min = 0
f_max = 1e+4
m_min = 2e+3
m_max = 2e+4
n_sweep = 4000

integration_time = 0.5
n_samples_interval = int(integration_time/T)


# Beat frequency into distance
def freq_dis(x1):
    return (c/2) * (T/B) * x1
# Distance into beat frequency
def dis_freq(x2):
    return (2 * x2 * B)/(c * T)

print(f"f1 = {dis_freq(d1)} Hz, f2 = {dis_freq(d2):.2E} Hz")
print(f"d = {freq_dis(1/T)} m")
print(f"sampling rate: {sampling_rate: .2E} Hz")


# Processing the Matlab data
matlab_data = sio.loadmat("/Users/Käyttäjä/Downloads/ancortek580ADpackage/ancortek580ADpackage/Matlab GUI_1Tx1Rx/FMCW_Sawtooth_exercise4.mat")
#print(matlab_data)
d_matlab = matlab_data["DATA"].reshape(-1)
data = pd.Series(d_matlab)

# Reshapes the data into a (*, 256 array)
sweep = data.to_numpy().reshape((-1, samples_per_sweep))
#print(sweep)
def plot_FMCW():
    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.set_title('Real part of FMCW Sawtooth wave')

interact(plot_FMCW())

def fft_sweep(n_sweep):
    _f = fftpack.fftfreq(samples_per_sweep, 1/sampling_rate)
    _m = np.abs(np.real(fftpack.fft(sweep[n_sweep])))
    _fft = pd.DataFrame({"f": _f, "m": _m})

    # The filtered FFT
    filtered_ftt = _fft[(_fft["f"] != 0) & (_fft["f"] > f_min) & (_fft["f"] < f_max)]
    #print(filtered_ftt)
    selected_f = filtered_ftt.loc[filtered_ftt["m"].idxmax(), "f"]
    avg_f = np.average(np.array(filtered_ftt["f"]), weights = np.array(np.array(filtered_ftt["m"])))
    return (_f, _m, selected_f, avg_f)

def plot_fft(n_sweep):
    fig, axs = plt.subplots(1, 2, figsize = (6, 6))
    fig.tight_layout(pad = 2)

    # Amplitude
    axs[0].plot(sweep[n_sweep])
    axs[0].set_xlabel('Sample')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_title(f"Amplitude for {n_sweep} sweep")

    #FFT for the sweep
    _f, _m, selected_f, avg_f = fft_sweep(n_sweep)

    # Places a text box in the upper left in axes coordinate
    axs[1].text(0.05, 0.90, f"f_max = {selected_f: .2E} Hz", transform = axs[1].transAxes, fontsize = 12, verticalalignment = "top", bbox = {'facecolor': 'red', 'alpha': 0.4, 'pad': 10})
    axs[1].text(0.05, 0.80, f"f_max = {selected_f: .2E} Hz", transform = axs[1].transAxes, fontsize = 12, verticalalignment = "top", bbox = {'facecolor': 'blue', 'alpha': 0.4, 'pad': 10})

    axs[1].scatter(_f, _m)
    axs[1].set_xlabel("f")
    axs[1].set_ylabel("m")
    axs[1].set_ylim(0, 1e+4)
    axs[1].set_title(f"FTT for {n_sweep} sweep")
    axs[1].axvline(selected_f, color = "red", ls = "--", alpha = 0.8)
    axs[1].axvline(avg_f, color = "blue", ls = "--", alpha = 0.8)
    axs[1].axvline(f_min, color = "grey", ls = "--", alpha = 0.3)
    axs[1].axvline(f_max, color = "grey", ls = "--", alpha = 0.5)
    axs[1].axvline(m_min, color = "grey", ls = "--", alpha = -5)

interact(plot_fft(n_sweep))


beat_freq = np.array([fft_sweep(i)[2] for i in range(0, len(sweep))])
#print(beat_freq)


def plot_distance(t, d, i):
    fig, ax = plt.subplots(figsize = (10, 10))
    ax.plot(t, d, marker="o")

    ax.text(0.55, 0.05, f"Integration_time = {i} s / {int(integration_time/T)} sweep", transform = ax.transAxes, fontsize = 12, verticalalignment = "bottom", bbox = {'facecolor': 'blue', 'alpha': 0.4, 'pad': 10})

    ax.set_xlabel("t [s]")
    ax.set_xlim(0, 10)
    ax.set_ylabel("d [m]")
    ax.set_title("Distance")


print(f"{n_samples_interval} sweeps or {integration_time} s as integration time.")

# Creating a empty vector, that contains the distance of the size (corresponding to the integrated unit)
d = np.empty(math.ceil(len(sweep)/n_samples_interval))
# Creating a time cell for the integrated time
t = np.arange(d.shape[0]) * integration_time


for i in range(0, d.shape[0]):
    low = i * n_samples_interval
    high = low + n_samples_interval

    d[i] = average_beat_freq = np.mean([freq_dis(f) for f in beat_freq[low: high]])


interact(plot_distance(t, d, integration_time))


plt.show()
