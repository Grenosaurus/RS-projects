# Workshop 4 CW Doppler radar (Made: Jauaries).

import pandas as pd
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
from scipy import fftpack


v1 = 0.83
v2 = 1.39
f0 = 5800000000
sweep_time = 1e-3
samples = 256


delta_f1 = ((2 * v1)/c) * f0
delta_f2 = ((2 * v2)/c) * f0
sampling_rate = (1/sweep_time) * samples

print(f"f1 = {delta_f1} Hz")
print(f"f2 = {delta_f2} Hz")
print(f"Sampling rate: {sampling_rate} Hz")


matlab_data = sio.loadmat("/Users/Käyttäjä/Downloads/ancortek580ADpackage/ancortek580ADpackage/Matlab GUI_1Tx1Rx/CW_exercise4.mat")
d_matlab = matlab_data["DATA"].reshape(-1)
data = pd.Series(d_matlab, index = [i/sampling_rate for i in range(len(d_matlab))])
#print(data)


fig1, ax1 = plt.subplots()
ax1.plot(data)
ax1.set_xlabel('Time')
ax1.set_ylabel('Amplitude')
ax1.set_title('Real part of CW wave')


d = data.loc[6 : 6.5]
f = fftpack.fftfreq(len(d), 1/sampling_rate)
m = np.abs(np.real(fftpack.fft(d)))
ftt = pd.DataFrame({"f": f, "m": m})
#print(ftt)


filtered_ftt = ftt[(ftt["f"] != 0) & (ftt["f"] < -100) & (ftt["f"] < 100)]
#print(filtered_ftt)
selected_f = filtered_ftt[filtered_ftt["m"] > 5e+4]["f"].median()
print(f"selected_f: {selected_f}")


fig2, ax2 = plt.subplots()
ax2.plot(ftt["f"], ftt["m"])
ax2.axvline(x = selected_f, color = "red")
ax2.set_xlabel('f')
ax2.set_xlim(-100, 100)
ax2.set_ylabel('Magnitude')
ax2.set_ylim(0, 5e+6)
ax2.set_title('FFT')


def intervals_speed(x, spacing):
    _bottom = x - (spacing/2)
    _up = x + (spacing/2)
    _d = data.loc[_bottom: _up]
    _f = fftpack.fftfreq(len(_d), 1/sampling_rate)
    _m = np.abs(np.real(fftpack.fft(_d)))
    _ftt = pd.DataFrame({"f": _f, "m": _m})
    _filtered_ftt = _ftt[(_ftt["f"] != 0) & (_ftt["f"] > -100) & (_ftt["f"] < 100)]

    f = _filtered_ftt[_filtered_ftt["m"] > 5e+5]["f"].median()

    return (c * f)/(2 * f0)


x = np.linspace(0, 10, 20)
spacing = x[1] - x[0]


doppler_speed = {
    "v": [],
    "t": []
}


for xx in x:
    doppler_speed["t"].append(xx)
    doppler_speed["v"].append(intervals_speed(xx, spacing))


speed = pd.DataFrame(doppler_speed)
#print(speed)


fig3, ax3 = plt.subplots()
ax3.plot(speed["t"], speed["v"])
ax3.set_xlabel('t [s]')
ax3.set_xlim(0, 10)
ax3.set_ylabel('v [m/s]')
ax3.set_ylim(-3, 3)
ax3.set_title("Speed")


plt.show()
