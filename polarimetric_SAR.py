# Workshop 7 SAR Polarimetry Made: Jauaries Loyala -- 28.11.2019)

# The needed packages
import pandas as pd
import scipy
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy import ndimage, misc
from scipy.constants import c
from scipy import fftpack


WORKDIR = os.getcwd()
pi = np.pi


# The MATLAB data
matlab_data = sio.loadmat("/Users/Käyttäjä/Downloads/EMISAR_L_may_1995.mat")
#print(matlab_data)

# Dividing the data in to parts
Shh = matlab_data["Shh"]
Svv = matlab_data["Svv"]
Shv = matlab_data["Shv"]
Svh = matlab_data["Svh"]


# Plotting task 1
def VV_polarization_plot():
    fig, ax = plt.subplots(1, figsize = (10, 10))

    # Figure
    ax.set_title("VV polarization [log($\sigma$)]")
    ax.imshow(np.log10(4 * pi * np.abs(np.real(np.conj(Svv) * Svv))), cmap = "Greys")
VV_polarization_plot()


# Plotting task 2
def Shv_plot():
    fig, ax = plt.subplots(1, figsize = (10, 10))

    # Figure
    ax.imshow(np.imag(Shv), cmap = "Greys")
    ax.set_title(r"HV [$\phi$]")
Shv_plot()


# Plotting task 3
def total_power_plot():
    total_power = np.stack([Shh, Svv, Shv, Svh])
    total_power = np.log(np.sum(4 * pi * np.abs(np.real(np.conj(total_power) * total_power)), axis = 0))

    fig, ax = plt.subplots(1, figsize = (10, 10))

    # Figure
    ax.set_title("Total Power [log($\sigma$)]")
    ax.imshow(total_power, cmap = "Greys")
total_power_plot()


# Plotting task 4
def Svv_distribution_plot():
    Svv_phase = np.imag(Svv)

    fig, ax = plt.subplots(1, figsize = (10, 10))

    # Figure
    ax.set_title("Distribution of VV")
    ax.hist(Svv_phase, bins = 10)
Svv_distribution_plot()


# Plotting task 5
def Svv_backscattering_coefficient_plot():
    sigma = pi * 4 * np.abs(np.real(np.conj(Svv) * Svv))

    fig, ax = plt.subplots(1, 2, figsize = (10, 10))

    # Figure
    ax[0].set_title("VV [$\sigma$]")
    ax[0].imshow(sigma, cmap = "Greys", vmin = 1e-6, vmax = 1)

    ax[1].set_title("VV [log($\sigma$)]")
    ax[1].imshow(np.log10(sigma), cmap = "Greys", vmin = -6, vmax = 1)
Svv_backscattering_coefficient_plot()


# Plotting task 6
def multilooking_plot():
    total_power = np.stack([Shh, Svv, Shv, Svh])
    total_power = np.log(np.sum(4 * pi * np.abs(np.real(np.conj(total_power) * total_power)), axis = 0))
    filtered = ndimage.uniform_filter(total_power, size = 5)

    fig, ax = plt.subplots(1, figsize = (10, 10))

    # Figure
    ax.set_title("Total power (boxcar filtered 5 * 5)")
    ax.imshow(filtered, cmap = "Greys")
multilooking_plot()


# Plotting task 7
def average_plot():
    sigma = 4 * pi * np.abs(np.real(np.conj(Shh) * Shh))
    filtered = ndimage.uniform_filter(sigma, size = 10)

    fig, ax = plt.subplots(1, 2, figsize = (10, 10))

    # Figure
    ax[0].set_title("HH [log($\sigma$)]")
    ax[0].imshow(np.log10(sigma), cmap = "Greys", vmin = -5, vmax = 1)

    ax[1].set_title("HH (boxcar filtered 10 * 10) [log($\sigma$)]")
    ax[1].imshow(np.log10(filtered), cmap = "Greys", vmin = -5, vmax = 1)
average_plot()


# Plotting task 8
def coherence_plot():
    coherence = Svv / Shh
    sigma = 4 * pi * np.abs(np.real(np.conj(coherence) * coherence))

    fig, ax = plt.subplots(1, 2, figsize = (10, 10))

    # Figure
    ax[0].set_title(r"Coherence [$\sigma$]")
    ax[0].imshow(sigma, cmap = "viridis", vmin = 0.8, vmax = 1)
    ax[0].imshow(np.log10(sigma))

    ax[1].set_title(r"Coherence [$\phi$]")
    ax[1].imshow(np.log(np.imag(coherence)))
coherence_plot()

plt.show()
