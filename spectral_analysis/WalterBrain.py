import spectrum
from scipy import signal
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
from dcclab.database import *
import random


def VerifyIsoFresh():
    isoPre = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20221216/isoPRE/').spectra()
    isoPost = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20221216/isoPOST/').spectra()

    isoPre.add(isoPost)
    isoPre.displayMeanSTD()

def VerifyIsoFixed():
    isoPre = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20221221/isoPRE/').spectra()
    isoPost = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20221221x/isoPOST/').spectra()

    isoPre.add(isoPost)
    isoPre.displayMeanSTD()

def testButterworthFilter():
    bg = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20220929/morning_verif/darknoise/').spectraSum()
    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/20221216/region_sort/'):
        if dir[0] == '.':
            continue
        data.append(
            spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20221216/region_sort/' + dir + '/').spectra())
    data = spectrum.Spectra(data)
    data.removeThermalNoise(bg)
    data.cut(500, 3025, WN=True)
    data.butterworthFilter()
    # data.displayMeanSTD()
    data.pca()
    # data.pcaScatterPlot(1, 2)

    data.pcaScatterPlot(3, 4)

    data.pcaScatterPlot(5, 6)

    data.pcaScatterPlot(7, 8)

    data.pcaScatterPlot(9, 10)

    data.pcaDisplay(1, 2, 3, 4)
    data.pcaDisplay(5, 6, 7)
    data.pcaDisplay(8, 9, 10)

def WalterBrainPCA():
    bg = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20220929/morning_verif/darknoise/').spectraSum()
    # data = []
    # for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/Walter/'):
    #     if dir[0] == '.':
    #         continue
    #     data.append(
    #         spectrum.Acquisition(
    #             '/Users/antoinerousseau/Desktop/maitrise/DATA/Walter/' + dir + '/').spectra())
    # data = spectrum.Spectra(data)
    # data.removeLabel('white')
    # data.removeThermalNoise(bg)
    # data.smooth()
    # data = data.combineSpectra(2)
    bests = []
    cut_freqs = [8, 9, 10, 11, 12, 13, 14, 15]
    for i in cut_freqs:
        data = []
        for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/Walter/'):
            if dir[0] == '.':
                continue
            data.append(
                spectrum.Acquisition(
                    '/Users/antoinerousseau/Desktop/maitrise/DATA/Walter/' + dir + '/').spectra())
        data = spectrum.Spectra(data)
        data.removeLabel('white')
        data.removeThermalNoise(bg)
        data.butterworthFilter(cutoff_frequency=i)
        data.cut(400, 1600, WN=True)
        best_acc = data.cluster(type='raw')
        bests.append(best_acc)
    plt.plot(cut_freqs, bests)
    plt.xlabel("Cutoff frequencies")
    plt.ylabel("Accuracy")
    plt.show()

    # data.pca()
    #
    # data.pcaScatterPlot(1, 2)
    #
    # data.pcaScatterPlot(3, 4)
    #
    # data.pcaScatterPlot(5, 6)

    # data.pcaScatterPlot(7, 8)

    # data.pcaScatterPlot(9, 10)

    # data.pcaDisplay(1, 2, 3, 4)
    # data.pcaDisplay(5, 6, 7)
    # data.pcaDisplay(8, 9, 10)


def FFTFilter(spectrum, cutoff_frequency, gaussian_width):
    """Remove low frequency noise from a Raman spectrum using FFT and a Gaussian filter.

    Parameters:
        spectrum (np.array): 1D array representing the Raman spectrum.
        cutoff_frequency (float): Center frequency of the Gaussian filter.
        gaussian_width (float): Width of the Gaussian filter.

    Returns:
        np.array: The input spectrum with low frequency noise removed.
    """
    # Perform FFT on the input spectrum
    spectrum_fft = np.fft.fft(spectrum)
    # Define the frequency axis
    n = len(spectrum)
    frequency = np.fft.fftfreq(n)
    # Create the Gaussian filter
    gaussian_filter = np.exp(-((frequency - cutoff_frequency) ** 2) / (2 * gaussian_width ** 2))
    # Multiply the spectrum_fft with the Gaussian filter
    spectrum_fft_filtered = spectrum_fft * gaussian_filter
    # Perform inverse FFT to get the filtered spectrum
    filtered_spectrum = np.fft.ifft(spectrum_fft_filtered)
    return filtered_spectrum.real


def test_fft_filter():
    import spectrum
    bg = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20220929/morning_verif/darknoise/').spectraSum()
    raw = spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/maitrise/DATA/20221216/SN5/').spectraSum()
    raw.removeThermalNoise(bg)
    raw.cut(100, 3025, WN=True)
    raw.label = "raw"
    raw.normalizeCounts()
    FFT = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20221216/SN5/').spectraSum()
    FFT.removeThermalNoise(bg)
    FFT.cut(100, 3025, WN=True)
    FFT.label = "FFT"
    BUT = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20221216/SN5/').spectraSum()
    BUT.removeThermalNoise(bg)
    BUT.cut(100, 3025, WN=True)
    BUT.butterworthFilter(cutoff_frequency=5)
    BUT.label = "BUT"
    BUT.normalizeCounts()


    for i in [0.005, 0.01, 0.03]:
        for j in [0.1, 1, 10, 100]:
            plt.plot(raw.wavenumbers, raw.counts, label="raw")
            plt.plot(raw.wavenumbers, BUT.counts, label="BUT")
            FFT.counts = FFTFilter(FFT.counts, cutoff_frequency=i, gaussian_width=j)
            FFT.normalizeCounts()
            plt.plot(raw.wavenumbers, FFT.counts, label="FFT")
            plt.title("cutoff={0}, gausswidth={1}".format(i, j))
            plt.legend()
            plt.show()




# test_fft_filter()


# testButterworthFilter()
# VerifyIsoFresh()
# VerifyIsoFixed()
WalterBrainPCA()


def test():
    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/Walter/'):
        if dir[0] == '.':
            continue
        data.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/maitrise/DATA/Walter/' + dir + '/').spectra())
    data = spectrum.Spectra(data)
    data.removeLabel("GPi")
    data.removeLabel('GPe')
    data.removeLabel('STN')
    data.removeLabel('Thalamus')
    data.removeLabel('SN')
    data.removeLabel('Putamen')
    data.removeLabel('Caudate')
    data.display()
    data.butterworthFilter(cutoff_frequency=7)
    data.display()
#
# test()

# import itertools
#
# def create_label_permutations(labels):
#     unique_labels = np.unique(labels)
#     permutations = list(itertools.permutations(unique_labels))
#     new_permutations = []
#     for perm in permutations:
#         new_perm = [perm[np.where(unique_labels == label)[0][0]] for label in labels]
#         new_permutations.append(new_perm)
#     return np.array(new_permutations)
#
# # Example usage:
# labels = np.array([1, 2, 3, 4, 1, 1, 4, 4, 2, 3, 1, 4, 2, 3, 3, 1, 4, 2, 3, 4 , 1, 3, 4, 1, 2, 3, 2, 4, 1, 4, 2, 3, 4 , 4, 1, 2, 3, 4])
# print(create_label_permutations(labels))

