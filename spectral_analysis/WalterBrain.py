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
    data.cut(500, 2000, WN=True)
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
    means = []
    stds = []
    bests = []
    cut_freqs = [8, 9, 10, 11, 12, 13, 14, 15]
    for i in cut_freqs:
        accuracys = []
        for j in range(10):
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
            accuracys.append(best_acc)
        mean = np.mean(accuracys)
        std = np.std(accuracys)
        means.append(mean)
        stds.append(std)

    plt.plot(cut_freqs, np.array(means), color="black")
    plt.fill_between(cut_freqs, np.array(means) - np.array(stds), np.array(means) + np.array(stds),
                     color="black", alpha=0.5)
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

def WalterLDA():
    bg = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20220929/morning_verif/darknoise/').spectraSum()
    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/Walter/'):
        if dir[0] == '.':
            continue
        data.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/maitrise/DATA/Walter/' + dir + '/').spectra())
    data = spectrum.Spectra(data)
    data.removeThermalNoise(bg)
    data.butterworthFilter(6)
    data.cut(2800, 3000, WN=True)
    data.lda(n_components=7, display=True)
    data.ldaScatterPlot(1, n_components=7)
    # data.ldaScatterPlot(2, n_components=7)
    # data.ldaScatterPlot(3, n_components=7)
    # data.ldaScatterPlot(4, n_components=7)
    # data.ldaScatterPlot(5, n_components=7)
    # data.ldaScatterPlot(6, n_components=7)
    # data.ldaScatterPlot(7, n_components=7)


def WalterBestALS():
    bg = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20220929/morning_verif/darknoise/').spectraSum()

    for i in [10**1, 10**2, 10**3, 10**4, 10**5, 10**6, 10**7]:
        for j in [0.0001, 0.001, 0.01, 0.1, 1]:
            for k in [1, 2]:
                data = []
                for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/Walter/'):
                    if dir[0] == '.':
                        continue
                    data.append(
                        spectrum.Acquisition(
                            '/Users/antoinerousseau/Desktop/maitrise/DATA/Walter/' + dir + '/').spectra())
                data = spectrum.Spectra(data)
                data.removeThermalNoise(bg)
                if k == 1:
                    data.smooth(n=3)
                    data.ALS(lam=i, p=j)
                    data.cut(2200, 2700, WN=True)
                    bad = data.KNNIndividialSpec(return_accuracy=True)
                if k == 2:
                    data.smooth(n=3)
                    data.ALS(lam=i, p=j)
                    data.cut(450, 1900, WN=True)
                    good = data.KNNIndividialSpec(return_accuracy=True)
            print('For: lam = {0}, p = {1}:'.format(i, j))
            print('Bad = {0}, Good = {1}, diff = {2}'.format(bad, good, good-bad))


def test_signal_in_time():
    bg = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20220929/morning_verif/darknoise/').spectraSum()
    data2 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20221025/signal_over_time/putamen/').spectra()
    data1 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20221025/signal_over_time/spot2/').spectra()
    data2.add(data1)
    data2.removeThermalNoise(bg)
    data2.cut(60, -4)
    data2.smooth(n=3)
    data2.ALS(display=False)
    data2.cut(450, 1900, WN=True)
    data2.display(label=False)
    data2.displayMeanSTD()
    data2.pca()

    data2.pcaScatterPlot(1, 2)

    # data2.pcaScatterPlot(3, 4)

    # data2.pcaScatterPlot(5, 6)

    # data2.pcaDisplay(1, 2, 3)
    # data2.pcaDisplay(4, 5, 6)


    # data2.bin(n=11)
    # data2.display()

    # y = data2.counts
    # x = data2.wavenumbers
    # x_min_ind = data2.getLocalMin()
    # y_min = []
    # x_min = []
    # for i in x_min_ind:
    #     y_min.append(data2.counts[i])
    #     x_min.append(data2.wavenumbers[i])
    # plt.plot(x, y)
    # plt.plot(x_min, y_min, 'ro')
    # plt.show()

    # data2.display()
    # data2.butterworthFilter()
    # data2.smooth(n=7)
    # data2.display()

def test_signal_in_depth():
    bg = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20220929/morning_verif/darknoise/').spectraSum()
    data1 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20221025/signal_over_depth/putamen/').spectra()
    data2 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20221025/signal_over_depth/caudate/').spectra()

    data2.removeThermalNoise(bg)
    data2.cut(60, -4)
    data2.smooth(n=3)
    data2.ALS(display=False)
    data2.cut(450, 1900, WN=True)
    # data2.display(label=False)
    # data2.displayMeanSTD()
    data2.pca()

    data2.pcaScatterPlot(1, 2)

    data2.pcaScatterPlot(3, 4)

    data2.pcaScatterPlot(5, 6)

    # data2.pcaDisplay(1, 2, 3)
    # data2.pcaDisplay(4, 5, 6)

def WalterPCA():
    bg = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20220929/morning_verif/darknoise/').spectraSum()
    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/Walter/'):
        if dir[0] == '.':
            continue
        data.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/maitrise/DATA/Walter/' + dir + '/').spectra())
    data = spectrum.Spectra(data)
    data.removeThermalNoise(bg)
    data.smooth(n=3)
    data.cut(60, -4)
    data.ALS()
    data.cut(400, 1900, WN=True)
    data.displayMeanSTD()
    data.pca()

    data.pcaScatterPlot(1, 2)

    data.pcaScatterPlot(3, 4)

    data.pcaScatterPlot(5, 6)

    data.pcaScatterPlot(7, 8)

    data.pcaScatterPlot(9, 10)

    data.pcaDisplay(1, 2, WN=True)
    data.pcaDisplay(3, 4, WN=True)
    data.pcaDisplay(5, 6, WN=True)
    data.pcaDisplay(7, 8, WN=True)
    data.pcaDisplay(9, 10, WN=True)

# Test label KNN
def test_label_NKK():

    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/Walter/sort_by_acquisition/'):
        if dir[0] == '.':
            continue
        data.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/maitrise/DATA/Walter/sort_by_acquisition/' + dir + '/').spectra())
    data = spectrum.Spectra(data)
    # data.displayMeanSTD()
    # data.butterworthFilter()
    data.normalizeIntegration()
    data.cut(450, 1900, WN=True)
    data.KNNIndividualLabel()


# Find best ALS parameters for KNN by labels
def bestALSwithKNN():
    bg = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20220929/morning_verif/darknoise/').spectraSum()

    for i in [10**1, 10**2, 10**3, 10**4, 10**5, 10**6, 10**7]:
        for j in [0.0001, 0.001, 0.01, 0.1, 1]:
            for k in [1, 2]:
                data = []
                for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/Walter/sort_by_acquisition/'):
                    if dir[0] == '.':
                        continue
                    data.append(
                        spectrum.Acquisition(
                            '/Users/antoinerousseau/Desktop/maitrise/DATA/Walter/sort_by_acquisition/' + dir + '/').spectra())
                data = spectrum.Spectra(data)
                data.removeThermalNoise(bg)
                if k == 1:
                    data.smooth(n=3)
                    data.ALS(lam=i, p=j)
                    data.cut(2200, 2700, WN=True)
                    bad = data.KNNIndividualLabel(return_accuracy=True)
                if k == 2:
                    data.smooth(n=3)
                    data.ALS(lam=i, p=j)
                    data.cut(2700, 3000, WN=True)
                    good = data.KNNIndividualLabel(return_accuracy=True)
            print('For: lam = {0}, p = {1}:'.format(i, j))
            print('Bad = {0}, Good = {1}, diff = {2}'.format(bad, good, good-bad))


def bestBWFwithKNN():
    bg = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20220929/morning_verif/darknoise/').spectraSum()

    for i in [0.5, 0.8, 1, 3, 5, 8, 10, 15]:
        for j in [1, 2, 3, 4]:
            for k in [1, 2]:
                data = []
                for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/Walter/sort_by_acquisition/'):
                    if dir[0] == '.':
                        continue
                    data.append(
                        spectrum.Acquisition(
                            '/Users/antoinerousseau/Desktop/maitrise/DATA/Walter/sort_by_acquisition/' + dir + '/').spectra())
                data = spectrum.Spectra(data)
                data.removeThermalNoise(bg)
                if k == 1:
                    data.smooth(n=3)
                    data.butterworthFilter(cutoff_frequency=i, order=j)
                    data.cut(2200, 2700, WN=True)
                    bad = data.KNNIndividualLabel(return_accuracy=True)
                if k == 2:
                    data.smooth(n=3)
                    data.butterworthFilter(cutoff_frequency=i, order=j)
                    data.cut(2700, 3000, WN=True)
                    good = data.KNNIndividualLabel(return_accuracy=True)
            print('For: Freq = {0}, Order = {1}:'.format(i, j))
            print('Bad = {0}, Good = {1}, diff = {2}'.format(bad, good, good-bad))


def WalterDisplay():
    bg = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20220929/morning_verif/darknoise/').spectraSum()
    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/Walter/sort_by_region/'):
        if dir[0] == '.':
            continue
        data.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/maitrise/DATA/Walter/sort_by_region/' + dir + '/').spectra())
    data = spectrum.Spectra(data)

    data.removeThermalNoise(bg)

    data.smooth()
    data.butterworthFilter(cutoff_frequency=8, order=4)
    data.cut(2700, 3000, WN=True)
    # data.displayMeanSTD(WN=True)
    data.pca()
    # data.KNNIndividualLabel()


    # data.pcaScatterPlot(1, 2)
    #
    # data.pcaScatterPlot(3, 4)
    #
    # data.pcaScatterPlot(5, 6)
    #
    # data.pcaScatterPlot(7, 8)
    #
    # data.pcaDisplay(1, 2, WN=True)
    # data.pcaDisplay(3, 4, WN=True)
    # data.pcaDisplay(5, 6, WN=True)
    # data.pcaDisplay(7, 8, WN=True)


WalterDisplay()

# bestBWFwithKNN()
# bestALSwithKNN()
# test_label_NKK()
# WalterPCA()
# test_signal_in_time()
# test_signal_in_depth()
# WalterBestALS()

# WalterLDA()
# test_fft_filter()
# testButterworthFilter()
# VerifyIsoFresh()
# VerifyIsoFixed()
# WalterBrainPCA()


# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import ConfusionMatrixDisplay
# import seaborn as sns
# label_str = ['Caudate', 'GPe', 'GPi', 'STN', 'Thalamus', 'SN', 'Putamen', 'WM']
# labelList = ['WM', 'WM', 'WM', 'WM', 'WM', 'WM', 'WM', 'WM', 'WM', 'WM', 'WM', 'WM', 'WM', 'WM', 'WM', 'WM', 'WM', 'WM', 'WM', 'WM', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Caudate','Caudate','Caudate','Caudate','Caudate','Caudate','Caudate','Caudate', 'GPe', 'GPe', 'GPe', 'GPe', 'GPe', 'GPe', 'GPe', 'GPe', 'GPi', 'GPi', 'GPi', 'GPi', 'GPi', 'GPi', 'GPi', 'GPi', 'GPi', 'GPi', 'GPi', 'GPi', 'GPi', 'GPi', 'STN', 'STN', 'STN', 'STN', 'STN', 'STN', 'STN', 'STN', 'STN', 'STN', 'STN', 'STN', 'STN', 'STN', 'STN', 'STN', 'STN', 'STN', 'STN', 'STN', 'STN', 'STN', 'STN', 'STN', 'STN', 'STN', 'STN', 'STN', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN']
# prediction_list_str = ['WM', 'WM', 'WM', 'WM', 'WM', 'WM', 'Caudate', 'WM', 'WM', 'WM', 'WM', 'WM', 'WM', 'STN', 'WM', 'WM', 'WM', 'WM', 'WM', 'WM', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'SN', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Caudate', 'Putamen', 'Putamen', 'Putamen', 'GPi', 'Putamen', 'GPe', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'STN', 'Putamen', 'Putamen', 'WM', 'Putamen', 'Putamen', 'Thalamus', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'Putamen', 'SN', 'Putamen', 'Caudate','Caudate','Caudate','Caudate','Caudate','Caudate','Caudate','Caudate', 'GPe', 'GPe', 'GPe', 'GPe', 'GPe', 'GPe', 'GPe', 'GPe', 'GPe', 'GPi', 'GPi', 'GPe', 'GPi', 'GPe', 'GPi', 'GPi', 'GPi', 'GPi', 'GPi', 'GPi', 'GPi', 'GPi', 'STN', 'STN', 'STN', 'STN', 'STN', 'STN', 'STN', 'GPe', 'STN', 'STN', 'SN', 'STN', 'STN', 'STN', 'STN', 'SN', 'GPi', 'STN', 'Thalamus', 'STN', 'STN', 'STN', 'STN', 'STN', 'WM', 'STN', 'STN', 'STN', 'STN', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Putamen', 'Thalamus', 'Caudate', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Caudate', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'Thalamus', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'STN', 'SN', 'SN', 'STN', 'SN', 'SN', 'SN', 'SN', 'GPe', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'GPi', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN', 'SN']
#
#
# mat = confusion_matrix(labelList, prediction_list_str, labels=label_str)
# cmn = np.round(mat.astype('float')*100 / mat.sum(axis=1)[:, np.newaxis])
# sns.heatmap(cmn, annot=True, fmt='.0f', xticklabels=label_str, yticklabels=label_str)
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# # plt.savefig('/Users/antoinerousseau/Desktop/confusion_matrix.eps')
# plt.show()
