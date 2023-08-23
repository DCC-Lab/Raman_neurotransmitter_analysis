import spectrum
from scipy import signal
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
from dcclab.database import *
import random

def iso_quartz_VS_cuvette():
    quartz = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20230323/fibril/iso_quartz/').spectraSum()
    cuvette = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20230323/fibril/isoPOST/').spectraSum()

    data = quartz.addSpectra(cuvette)
    data.cut(60, -4)
    data.normalizeIntegration()
    data.display()


def pbs_verif():
    pbs1 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20230323/fibril/pbs1/').spectraSum()
    pbs2 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20230323/fibril/pbs2/').spectraSum()

    data = pbs1.addSpectra(pbs2)
    data.cut(60, -4)
    data.display()

def fibril():
    dn = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20230323/fibril/dark/').spectraSum()

    pbss = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/20230323/geo_test/'):
        if dir[0] == '.':
            continue
        pbss.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/maitrise/DATA/20230323/geo_test/' + dir + '/').spectra())
    pbss = spectrum.Spectra(pbss)

    pbs1 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20230323/fibril/pbs1/').spectra()
    pbs1.keep(20)
    pbs2 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20230323/fibril/pbs2/').spectra()
    pbs2.keep(20)
    pbs1.removeThermalNoise(dn)
    pbs2.removeThermalNoise(dn)

    fibril100ms = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20230323/fibril/fibril100ms/').spectra()
    fibril1s = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20230323/fibril/fibril1s/').spectra()
    fibril1s.keep(20)
    fibril100ms.removeThermalNoise(dn)
    fibril1s.removeThermalNoise(dn)



    fibril1s.add(pbs1, pbs2, pbss)
    # fibril1s.ALS()
    fibril1s.smooth(n=3)
    fibril1s.cut(500, 1900, WN=True)
    fibril1s.normalizeIntegration()
    # fibril1s.displayMeanSTD(WN=True)
    fibril1s.pca()

    fibril1s.pcaScatterPlot(1, 2)

    fibril1s.pcaScatterPlot(3, 4)

    # fibril1s.pcaScatterPlot(5, 6)

    # fibril1s.pcaScatterPlot(7, 8)

    fibril1s.pcaDisplay(1, 2)
    fibril1s.pcaDisplay(3, 4)
    # fibril1s.pcaDisplay(5, 6)
    # fibril1s.pcaDisplay(7, 8)


def pbs_experiment():
    dn = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20230323/fibril/dark/').spectraSum()

    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/20230323/geo_test/'):
        if dir[0] == '.':
            continue
        data.append(
            spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20230323/geo_test/' + dir + '/').spectra())
    data = spectrum.Spectra(data)
    data.removeThermalNoise(dn)
    data.cut(450, None, WN=True)
    data.normalizeIntegration()
    data.pca()
    data.pcaDisplay(1, 2)
    data.pcaDisplay(3, 4)
    # data.displayMeanSTD()

def fibrilsPM():
    dn = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20230323/fibril/dark/').spectraSum()

    f11 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20230323/fibril/fibril1s/').spectra()
    # f11.keep(n=20)

    pbs1 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20230323/fibril/pbs1/').spectra()
    # pbs1.keep(n=20)
    pbs2 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20230323/fibril/pbs2/').spectra()
    # pbs2.keep(n=20)

    pbss = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/20230323/geo_test/'):
        if dir[0] == '.':
            continue
        pbss.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/maitrise/DATA/20230323/geo_test/' + dir + '/').spectra())
    pbss = spectrum.Spectra(pbss)

    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/20230323/fibrilPM/'):
        if dir[0] == '.':
            continue
        data.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/maitrise/DATA/20230323/fibrilPM/' + dir + '/').spectra())
    data = spectrum.Spectra(data)
    data.removeThermalNoise(dn)
    # data.add(f11, pbs1, pbs2, pbss)
    data.butterworthFilter(cutoff_frequency=8, order=3)
    # data.ALS(lam=1000000, p=0.1)
    # data = data.combineSpectra(add=2)
    data.smooth(n=5)

    # data.cut(450, 1900, WN=True)
    # data.normalizeIntegration()
    # data.KNNIndividualLabel()
    data.cut(450, 3000, WN=True)
    data.shortenLabels()
    # data.displayMeanSTD(WN=True)
    data.lda(display=True)
    data.ldaScatterPlot(LDx=1)
    # data.pca()

    # data.pcaScatterPlot(1, 2)

    # data.pcaScatterPlot(3, 4)

    # data.pcaScatterPlot(5, 6)


    # data.pcaScatterPlot(7, 8)
    #
    # data.pcaScatterPlot(9, 10)


    # data.pcaDisplay(1, 2, WN=True)
    # data.pcaDisplay(3, 4, WN=True)
    # data.pcaDisplay(5, 6, WN=True)
    # data.pcaDisplay(7, 8)
    # data.pcaDisplay(9, 10)

def fibril_exp():
    dn = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20230421/dn/').spectraSum()
    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/20230427/fibril/fibril_exp/'):
        if dir[0] == '.':
            continue
        data.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/maitrise/DATA/20230427/fibril/fibril_exp/' + dir + '/').spectra())
    data = spectrum.Spectra(data)
    data.removeThermalNoise(dn)
    data.smooth(n=5)
    data.butterworthFilter(cutoff_frequency=8, order=3)
    data.cut(450, 1900, WN=True)
    # data.normalizeIntegration()
    # data.displayMeanSTD()
    data.pca()
    data.pcaScatterPlot(1, 2)
    data.pcaScatterPlot(3, 4)
    data.pcaDisplay(1, 2)
    data.pcaDisplay(3, 4)




def alpha_syn():
    dn10 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20230421/dn/').spectraSum()
    dn1 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20230323/fibril/dark/').spectraSum()

    # Load fibril 1sec integration data and get make them 10s per spectrum object
    fibril1 = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/20230323/fibrilPM/'):
        if dir[0] == '.':
            continue
        fibril1.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/maitrise/DATA/20230323/fibrilPM/' + dir + '/').spectra())
    fibril1 = spectrum.Spectra(fibril1)
    fibril1.removeThermalNoise(dn1)
    fibril1 = fibril1.combineSpectra(add=10)

    # Load monomere data
    mono = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/20230421/data/'):
        if dir[0] == '.':
            continue
        mono.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/maitrise/DATA/20230421/data/' + dir + '/').spectra())
    mono = spectrum.Spectra(mono)
    mono.removeThermalNoise(dn10)

    # Load fibril data set 2
    fibril10 = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/20230427/fibril/fibril_exp/'):
        if dir[0] == '.':
            continue
        fibril10.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/maitrise/DATA/20230427/fibril/fibril_exp/' + dir + '/').spectra())
    fibril10 = spectrum.Spectra(fibril10)
    fibril10.removeThermalNoise(dn10)

    fibril10.add(mono)
    fibril10.shortenLabels()
    fibril10.smooth(n=5)
    # fibril10.normalizeIntegration()
    fibril10.butterworthFilter(cutoff_frequency=8, order=3)
    fibril10.cut(400, 3000, WN=True)
    fibril10.displayMeanSTD()
    # fibril10.pca()
    # fibril10.pcaScatterPlot(1, 2)
    # fibril10.pcaScatterPlot(3, 4)
    # fibril10.pcaDisplay(1, 2)
    # fibril10.pcaDisplay(3, 4)
    fibril10.tsne()


    # mono.shortenLabels()
    # mono.smooth(n=5)
    # mono.cut(450, 1900, WN=True)
    # mono.pca()
    # mono.pcaScatterPlot(1, 2)
    # mono.pcaScatterPlot(3, 4)
    # mono.pcaDisplay(1, 2)
    # mono.pcaDisplay(3, 4)


# iso_quartz_VS_cuvette()
# pbs_verif()
# fibril()
# pbs_experiment()
# fibrilsPM()
# fibril_exp()
alpha_syn()