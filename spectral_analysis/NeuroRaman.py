import spectrum
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
from dcclab.database import *
import random


def VealBrain():
    bg = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220929/morning_verif/darknoise/').spectraSum()
    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/20221028/data_filed_by_regions/'):
        if dir[0] == '.':
            continue
        data.append(
            spectrum.Acquisition('/Users/antoinerousseau/Desktop/20221028/data_filed_by_regions/' + dir + '/').spectra())
    data = spectrum.Spectra(data)

    data.removeThermalNoise(bg)
    data.fixAberations()
    data.cut(60, -5)
    data.normalizeIntegration()
    data.displayMeanSTD()
    data.pca()

    data.pcaScatterPlot(1, 2)

    data.pcaScatterPlot(3, 4)

    data.pcaScatterPlot(5, 6)

    data.pcaScatterPlot(7, 8)

    data.pcaDisplay(1, 2, 3, 4)

    data.pcaDisplay(5, 6, 7, 8)


    # putamen = data.getLabelSpectra('SNC5_1')
    # putamen6 = data.getLabelSpectra('SNC5_2')
    # putamen7 = data.getLabelSpectra('SNC6')
    # putamen.add(putamen6, putamen7)
    # putamen.removeThermalNoise(bg)
    # putamen.normalizeIntegration()
    # putamen.cut(60, -5)
    # putamen.displayMeanSTD()
    # putamen.pca()
    #
    # putamen.pcaScatterPlot(1, 2)
    #
    # putamen.pcaScatterPlot(3, 4)
    #
    # putamen.pcaScatterPlot(5, 6)
    #
    # putamen.pcaScatterPlot(7, 8)
    #
    # putamen.pcaScatterPlot(9, 10)
    #
    # putamen.pcaDisplay(1, 2, 3)
    # putamen.pcaDisplay(4, 5, 6)


def FixedVSNotFixed():
    bg = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220929/morning_verif/darknoise/').spectraSum()
    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/20221028/'):
        if dir[0] == '.':
            continue
        data.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/20221028/' + dir + '/').spectra())
    data = spectrum.Spectra(data)

    putamen = data.getLabelSpectra('thalamus6')
    putamen6 = data.getLabelSpectra('thalamus5')

    fixed = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/20221103/'):
        if dir[0] == '.':
            continue
        if dir[0] == 'iso_start':
            continue
        fixed.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/20221103/' + dir + '/').spectra())
    fixed = spectrum.Spectra(fixed)

    dopamine = spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/neuro_con/dopamine60s/dopamine500/').spectraSum()
    GABA = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20221118/GABA500/60s/').spectraSum()
    GABA.label = 'GABA'
    glut = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20221124/glutamate50/60s/').spectraSum()
    glut.label = 'glut'
    neuros = dopamine.addSpectra(GABA)
    neuros.add(glut)
    neuros.removeThermalNoise(bg)
    neuros.cut(60, -5)
    neuros.normalizeIntegration()
    # neuros.display()

    putamen_fixed = fixed.getLabelSpectra('thalamus5')
    putamen_fixed.changeLabel('putamen_fixed')

    putamen.add(putamen6, putamen_fixed)
    putamen.removeThermalNoise(bg)
    putamen.fixAberations()
    putamen.cut(50, -5)
    putamen.normalizeIntegration()
    # putamen.displayMeanSTD()
    # putamen.display()
    putamen.pca()
    PC1 = putamen.PC[0]
    PC2 = putamen.PC[1]
    PC3 = putamen.PC[2]
    PC4 = putamen.PC[3]
    PC5 = putamen.PC[4]
    # PC3.factor(-1)
    # neuros.add(PC1)
    glut = dopamine.addSpectra(PC1)
    glut.add(PC2, PC3, PC4, PC5)
    glut.normalizeIntegration()
    glut.display()

    # putamen.pcaScatterPlot(1, 2)

    # putamen.pcaScatterPlot(3, 4)

    # putamen.pcaScatterPlot(5, 6)

    # putamen.pcaScatterPlot(7, 8)

    # putamen.pcaDisplay(1, 2, 3, 4)

    # putamen.pcaDisplay(5, 6, 7, 8)

    # putamen.pcaDisplay(3, WN=True)

def Neuro_con():
    dist_water = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/20221116/DWater/60sec/').spectraSum()
    bg = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/20221116/darkPM/').spectraSum()
    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/neuro_con/dopamine60s/'):
        if dir[0] == '.':
            continue
        data.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/neuro_con/dopamine60s/' + dir + '/').spectraSum())
    data = spectrum.Spectra(data)
    # data.removeThermalNoise(bg)
    data.subtract(dist_water)
    data.fixAberations()
    # data.cut()
    data.displayMeanSTD()


def ThermalNoise():
    day1AM = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20221116/darkAM/1min/').spectra()
    day1PM = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20221116/darkPM/').spectra()
    day2AM = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20221117/darkAM/').spectra()
    day2PM = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20221117/darkPM/').spectra()
    day3AM = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20221118/darkAM/').spectra()
    day4AM = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20221121/darkAM/').spectra()
    day4AMfin = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20221121/darkAMfin/').spectra()
    day4PM = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20221121/darkPM/').spectra()
    day4PMfin = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20221121/darkPMfin/').spectra()
    day5AM = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20221124/darkAM/').spectra()
    day5PM = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20221124/darkPM/').spectra()

    day1AM.add(day1PM, day2AM, day2PM, day3AM, day4AM, day4PM, day4PMfin, day4AMfin, day5AM, day5PM)
    day1AM.display(label=False)

def isoVerif():
    day1AM = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20221116/day1_isoAM/').spectra()
    day1PM = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20221116/day1_isoPM/').spectra()
    day2AMg = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20221117/day2_isoAM/optDist16100/').spectra()
    day2AMb = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20221117/day2_isoAM/18650/').spectra()
    day2PM = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20221117/day2_isoPM/').spectra()
    day2PMfin = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20221117/day2_isofin/').spectra()
    day3AM = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20221118/day3_isoAM/').spectra()
    day3PM = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20221118/day3_isofin/').spectra()
    day4AM = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20221121/day4_isoAM/').spectra()
    day4AMfin = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20221121/day4_isoAMfin/').spectra()
    day4PM = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20221121/day4_isoPM/').spectra()
    day4PMfin = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20221121/day4_isoPMfin/').spectra()
    day5AM = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20221124/day5_isoAM/').spectra()
    day5PM = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20221124/day5_isoPM/').spectra()

    day1AM.add(day1PM, day2AMg, day2AMb, day2PM, day2PMfin, day3AM, day3PM, day4AM, day4PM, day4PMfin, day4AMfin, day5AM, day5PM)
    day1AM.displayMeanSTD()

# VealBrain()
# FixedVSNotFixed()
Neuro_con()
# ThermalNoise()
# isoVerif()
