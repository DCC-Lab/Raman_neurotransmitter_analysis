import spectrum
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
from dcclab.database import *
import random


def BarCode(Section, length, TW, TM, TG):
    TRUTH = pd.read_csv(Section)['Gray_Value'].to_numpy()
    step_size = len(TRUTH) / length
    GL_TRUTH = []
    for i in range(length):
        GL_TRUTH.append(TRUTH[int(i * step_size)])
    GL_TRUTH = np.array(GL_TRUTH)
    GL_TRUTH[GL_TRUTH > TW] = 1
    GL_TRUTH[GL_TRUTH > TM] = 2
    GL_TRUTH[GL_TRUTH > TG] = 3
    GL_TRUTH = np.where(GL_TRUTH == 1, 'WHITE', GL_TRUTH)
    GL_TRUTH = np.where(GL_TRUTH == '2.0', 'MIXED', GL_TRUTH)
    GL_TRUTH = np.where(GL_TRUTH == '3.0', 'GREY', GL_TRUTH)

    return GL_TRUTH


def _getShavData():
    db = SpectraDB(databaseURL="mysql+ssh://dcclab@cafeine3.crulrg.ulaval.ca:cafeine3.crulrg.ulaval.ca/dcclab@labdata")
    db.describeDatasets()
    DataSha_RGPI = db.getSpectralDataFrame(datasetId="SHAVASANA-001", id1="DRS", id2="RGPI")
    DataSha_RGPI_x = np.array(DataSha_RGPI.index)
    DataSha_RGPI = DataSha_RGPI.to_numpy().T

    DataSha_RSTN = db.getSpectralDataFrame(datasetId="SHAVASANA-001", id1="DRS", id2="RSTN")
    DataSha_RSTN_x = np.array(DataSha_RSTN.index)
    DataSha_RSTN = DataSha_RSTN.to_numpy().T

    DataSha_ROFF = db.getSpectralDataFrame(datasetId="SHAVASANA-001", id1="DRS", id2="ROFF")
    DataSha_ROFF_x = np.array(DataSha_ROFF.index)
    DataSha_ROFF = DataSha_ROFF.to_numpy().T

    DataSha_LGPI = db.getSpectralDataFrame(datasetId="SHAVASANA-001", id1="DRS", id2="LGPI")
    DataSha_LGPI_x = np.array(DataSha_LGPI.index)
    DataSha_LGPI = DataSha_LGPI.to_numpy().T

    DataSha_LSTN = db.getSpectralDataFrame(datasetId="SHAVASANA-001", id1="DRS", id2="LSTN")
    DataSha_LSTN_x = np.array(DataSha_LSTN.index)
    DataSha_LSTN = DataSha_LSTN.to_numpy().T

    DataSha_LOFF = db.getSpectralDataFrame(datasetId="SHAVASANA-001", id1="DRS", id2="LOFF")
    DataSha_LOFF_x = np.array(DataSha_LOFF.index)
    DataSha_LOFF = DataSha_LOFF.to_numpy().T

    # get labels
    DRS_RGPI_label = BarCode('/Users/antoinerousseau/Downloads/RGPI_barcode.csv', len(DataSha_RGPI), 245, 180, 105)
    DRS_RSTN_label = BarCode('/Users/antoinerousseau/Downloads/RSTN_barcode.csv', len(DataSha_RSTN), 240, 180, 40)
    DRS_ROFF_label = BarCode('/Users/antoinerousseau/Downloads/Roff.csv', len(DataSha_ROFF), 254, 180, 105)
    DRS_LGPI_label = BarCode('/Users/antoinerousseau/Downloads/LGPi.csv', len(DataSha_LGPI), 245, 180, 105)
    DRS_LSTN_label = BarCode('/Users/antoinerousseau/Downloads/LSTN (1).csv', len(DataSha_LSTN), 245, 180, 100)
    DRS_LOFF_label = BarCode('/Users/antoinerousseau/Downloads/LOFF (1).csv', len(DataSha_LOFF), 245, 180, 105)

    # get rid of crap data in Right side
    DataSha_RSTN = np.delete(DataSha_RSTN, np.s_[1::2], 0)
    DRS_RSTN_label = np.delete(DRS_RSTN_label, np.s_[1::2], 0)
    DataSha_RGPI = np.delete(DataSha_RGPI, np.s_[1::2], 0)
    DRS_RGPI_label = np.delete(DRS_RGPI_label, np.s_[1::2], 0)
    DataSha_ROFF = np.delete(DataSha_ROFF, np.s_[1::2], 0)
    DRS_ROFF_label = np.delete(DRS_ROFF_label, np.s_[1::2], 0)

    # get data as Spectrum objects
    DRS_RGPI = spectrum.ArrayToSpectra(DataSha_RGPI_x, DataSha_RGPI, label=DRS_RGPI_label).asSpectra()
    DRS_RGPI.addAnnotation('RGPI')
    DRS_RSTN = spectrum.ArrayToSpectra(DataSha_RSTN_x, DataSha_RSTN, label=DRS_RSTN_label).asSpectra()
    DRS_RSTN.addAnnotation('RSTN')
    DRS_ROFF = spectrum.ArrayToSpectra(DataSha_ROFF_x, DataSha_ROFF, label=DRS_ROFF_label).asSpectra()
    DRS_ROFF.addAnnotation('ROFF')
    DRS_LGPI = spectrum.ArrayToSpectra(DataSha_LGPI_x, DataSha_LGPI, label=DRS_LGPI_label).asSpectra()
    DRS_LGPI.addAnnotation('LGPI')
    DRS_LSTN = spectrum.ArrayToSpectra(DataSha_LSTN_x, DataSha_LSTN, label=DRS_LSTN_label).asSpectra()
    DRS_LSTN.addAnnotation('LSTN')
    DRS_LOFF = spectrum.ArrayToSpectra(DataSha_LOFF_x, DataSha_LOFF, label=DRS_LOFF_label).asSpectra()
    DRS_LOFF.addAnnotation('LOFF')

    return DRS_LOFF, DRS_LGPI, DRS_LSTN, DRS_ROFF, DRS_RGPI, DRS_RSTN


def AllRawTracks():
    DRS_LOFF, DRS_LGPI, DRS_LSTN, DRS_ROFF, DRS_RGPI, DRS_RSTN = _getShavData()

    DRS_LOFF.changeLabel('LOFF')
    DRS_LSTN.changeLabel('LSTN')
    DRS_LGPI.changeLabel('LGPI')
    DRS_ROFF.changeLabel('ROFF')
    DRS_RSTN.changeLabel('RSTN')
    DRS_RGPI.changeLabel('RGPI')

    # # Display all tracks mean + STD
    # data = DRS_RSTN
    # data.add(DRS_LOFF, DRS_LGPI, DRS_LSTN, DRS_ROFF, DRS_RGPI)
    # data.displayMeanSTD()

    # # Display LSTN raw
    # DRS_LSTN.display(label=False, WN=False)

    # # Display LGPI raw
    # DRS_LGPI.display(label=False, WN=False)

    # # Display LOFF raw
    # DRS_LOFF.display(label=False, WN=False)

    # # Display RSTN raw
    # DRS_RSTN.display(label=False, WN=False)

    # # Display RGPI raw
    # DRS_RGPI.display(label=False, WN=False)

    # # Display ROFF raw
    # DRS_ROFF.display(label=False, WN=False)


def normalizedLeftSpectra():
    DRS_LOFF, DRS_LGPI, DRS_LSTN, DRS_ROFF, DRS_RGPI, DRS_RSTN = _getShavData()

    DRS_LOFF.changeLabel('LOFF')
    DRS_LSTN.changeLabel('LSTN')
    DRS_LGPI.changeLabel('LGPI')
    DRS_ROFF.changeLabel('ROFF')
    DRS_RSTN.changeLabel('RSTN')
    DRS_RGPI.changeLabel('RGPI')

    # # Display left side tracks normalized
    # data = DRS_LSTN
    # data.add(DRS_LGPI, DRS_LOFF)
    # data.normalizeIntegration()
    # data.displayMeanSTD()

    # # Display LSTN raw
    # DRS_LSTN.normalizeIntegration()
    # DRS_LSTN.display(label=False, WN=False)
    #
    # # Display LGPI raw
    # DRS_LGPI.normalizeIntegration()
    # DRS_LGPI.display(label=False, WN=False)
    #
    # # Display LOFF raw
    # DRS_LOFF.normalizeIntegration()
    # DRS_LOFF.display(label=False, WN=False)


def polyfitLeftSPectra():
    DRS_LOFF, DRS_LGPI, DRS_LSTN, DRS_ROFF, DRS_RGPI, DRS_RSTN = _getShavData()

    DRS_LOFF.changeLabel('LOFF')
    DRS_LSTN.changeLabel('LSTN')
    DRS_LGPI.changeLabel('LGPI')
    DRS_ROFF.changeLabel('ROFF')
    DRS_RSTN.changeLabel('RSTN')
    DRS_RGPI.changeLabel('RGPI')

    # Display left side tracks normalized
    # data = DRS_LSTN
    # data.add(DRS_LGPI, DRS_LOFF)
    # data.cut(5, None)
    # data.polyfit(5, replace=True)
    # data.displayMeanSTD()

    # # Display LSTN raw
    # DRS_LSTN.polyfit(5, replace=True)
    # DRS_LSTN.display(label=False, WN=False)
    #
    # # Display LGPI raw
    # DRS_LGPI.polyfit(5, replace=True)
    # DRS_LGPI.display(label=False, WN=False)
    #
    # # Display LOFF raw
    # DRS_LOFF.polyfit(5, replace=True)
    # DRS_LOFF.display(label=False, WN=False)


def RightSideTreatement():
    DRS_LOFF, DRS_LGPI, DRS_LSTN, DRS_ROFF, DRS_RGPI, DRS_RSTN = _getShavData()

    DRS_LOFF.changeLabel('LOFF')
    DRS_LSTN.changeLabel('LSTN')
    DRS_LGPI.changeLabel('LGPI')
    DRS_ROFF.changeLabel('ROFF')
    DRS_RSTN.changeLabel('RSTN')
    DRS_RGPI.changeLabel('RGPI')

    # # Remove unwanted signal for right side data
    # data = DRS_RSTN
    # data.add(DRS_RGPI, DRS_ROFF)
    # data.cut(400, 587, WL=True)
    # data.remove(513, 527, WL=True)
    # data.displayMeanSTD()

    # # Display RSTN treated
    # DRS_RSTN.cut(400, 587, WL=True)
    # DRS_RSTN.remove(513, 527, WL=True)
    # DRS_RSTN.display(label=False, WN=False)
    #
    # # Display RGPI treated
    # DRS_RGPI.cut(400, 587, WL=True)
    # DRS_RGPI.remove(513, 527, WL=True)
    # DRS_RGPI.display(label=False, WN=False)
    #
    # # Display ROFF treated
    # DRS_ROFF.cut(400, 587, WL=True)
    # DRS_ROFF.remove(513, 527, WL=True)
    # DRS_ROFF.display(label=False, WN=False)


def RightSideNormalized():
    DRS_LOFF, DRS_LGPI, DRS_LSTN, DRS_ROFF, DRS_RGPI, DRS_RSTN = _getShavData()

    DRS_LOFF.changeLabel('LOFF')
    DRS_LSTN.changeLabel('LSTN')
    DRS_LGPI.changeLabel('LGPI')
    DRS_ROFF.changeLabel('ROFF')
    DRS_RSTN.changeLabel('RSTN')
    DRS_RGPI.changeLabel('RGPI')

    # # Remove unwanted signal for right side data
    # data = DRS_RSTN
    # data.add(DRS_RGPI, DRS_ROFF)
    # data.cut(400, 587, WL=True)
    # data.remove(513, 527, WL=True)
    # data.normalizeIntegration()
    # data.displayMeanSTD()

    # Display RSTN treated
    DRS_RSTN.cut(400, 587, WL=True)
    DRS_RSTN.remove(513, 527, WL=True)
    DRS_RSTN.normalizeIntegration()
    DRS_RSTN.display(label=False, WN=False)

    # Display RGPI treated
    DRS_RGPI.cut(400, 587, WL=True)
    DRS_RGPI.remove(513, 527, WL=True)
    DRS_RGPI.normalizeIntegration()
    DRS_RGPI.display(label=False, WN=False)

    # Display ROFF treated
    DRS_ROFF.cut(400, 587, WL=True)
    DRS_ROFF.remove(513, 527, WL=True)
    DRS_ROFF.normalizeIntegration()
    DRS_ROFF.display(label=False, WN=False)


def LDA():
    DRS_LOFF, DRS_LGPI, DRS_LSTN, DRS_ROFF, DRS_RGPI, DRS_RSTN = _getShavData()


    # DRS_LSTN.pca()
    # DRS_LSTN.plotPCOnBarCode(1)

    # DRS_LGPI.shiftSpectra(6)

    # DRS_LGPI.removeLabel('MIXED')
    # DRS_LGPI.cut(474, 735, WL=True)
    # DRS_LGPI.lda(WN=False)
    # DRS_LGPI.ldaScatterPlot(1)


    # # DÉMO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    DRS_LOFF.cut(1, None)
    DRS_LOFF.smooth()
    DRS_LOFF.removeLabel('MIXED')
    DRS_LOFF.display2Colored('WHITE', 'GREY', WN=False)
    DRS_LOFF.lda(WN=False)
    DRS_LOFF.ldaScatterPlot(1)

    # DRS_RSTN.pca()
    # DRS_RSTN.shiftSpectra(-5)
    # DRS_RSTN.cut(450, 587, WL=True)
    # DRS_RSTN.remove(513, 527, WL=True)
    # # DRS_RSTN.plotPCOnBarCode(1)
    # # DRS_RSTN.savgolFilter()
    # DRS_RSTN.removeLabel('MIXED')
    # DRS_RSTN.lda(WN=False)
    # DRS_RSTN.ldaScatterPlot(1)

    # DRS_RGPI.pca()
    # DRS_RGPI.cut(425, 587, WL=True)
    # DRS_RGPI.remove(513, 527, WL=True)
    # # DRS_RGPI.plotPCOnBarCode(1)
    # DRS_RGPI.savgolFilter()
    # DRS_RGPI.removeLabel('MIXED')
    # DRS_RGPI.lda(WN=False)
    # DRS_RGPI.ldaScatterPlot(1)

    # DRS_ROFF.pca()
    # DRS_ROFF.shiftSpectra(1)
    # DRS_ROFF.cut(425, 587, WL=True)
    # DRS_ROFF.remove(513, 527, WL=True)
    # # DRS_ROFF.plotPCOnBarCode(1)
    # DRS_ROFF.savgolFilter()
    # DRS_ROFF.removeLabel('MIXED')
    # DRS_ROFF.lda(WN=False)
    # DRS_ROFF.ldaScatterPlot(1)


def ABS():
    DRS_LOFF, DRS_LGPI, DRS_LSTN, DRS_ROFF, DRS_RGPI, DRS_RSTN = _getShavData()
    WR = spectrum.Acquisition('/Users/antoinerousseau/Desktop/ddpaoli/20161128_ProbeCharacterization/WR/',
                              fileType='Depaoli').spectraSum()
    WR.integrationTime = 26 * 0.025

    # WR.display(WN=False)
    # DRS_LGPI.spectra[14].display(WN=False)
    DRS_LGPI.shiftSpectra(6)
    DRS_LGPI.displayMeanSTD()
    LGPI_abs = DRS_LGPI.getAbsorbance(WR)
    LGPI_abs.cut(380, 900, WL=True)
    LGPI_abs.displayMeanSTD(WN=False)
    # LGPI_abs.smooth()
    # LGPI_abs.cut(20, -20)
    # LGPI_abs.pca()
    # LGPI_abs.pcaScatterPlot(1, 2)
    # LGPI_abs.pcaScatterPlot(3, 4)
    # LGPI_abs.pcaDisplay(1, 2, 3)
    # LGPI_abs.lda(WN=False)
    # LGPI_abs.ldaScatterPlot(1)


def Umap():
    DRS_LOFF, DRS_LGPI, DRS_LSTN, DRS_ROFF, DRS_RGPI, DRS_RSTN = _getShavData()

    data = DRS_LGPI
    data.removeLabel('MIXED')
    print(data.labelList)
    data.normalizeIntegration()
    data.smooth()
    data.cut(375, 950, WL=True)
    # data.shiftSpectra(10)
    # data.displayMeanSTD()
    shifts = []
    accuracys = []
    shift = 0
    for k in [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]:
        shift += k
        data.shiftSpectra(k)
        best_accuracy = 0
        nb_of_equal_ratio = 0
        equal_meta_data_list = []
        for i in [10, 20, 30, 40, 50, 60]:
            for j in [0.0, 0.1, 0.3, 0.6, 0.9]:
                data.umap(n_components=2, n_neighbors=i, min_dist=j, display=False, title='NN = {0}, MD = {1}'.format(i, j))
                data.kmeanSHAV(data='UMAP2d', barcode=True, title='NN = {0}, MD = {1}'.format(i, j))
                ratio = data.kmean_accuracy_ratio
                if ratio >= best_accuracy:
                    if ratio > best_accuracy:
                        nb_of_equal_ratio = 1
                        equal_meta_data_list = []
                        equal_meta_data_list.append([i, j])
                    if ratio == best_accuracy:
                        nb_of_equal_ratio += 1
                        equal_meta_data_list.append([i, j])

                    best_accuracy = ratio
                    meta_data = [i, j]
        shifts.append(shift)
        accuracys.append(best_accuracy)
        print('SHIFT = {0}'.format(shift))
        print('Best accuracy = {0}'.format(best_accuracy))
        # print('Parameters : NN = {0}, MD = {1}'.format(meta_data[0], meta_data[1]))
        print('Nombre de solutions possédant la meilleure accuracy = {0}, les voici: {1}'.format(nb_of_equal_ratio, equal_meta_data_list))
        # DRS_LSTN.umap(n_components=2, n_neighbors=meta_data[0], min_dist=meta_data[1], display=False, title='NN = {0}, MD = {1}'.format(meta_data[0], meta_data[1]))
        # DRS_LSTN.kmean(data='UMAP2d', graph=True, barcode=True, barplot=True, title='NN = {0}, MD = {1}'.format(meta_data[0], meta_data[1]))

    plt.plot(shifts, accuracys)
    plt.xlabel('shift par itérations de 500um [-]')
    plt.ylabel('Best accuracy obtained with Umap + kmean')
    plt.show()


def tsne():
    DRS_LOFF, DRS_LGPI, DRS_LSTN, DRS_ROFF, DRS_RGPI, DRS_RSTN = _getShavData()

    DRS_LSTN.shiftSpectra(10)
    for i in [5, 10, 15, 20, 25, 30, 40, 50]:
        DRS_LSTN.tsne(n_components=2, perplexity=i)


# AllRawTracks()
# normalizedLeftSpectra()
# polyfitLeftSPectra()
# RightSideTreatement()
# RightSideNormalized()
# LDA()
# ABS()
Umap()
# tsne()