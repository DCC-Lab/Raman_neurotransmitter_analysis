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


def PCADeapoliLiveMonkeyDataSTNr():
    STNr = spectrum.Acquisition('/Users/antoinerousseau/Desktop/ddpaoli/20161103_InVivoMonkeySurgery/RSTN_labelfixed/', fileType='Depaoli').spectra()
    # STNr.removeSpectra(0, 7)
    STNr_label = BarCode('/Users/antoinerousseau/Downloads/STNr_DDplot_profile_2.csv', len(STNr.spectra), 240, 185, 60)
    STNr.changeLabel(STNr_label)
    # STNr.shiftSpectra(1)
    # STNr.removeSpectra(0, 67)
    # STNr.cut(500, 650, WL=True)
    # STNr.smooth()
    STNr.cut(350, 800, WL=True)
    # STNr.normalizeIntegration()
    # STNr.display3Colored(label1='WHITE', label2='GREY', label3="MIXED", WN=False)
    # STNr.polyfit(4)
    STNr.pca()
    STNr.plotPCOnBarCode(1)
    STNr.plotPCOnBarCode(2)
    STNr.plotPCOnBarCode(3)
    STNr.plotPCOnBarCode(4)
    STNr.plotPCOnBarCode(5)
    STNr.pcaScatterPlot(1, 2)

    STNr.pcaScatterPlot(3, 4)

    STNr.pcaScatterPlot(5, 6)
    STNr.pcaDisplay(1, 2, 3)
    STNr.pcaDisplay(4, 5)

def PCADeapoliLiveMonkeyDataSTNl():
    STNl = spectrum.Acquisition('/Users/antoinerousseau/Desktop/ddpaoli/20161103_InVivoMonkeySurgery/20161103_DBSLead1_STNLeftHem/', fileType='Depaoli').spectra()
    STNl_label = BarCode('/Users/antoinerousseau/Downloads/2016-DRS-OFFleft(STN)-MonkeyBrain-barcodeGWM.csv', len(STNl.spectra), 250, 90, 50)
    STNl.changeLabel(STNl_label)
    # STNl.shiftSpectra(12)
    # STNl.removeLabel('MIXED')
    # STNl.savgolFilter()

    STNl.cut(500, 650, WL=True)
    STNl.normalizeIntegration()
    STNl.display3ColoredMeanSTD(label1='WHITE', label2='GREY', label3="MIXED", WN=False)
    # STNl.polyfit(4)
    # STNl.ldaScatterPlot(1)
    STNl.pca()
    STNl.plotPCOnBarCode(1)
    STNl.plotPCOnBarCode(2)
    STNl.plotPCOnBarCode(3)
    STNl.plotPCOnBarCode(4)
    STNl.plotPCOnBarCode(5)
    STNl.plotPCOnBarCode(6)
    STNl.plotPCOnBarCode(7)
    #
    # STNl.pcaScatterPlot(1, 2)
    #
    # STNl.pcaScatterPlot(3, 4)
    #
    # STNl.pcaDisplay(1, 2)
    # STNl.pcaDisplay(3, 4)


def PCADeapoliLiveMonkeyDataGPIl():
    GPIl = spectrum.Acquisition('/Users/antoinerousseau/Desktop/ddpaoli/20161103_InVivoMonkeySurgery/20161103_MERProbe_GPiLeftHem/', fileType='Depaoli').spectra()
    GPIl_label = BarCode('/Users/antoinerousseau/Downloads/2016-DRS-GPeleft(GPi)-MonkeyBrain-barcodeGWM.csv', len(GPIl.spectra), 250, 70, 40)
    GPIl.changeLabel(GPIl_label)

    # GPIl.display3Colored(label1='WHITE', label2='GREY', label3="MIXED", WN=False)
    # GPIl.polyfit(4)
    GPIl.pca()
    GPIl.plotPCOnBarCode(1, GPIl_label)
    GPIl.plotPCOnBarCode(2, GPIl_label)
    GPIl.plotPCOnBarCode(3, GPIl_label)
    GPIl.plotPCOnBarCode(4, GPIl_label)
    # GPIl.pcaScatterPlot(1, 2)
    #
    # GPIl.pcaScatterPlot(3, 4)
    #
    # GPIl.pcaScatterPlot(5, 6)
    GPIl.pcaDisplay(1, 2, 3)
    # GPIl.pcaDisplay(4, 5)


def PCAOnAllMonkeyData():
    WR = spectrum.Acquisition('/Users/antoinerousseau/Desktop/ddpaoli/20161128_ProbeCharacterization/WR/', fileType='Depaoli').spectraSum()
    WR.integrationTime = 26 * 0.025

    # GPIl = spectrum.Acquisition('/Users/antoinerousseau/Desktop/ddpaoli/20161103_InVivoMonkeySurgery/20161103_MERProbe_GPiLeftHem/', fileType='Depaoli').spectra()
    # GPIl_label = BarCode('/Users/antoinerousseau/Downloads/2016-DRS-GPeleft(GPi)-MonkeyBrain-barcodeGWM.csv', len(GPIl.spectra), 250, 70, 40)
    # GPIl.changeLabel(GPIl_label)
    STNr = spectrum.Acquisition('/Users/antoinerousseau/Desktop/ddpaoli/20161103_InVivoMonkeySurgery/RSTN_labelfixed/',
                                fileType='Depaoli').spectra()
    STNr = STNr.getAbsorbance(WR)
    STNr_label = BarCode('/Users/antoinerousseau/Downloads/2016-DRS-STNright-MonkeyBrain-barcodeGWM.csv',
                         len(STNr.spectra), 250, 150, 100)
    STNr.changeLabel(STNr_label)
    STNr.addAnnotation('STNr')
    STNl = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/ddpaoli/20161103_InVivoMonkeySurgery/20161103_DBSLead1_STNLeftHem/',
        fileType='Depaoli').spectra()
    STNl = STNl.getAbsorbance(WR)
    STNl_label = BarCode('/Users/antoinerousseau/Downloads/2016-DRS-OFFleft(STN)-MonkeyBrain-barcodeGWM.csv',
                         len(STNl.spectra), 250, 90, 50)
    STNl.changeLabel(STNl_label)
    STNl.addAnnotation('STNl')
    data = STNr
    data.add(STNl)
    # data.savgolFilter()
    # data.cut(380, 700, WL=True)
    data.displayMeanSTD()
    # data.normalizeIntegration()
    # data.displayMeanSTD(WN=False)
    # data.pca()
    # data.pcaScatterPlot(1, 2)
    # data.pcaScatterPlot(3, 4)
    # data.pcaScatterPlot(1, 3)
    # data.pcaDisplay(1, 2, 3)
    # data.pcaDisplay(4, 5)


def getMyBloodAbsorption():
    blood = spectrum.Acquisition('/Users/antoinerousseau/Downloads/DRS_006/AntoineBLOOD/blood/',
                                 fileType='USB2000').spectra()
    blood_ref = spectrum.Acquisition('/Users/antoinerousseau/Downloads/DRS_006/AntoineBLOOD/ref/',
                                     fileType='USB2000').spectraSum()
    blood.cut(50, None)
    blood_ref.cut(50, None)
    blood.display(WN=False)
    blood_abs = blood.getAbsorbance(blood_ref)
    blood_abs.display(WN=False)
    blooderini = blood_abs.sumSpec()
    blooderini.display(WN=False)

def find_mean(x, y):
    for i in range(len(x)):
        if y[i] >= 0.5:
            break
        else:
            continue
    return x[i]

def derivate(x, y):
    big_x = 0
    for i in range(len(x)):
        if x[i]-x[i-1] > big_x:
            big_x = x[i]-x[i-1]
    # first_comp = big_x*(y[1]-y[0])/(x[1]-x[0])
    dy = [0]
    dy_norm = []
    for i in range(1, len(x)):
        dy.append(big_x*(y[i]-y[i-1])/(x[i]-x[i-1]))
    dy_max = np.amax(dy)
    for j in dy:
        dy_norm.append(j/dy_max)
    return dy_norm

def gauss_func(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def get_FWHM(mean, sigma):
    FWHM = 2.335*sigma
    x1 = mean - FWHM/2
    x2 = mean + FWHM/2
    return x1, x2, -FWHM

def KnifeEdgeTrick(path, stepsize=50):
    data = spectrum.Acquisition(path, fileType='USB2000').spectra()
    data.display()
    x = []
    curr_dist = 0
    for i in range(len(data.spectra)):
        x.append(curr_dist)
        curr_dist += stepsize
    y = []
    for spec in data.spectra:
        y.append(np.mean(spec.counts))
    max = np.amax(y)
    normalized_y = []
    for i in y:
        i = i / max
        normalized_y.append(i)
    y = normalized_y
    x = np.array(x)
    approx_mean = find_mean(x, y)
    print(y)
    print('x = ', x)
    print(approx_mean)
    approx_sigma = sum(derivate(x, y) * (x - approx_mean)**2)/len(x)
    # popt, pcov = curve_fit(gauss_func, x, derivate(x, y), p0=[1, approx_mean, approx_sigma])
    # x1, x2, FWHM = get_FWHM(popt[1], popt[2])

    # Présenntation graphique
    plt.plot(x, y, label='Normalized ERF')
    # plt.plot(x, gauss_func(x, *popt), label='Gaussian fit of ERF\'s derivate')
    # plt.plot(x, derivate(x, y))
    # plt.axvline(x1, ymin=0, ymax=1, linestyle='--', color='r', label='Gaussian\'s FHWM=%i um' % FWHM)
    # plt.axvline(x2, ymin=0, ymax=1, linestyle='--', color='r')
    plt.xlabel('Micrometric stage position [um]')
    plt.ylabel('Normalized count number [-]')
    plt.legend()
    plt.show()

def PCAOnAbsorbance(path, ref):
    WR = spectrum.Acquisition(ref, fileType='Depaoli').spectraSum()
    WR.integrationTime = 26 * 0.025
    LSTN = spectrum.Acquisition(path, fileType='Depaoli').spectra()
    STNl_label = BarCode('/Users/antoinerousseau/Downloads/2016-DRS-OFFleft(STN)-MonkeyBrain-barcodeGWM.csv',
                         len(LSTN.spectra), 250, 90, 50)
    LSTN.changeLabel(STNl_label)
    LSTNabs = LSTN.getAbsorbance(WR)
    LSTNabs.cut(430, 900, WL=True)
    LSTNabs.pca()

    LSTNabs.pcaScatterPlot(1, 2)

    # LSTNabs.pcaScatterPlot(3, 4)
    #
    # LSTNabs.pcaScatterPlot(5, 6)

    LSTNabs.pcaDisplay(1, 2)
    LSTNabs.pcaDisplay(3, 4)


    # LSTNabs.display(WN=False)

def LabDRS():
    WR = spectrum.Acquisition('/Users/antoinerousseau/Downloads/DRS_006/AntoineBLOOD/ref/', fileType='USB2000').spectra()
    GREY = []
    WHITE = []
    for dir in os.listdir('/Users/antoinerousseau/Downloads/DRS_006/grey/'):
        if dir[0] == '.':
            continue
        GREY.append(spectrum.Acquisition('/Users/antoinerousseau/Downloads/DRS_006/grey/' + dir + '/',
                                         fileType='USB2000').spectra())
    for dir in os.listdir('/Users/antoinerousseau/Downloads/DRS_006/white/'):
        if dir[0] == '.':
            continue
        WHITE.append(spectrum.Acquisition('/Users/antoinerousseau/Downloads/DRS_006/white/' + dir + '/',
                                          fileType='USB2000').spectra())

    GREY = spectrum.Spectra(GREY)
    # GREY = GREY.getAbsorbance(WR)
    GREY.changeLabel('GREY')
    WHITE = spectrum.Spectra(WHITE)
    # WHITE = WHITE.getAbsorbance(WR)
    WHITE.changeLabel('WHITE')

    data = GREY
    data.add(WHITE, WR)
    # data.cut(380, 700, WL=True)
    data.normalizeIntegration()
    data.displayMeanSTD(WN=False)
    # data.normalizeIntegration()
    # data.savgolFilter()
    # data.cut(530, 630, WL=True)
    # data.display2Colored(label1='WHITE', label2='GREY', WN=False)
    data.pca()
    #
    # data.pcaScatterPlot(1, 2)
    #
    # data.pcaScatterPlot(3, 4)
    #
    # data.pcaScatterPlot(5, 6)
    #
    data.pcaDisplay(1, 2, 3)
    data.pcaDisplay(4, 5, 6)



def shavDataRaw():
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

    #get labels
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

    DRS_RGPI_label = BarCode('/Users/antoinerousseau/Downloads/RGPi.csv', len(DataSha_RGPI), 245, 180, 105)
    DRS_RSTN_label = BarCode('/Users/antoinerousseau/Downloads/RSTN.csv', len(DataSha_RSTN), 254, 180, 105)
    DRS_ROFF_label = BarCode('/Users/antoinerousseau/Downloads/Roff.csv', len(DataSha_ROFF), 254, 180, 105)
    DRS_LGPI_label = BarCode('/Users/antoinerousseau/Downloads/LGPi.csv', len(DataSha_LGPI), 254, 180, 105)
    DRS_LSTN_label = BarCode('/Users/antoinerousseau/Downloads/LSTN.csv', len(DataSha_LSTN), 254, 180, 105)
    DRS_LOFF_label = BarCode('/Users/antoinerousseau/Downloads/Loff.csv', len(DataSha_LOFF), 254, 180, 105)

    #get rid of crap data in Right side
    DataSha_RSTN = np.delete(DataSha_RSTN, np.s_[1::2], 0)
    DRS_RSTN_label = np.delete(DRS_RSTN_label, np.s_[1::2], 0)
    DataSha_RGPI = np.delete(DataSha_RGPI, np.s_[1::2], 0)
    DRS_RGPI_label = np.delete(DRS_RGPI_label, np.s_[1::2], 0)
    DataSha_ROFF = np.delete(DataSha_ROFF, np.s_[1::2], 0)
    DRS_ROFF_label = np.delete(DRS_ROFF_label, np.s_[1::2], 0)

    #get data as Spectrum objects
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


    DRS_LOFF.changeLabel('LOFF')
    DRS_LSTN.changeLabel('LSTN')
    DRS_LGPI.changeLabel('LGPI')
    DRS_ROFF.changeLabel('ROFF')
    DRS_RSTN.changeLabel('RSTN')
    DRS_RGPI.changeLabel('RGPI')


    LOFF_abs = DRS_LOFF.getAbsorbance()
    LOFF_abs.display()
    # data = DRS_RSTN
    # data.add(DRS_LOFF, DRS_LGPI, DRS_LSTN, DRS_ROFF, DRS_RGPI)
    # data.cut(450, 590, WL=True)
    # data.remove(529, 543, WL=True)
    # data.remove(598, 608, WL=True)
    # data.pca()
    # data.shiftSpectra(7)
    # data.displayMeanSTD()
    # data.pcaScatterPlot(3, 5)
    # data.pcaDisplay(3, 5)
    # data.plotPCOnBarCode(1)
    # data.plotPCOnBarCode(2)
    # data.plotPCOnBarCode(3)
    # data.plotPCOnBarCode(4)
    # data.plotPCOnBarCode(5)
    # data.plotPCOnBarCode(6)
    # data.plotPCOnBarCode(7)






def RMLensesOptimisation():
    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/20220921/'):
        if dir[0] == '.':
            continue
        data.append(spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220921/' + dir + '/').spectra())

    data = spectrum.Spectra(data)
    data.displayMeanSTD()

def SheepBrainRaman():
    bg = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220929/morning_verif/darknoise/').spectraSum()
    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/20220929/data_filed_by_region/'):
        if dir[0] == '.':
            continue
        data.append(spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220929/data_filed_by_region/' + dir + '/').spectra())
    data = spectrum.Spectra(data)
    data.removeThermalNoise(bg)
    # data.normalizeIntegration()
    data.fixAberations()
    data.smooth()
    data.cut(450, 1600, WN=True)
    # data.savgolFilter(window_length=5)
    data.polyfit(4)
    data.displayMeanSTD(WN=True)
    data.pca()
    # data.lda()
    data.ldaScatterPlot(1)

    # data.pcaScatterPlot(1, 2)

    # data.pcaScatterPlot(3, 4)
    # data.pcaScatterPlot(3, 2)
    # data.pcaScatterPlot(3, 1)
    # data.pcaScatterPlot(5, 6)

    # data.pcaScatterPlot(7, 8)

    # data.pcaScatterPlot(9, 10)

    # data.pcaDisplay(1, 2, 3)
    # data.pcaDisplay(4, 5, 6)
    # data.pcaDisplay(7)


def ScaleWhiteRef():
    spec = []
    for i in range(100):
        rand_dude = random.uniform(0, 1)
        WR = spectrum.Acquisition('/Users/antoinerousseau/Downloads/DRS_006/AntoineBLOOD/ref/',
                                  fileType='USB2000').spectraSum()
        WR.factor(rand_dude)
        spec.append(WR)

    data = spectrum.Spectra(spec)
    mean_data = np.mean(data.data, axis=0)
    mean_spec = spectrum.Spectrum(data.spectra[0].wavelenghts, mean_data, 1, 'mean_spectrum')
    data.add(mean_spec)
    data.pca()
    data.pcaDisplay(1)


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
    # data.smooth()
    data.cut(400, 3012, WN=True)
    # data.normalizeIntegration()
    # data.polyfit(3, replace=True)
    data.displayMeanSTD()
    data.pca()

    data.pcaScatterPlot(1, 2)

    data.pcaScatterPlot(3, 4)

    data.pcaScatterPlot(5, 6)

    data.pcaScatterPlot(7, 8)

    data.pcaScatterPlot(9, 10)

    data.pcaDisplay(1, 2, 3)
    data.pcaDisplay(4, 5, 6)
    data.ldaScatterPlot(1)


def ElaheData():
    bg = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220929/morning_verif/darknoise/').spectraSum()
    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/20221107/probe/'):
        if dir[0] == '.':
            continue
        data.append(
            spectrum.Acquisition('/Users/antoinerousseau/Desktop/20221107/probe/' + dir + '/').spectraSum())
    data = spectrum.Spectra(data)
    data.removeThermalNoise(bg)
    data.cut(2800, 3000, WN=True)
    # data.pca()
    data.smooth()
    data.display()

    # data.pcaScatterPlot(1, 2)

    # data.pcaScatterPlot(3, 4)

    # data.pcaDisplay(1, 2, 3)



# shavDataRaw()
# PCADeapoliLiveMonkeyDataSTNl()
# KnifeEdgeTrick('/Users/antoinerousseau/Downloads/Field_of_view/Grey_IMV/')
# PCAOnAbsorbance('/Users/antoinerousseau/Desktop/ddpaoli/20161103_InVivoMonkeySurgery/20161103_DBSLead1_STNLeftHem/',
#                 '/Users/antoinerousseau/Desktop/ddpaoli/20161128_ProbeCharacterization/WR/')
# RMLensesOptimisation()
# PCAOnAllMonkeyData()
# LabDRS()
VealBrain()
# ScaleWhiteRef()
# ElaheData()
