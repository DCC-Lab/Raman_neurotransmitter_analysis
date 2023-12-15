import numpy as np
from dcclab.database import *
import spectrum
import pandas as pd
import matplotlib.pyplot as plt
import os


# A NEW ERA HAS BEGUN!!!!!!!!!!!!!!!!!!!!!!!

# bg = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20220223/backgrounds/10min_0light/').spectraSum()
# nbg = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220802/bg/').spectraSum()


# watersum = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220223/eau_30sec/').spectraSum()
# watersum.removeThermalNoise(bg)
# watersum.normalizeCounts()
# watersum.polyfit(6)

# water1 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220223/eau_30sec/').spectraSum()
# water.removeThermalNoise(bg)
# water.normalizeCounts()
# water.display()

# GABA = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220315/GABA/').spectraSum()
# GABA.removeThermalNoise(bg)
# GABA.polyfit(6)
# GABA.subtract(watersum)
# GABA.display()
# GABAsum.normalizeCounts()
# GABAsum.subtract(water)
# GABAsum.fftFilter()
# GABAsum.display()




# dopamine = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220315/dopamine/').spectraSum()
# dopamine.removeThermalNoise(bg)
# water = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220420/water/').spectraSum()
# water.removeThermalNoise(bg)
# dopamine.polyfit(6)
# dopamine.subtract(watersum)
# dopamine.subtract(watersum)
# dopamine.normalizeCounts()
# dopamine.display()

# plt.plot(dopamine.wavenumbers, y1, label='dopamine raw')
# plt.plot(dopamine.wavenumbers, y2, label='dopamine not noisy')
# plt.plot(dopamine.wavenumbers, dopamine.counts, label='dopamine not noisy, no background')
# plt.xlabel('Wavenumber [cm-1]')
# plt.ylabel('Counts [-]')
# plt.legend()
# plt.show()


# dataToPCA.pca()
# dataToPCA.pcaScatterPlot(PCx=1, PCy=2)


# Photon count data

# iso100full = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20220413/100ms_iso/').spectraSum()
# iso300full = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220413/300ms_iso/').spectra()
# iso100half = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220413/100ms_iso_halfpower/').spectra()

# iso100full.removeThermalNoise(bg)
# iso100full.cut(100, None)
# iso100full.getRatioPhotonPerCount()
# iso100full.display()

# Erika data
# path = '/Users/antoinerousseau/Desktop/20220314/'
# path1 = '/Users/antoinerousseau/Desktop/20220411/'
#
# data = []
# for dir in os.listdir(path):
#     if dir[0] == '.':
#         continue
#     data.append(spectrum.Acquisition(path + dir + '/').spectra())
# for dir in os.listdir(path1):
#     if dir[0] == '.':
#         continue
#     data.append(spectrum.Acquisition(path1 + dir + '/').spectra())
#
# data = spectrum.Spectra(data)
# data.removeThermalNoise(bg)
# data.pca()
# data.pcaDisplay(1, 2, 3, 4)
# data.pcaScatterPlot(PCx=1, PCy=2)


# GABA = spectrum.Acquisition('/Users/antoinerousseau/Desktop/PM/cm_30sec_lightoff_1A_2/').spectraSum()
# water1 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220420/water/').spectraSum()
#
# x = GABA.counts
# GABA.removeThermalNoise(bg)
# GABA.normalizeIntegration()
# water.removeThermalNoise(bg)
# water.normalizeCounts()
# GABA.display(WN=False)
# water.display()

# plt.plot(GABA.wavenumbers, GABA.counts)
# plt.plot(GABA.wavenumbers, x)
# plt.show()

# plt.plot(water.wavenumbers, water.counts, label='water 4/20, integ: {0}'.format(water.integrationTime))
# plt.plot(watersum.wavenumbers, watersum.counts, label="water 2/23, integ: {0}".format(watersum.integrationTime))
# plt.ylabel('Counts')
# plt.xlabel('Wavenumbers')
# plt.legend()
# plt.show()

# D4 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220421/D4/').spectraSum()
# D3 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220421/D3/').spectraSum()
# D2 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220421/D2/').spectraSum()
# D5 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220421/D5/').spectraSum()
# D1 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220421/D1/').spectraSum()
# D6 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220421/D6/').spectraSum()
#
# bg1 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220421/background/').spectra()
#
# iso1 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220413/100ms_iso/').spectra()
# iso2 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220421/iso/').spectraSum()
# iso3 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220422/iso/').spectraSum()
# iso3.display(WN=False)
# iso2.fixSpec()
#
# iso2.setZero(20)
#
# plt.plot(iso1.spectra[0].wavenumbers, iso1.spectra[0].counts, label='old')
# plt.plot(iso2.wavenumbers, iso2.counts, label='new')
# plt.plot(iso3.wavenumbers, iso3.counts, label='new')


# plt.plot(bg1.spectra[0].wavenumbers, bg1.spectra[0].counts, label='bg1')
# plt.plot(bg1.spectra[1].wavenumbers, bg1.spectra[1].counts, label='bg2')
# plt.plot(bg1.spectra[2].wavenumbers, bg1.spectra[2].counts, label='bg3')




# plt.plot(D4.wavenumbers, D4.counts, label='D4')
# plt.plot(D3.wavenumbers, D3.counts, label='D3')
# plt.plot(D2.wavenumbers, D2.counts, label='D2')
# plt.plot(D5.wavenumbers, D5.counts, label='D5')
# plt.plot(D1.wavenumbers, D1.counts, label='D1')
# plt.plot(D6.wavenumbers, D6.counts, label='D6')
# plt.legend()
# plt.show()

# GABA = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220420/GABA_100mM/').spectraSum()

# water.normalizeCounts()
# GABA.removeThermalNoise(bg)
# GABA.subtract(water)
# GABA.display()
# GABA.normalizeCounts()
# water.removeThermalNoise(bg)
# water.normalizeCounts()



#
# plt.plot(GABA.wavenumbers, GABA.counts, 'r', label='GABA')
# plt.plot(dopamine.wavenumbers, dopamine.counts, 'k', label='Dopamine')
# plt.ylim(0, 1.03)
# plt.rcParams.update({'font.size': 24})
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)
# plt.xlabel('Wavenumbers [cm-1]', fontsize=24)
# plt.ylabel('Normalized counts [-]', fontsize=24)
# plt.legend()
# plt.show()

# GABA.subtract(water)
# GABA.smooth()
# GABA.display()



#victoria

# path = '/Users/antoinerousseau/Desktop/M83(ho)/Photopic/6mo/'
#
# data = []
# for dir in os.listdir(path):
#     if dir[0] == '.':
#         continue
#     data.append(spectrum.Acquisition(path + dir + '/', fileType='VF', extension='').spectra())
#
# data = spectrum.Spectra(data)
# # data.display(WN=False)
#
# data.pca()
#
# data.pcaDisplay(1, 2)
# data.pcaScatterPlot(1, 2)



# data spectro shifted
# iso_new = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220627/').spectra()
# iso_new.display()
# iso_new.normalizeIntegration()

# iso_good = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220413/100ms_iso/').spectra()
#
# x = iso_good.spectra[0].wavelenghts
# label = []
# for i in iso_good.spectra[:]:
#     label.append(i.label)
# integ = iso_good.spectra[:3]
# iso_good = np.array(iso_good.data)


# new_good = spectrum.ArrayToSpectra(x, iso_good, label=label).spectra
#
# new_good = spectrum.Spectra(new_good)
# new_good.display()

# constant = []
# for i in range(16):
#     constant.append(40)
#
# iso_bad = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220422/iso/').spectra()
# iso_bad = iso_bad.spectra[300]
# plt.plot(iso_bad.counts[190:205])
# plt.plot(constant, '--k')
# # plt.show()
#
# integration = 0
# for i in iso_bad.counts[190:205]:
#     integration += i - 40
#
# print(integration)
# iso_bad.normalizeIntegration()
# iso_bad.display()
# plt.plot(iso_new.wavenumbers, iso_new.counts, 'k-', label='New')
# plt.plot(iso_bad.wavenumbers, iso_bad.counts, 'r-', label='After crash')
# plt.legend()
# plt.show()

# bad = iso_bad
# old = iso_good
# new = iso_new
# bad.cut(100, None)
# old.cut(100, None)
# new.cut(100, None)
# plt.plot(old.spectra[0].wavelenghts, old.spectra[0].counts, 'k', label='Old')
# plt.plot(bad.spectra[0].wavelenghts, bad.spectra[0].counts, 'r', label='"Broken"')
# plt.plot(new.spectra[0].wavelenghts, new.spectra[0].counts, 'b', label='Repaired')
# plt.xlabel('Wavelenghts [nm]', fontsize=24)
# plt.ylabel('Counts [-]', fontsize=24)
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)
# plt.legend()
# plt.show()



# path = '/Users/antoinerousseau/Downloads/PegahAndAlexExperiment/'
#
# data = []
# for dir in os.listdir(path):
#     if dir[0] == '.':
#         continue
#     data.append(spectrum.Acquisition(path + dir + '/', fileType='USB2000').spectra())
#
# data = spectrum.Spectra(data)
# data.removeThermalNoise(bg)
# data.pca()
# data.pcaDisplay(1, 2)
# data.pcaScatterPlot(PCx=1, PCy=2)
# data.lda(SC=True, n_components=3)
# data.ldaScatterPlot(2, 3, SC=True)
# data.pca(SC=True)
# data.pcaScatterPlot(1, 2, SC=True)





# dev nonlin
# x = np.linspace(-0.01, 0.01, num=101)
#
# def spotSizeRadius(z, w0=0.0005, WL=500e-6):
#     return w0 * np.sqrt((1 + ((z * WL) / (np.pi * w0**2))**2))
#
# def intensity(x):
#     return 1  / (x**2 * np.pi)
#
# def intensity2(x):
#     return (1 / (x**2 * np.pi))**2
#
# def intensity3(x):
#     return (1 / (x**2 * np.pi))**3
#
#
# def normalise(x):
#     max = np.amax(x)
#     new = []
#     for i in x:
#         new.append(i / max)
#     return new
#
#
# y1 = []
# y = []
# y2 = []
# y3 = []
# for i in x:
#     ss = spotSizeRadius(i)
#     y.append(ss)
#     y1.append(intensity(ss))
#     y2.append(intensity2(ss))
#     y3.append(intensity3(ss))
#
# y1 = normalise(y1)
# y2 = normalise(y2)
# y3 = normalise(y3)
#
# x = np.array(x)*1000
#
# plt.plot(x, y1, label='1 photon')
# plt.plot(x, y2, label='2 photons')
# plt.plot(x, y3, label='3 photons')
# plt.legend()
# plt.xlabel('Distance from focal spot [um]')
# plt.ylabel('Relative Intensity [-]')
# plt.show()





# from dcclab.database import *
#
# db = SpectraDB()
# db.describeDatasets()
# datasetId = "DRS-003"
# spectra, spectrumIds = db.getSpectra(datasetId=datasetId)
# spectra = spectra.T
# frequency = db.getFrequencies(datasetId=datasetId)
#
# data = spectrum.ArrayToSpectra(frequency, spectra, label=spectrumIds)
# data.cleanLabel([2])
# data = data.asSpectra()

# data.display()
# data.pca()
# dumbo = data.subtractPCToData(1)
# newdata = spectrum.ArrayToSpectra(frequency, dumbo, label=spectrumIds)
# newdata.cleanLabel([2])
# newdata = newdata.asSpectra()
# newdata.pca()
# newdata.pcaDisplay(1, 2, 3)
# data.pcaDisplay(2, 3, 4)
# data.pcaScatterPlot(1, 2)

# data.lda(n_components=1, SC=True)
# data.ldaScatterPlot(1, SC=True)


# from dcclab.database import *
# from scipy.signal import savgol_filter
#
# db = SpectraDB()
#
# datasetId = "SHAVASANA-001"
# db.describeDatasets(datasetId=datasetId)  # if needed, look at it
#
# print("\nCurrently in the database, validated ids are   : ")
# print("================================================")
# values = db.getPossibleIdValues(datasetId)
# for genericIdLabel in ["id1", "id2", "id3", "id4"]:
#     print("'{0}' is one of {1}".format(genericIdLabel, values[genericIdLabel]))
#
# spectraGroup = []
# for i in range(2,42):
#     spectra = db.getSpectralDataFrame(datasetId= "SHAVASANA-001", id1= "CARS", id2= "RSTN", distance=i)
#     spectra = np.sum(spectra, axis=1)
#     spectra = savgol_filter(spectra, 25, 2)
#     spectraGroup.append(spectra)
# frequency = db.getFrequencies(datasetId= "SHAVASANA-001", id1="CARS")
#
# plt.plot(frequency, np.transpose(spectraGroup))
# plt.show()
#
# plt.plot(frequency, np.sum(np.transpose(spectraGroup), axis=1))
# plt.show()



# x=[35, 100, 150, 200]
# y1=[1800, 1480, 1375, 1130]
# y1_norm = [1, 1480/1800, 1375/1800, 1130/1800]
#
# denom = 1800/58
# y2=[1, 1480/51/denom, 1375/45/denom, 1130/43/denom]
#
# plt.plot(x, y1_norm, 'k', label='Intensity Normalized')
# plt.plot(x, y2, 'r', label='Relative Intensity Normalized')
# plt.legend()
# plt.xlabel('f of L2 in the 4F (L1: 35mm)')
# plt.show()



# A = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220711/L3ASH17.5/').spectraSum()
# B = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220711/L3ASH17.5L4ASH/').spectraSum()
# C = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220711/L4ACH50/').spectraSum()
# D = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220711/L4ASH40F/').spectraSum()
# E = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220711/L4ASH50/').spectraSum()
# data = A.addSpectra(B)
# data.add(C, D, E)
# data.cut(30, None)
# data.display()

#-------------------------------------------------------------
# data brain monkey gris et blanc

# NoLens = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220802/noL1L2_iso/').spectraSum()
# WithLens = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220802/optim_iso/').spectraSum()
# pbs1 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220802/pbs1/').spectraSum()
# pbs2 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220802/pbs2/').spectraSum()
# grey = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220802/Monkey_brain/gris/').spectra()
# white = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220802/Monkey_brain/blanc/').spectra()
#
# g_sum = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220802/Monkey_brain/gris/').spectraSum()
# g_sum.removeThermalNoise(nbg)
# g_sum.cut(30, -4)
# g_fit = g_sum.fft(return_fit=True, b=0.02, shift=105)
# grey.removeThermalNoise(nbg)
# grey.cut(30, -4)
# grey.fixAberations()
# grey.subtract(g_fit)
#
#
# w_sum = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220802/Monkey_brain/blanc/').spectraSum()
# w_sum.removeThermalNoise(nbg)
# w_sum.cut(30, -4)
# w_sum.fixAberations()
# w_fit = w_sum.fft(return_fit=True, b=0.02, shift=105)
# white.removeThermalNoise(nbg)
# white.cut(30, -4)
# white.fixAberations()
# white.subtract(w_fit)


# grey.add(white)

# grey.cut(110, None)
# grey.cut(915, -35)

# grey.displayColored(display_label=False, label1='gris', label2='blanc')
# grey.pca()
# grey.pcaScatterPlot(1, 2)
# grey.pcaScatterPlot(1, 3)
# grey.pcaScatterPlot(1, 4)
# grey.pcaScatterPlot(2, 3)
# grey.pcaScatterPlot(2, 4)
# grey.pcaScatterPlot(3, 4)
# grey.pcaScatterPlot(2, 5)
# grey.pcaScatterPlot(3, 5)
# grey.pcaScatterPlot(4, 5)

# grey.pcaDisplay(1, 2, 3, WN=True)
# grey.pcaDisplay(4, 5, 6, WN=True)
# grey.pcaDisplay(7, 8, 9, WN=True)




#shavasana ----------------------------------------
# get data
# db = SpectraDB()
# DataSha_RGPI = db.getSpectralDataFrame(datasetId="SHAVASANA-001", id1="DRS", id2="RGPI")
# DataSha_RGPI_x = np.array(DataSha_RGPI.index)
# DataSha_RGPI = DataSha_RGPI.to_numpy().T
#
# DataSha_RSTN = db.getSpectralDataFrame(datasetId="SHAVASANA-001", id1="DRS", id2="RSTN")
# DataSha_RSTN_x = np.array(DataSha_RSTN.index)
# DataSha_RSTN = DataSha_RSTN.to_numpy().T
#
# DataSha_ROFF = db.getSpectralDataFrame(datasetId="SHAVASANA-001", id1="DRS", id2="ROFF")
# DataSha_ROFF_x = np.array(DataSha_ROFF.index)
# DataSha_ROFF = DataSha_ROFF.to_numpy().T
#
# DataSha_LGPI = db.getSpectralDataFrame(datasetId="SHAVASANA-001", id1="DRS", id2="LGPI")
# DataSha_LGPI_x = np.array(DataSha_LGPI.index)
# DataSha_LGPI = DataSha_LGPI.to_numpy().T
#
# DataSha_LSTN = db.getSpectralDataFrame(datasetId="SHAVASANA-001", id1="DRS", id2="LSTN")
# DataSha_LSTN_x = np.array(DataSha_LSTN.index)
# DataSha_LSTN = DataSha_LSTN.to_numpy().T
#
# DataSha_LOFF = db.getSpectralDataFrame(datasetId="SHAVASANA-001", id1="DRS", id2="LOFF")
# DataSha_LOFF_x = np.array(DataSha_LOFF.index)
# DataSha_LOFF = DataSha_LOFF.to_numpy().T
#
# DataLab_DRS3 = db.getSpectralDataFrame(datasetId='DRS-003')
# DRS3_label = list(DataLab_DRS3)
# DataLab_DRS3_x = np.array(DataLab_DRS3.index)
# DataLab_DRS3 = DataLab_DRS3.to_numpy().T
#
# DataLab_DRS4 = db.getSpectralDataFrame(datasetId='DRS-004')
# DRS4_label = list(DataLab_DRS4)
# DataLab_DRS4_x = np.array(DataLab_DRS4.index)
# DataLab_DRS4 = DataLab_DRS4.to_numpy().T
#
#
# #get labels
# def BarCode(Section, lenght):
#     TRUTH = pd.read_csv(Section)[:-3]['Gray_Value'].to_numpy()
#     splits = np.array_split(TRUTH, lenght)
#     GL_TRUTH = []
#     for i in splits:
#         GL_TRUTH.append(np.mean(i))
#     GL_TRUTH = np.array(GL_TRUTH)
#     GL_TRUTH[GL_TRUTH > 245] = 1
#     GL_TRUTH[GL_TRUTH > 180] = 2
#     GL_TRUTH[GL_TRUTH > 105] = 3
#     GL_TRUTH = np.where(GL_TRUTH == 1, 'WHITE', GL_TRUTH)
#     GL_TRUTH = np.where(GL_TRUTH == '2.0', 'MIXED', GL_TRUTH)
#     GL_TRUTH = np.where(GL_TRUTH == '3.0', 'GREY', GL_TRUTH)
#
#     return GL_TRUTH
#
# DRS_RGPI_label = BarCode('/Users/antoinerousseau/Downloads/RGPi.csv', len(DataSha_RGPI))
# DRS_RSTN_label = BarCode('/Users/antoinerousseau/Downloads/RSTN.csv', len(DataSha_RSTN))
# DRS_ROFF_label = BarCode('/Users/antoinerousseau/Downloads/Roff.csv', len(DataSha_ROFF))
# DRS_LGPI_label = BarCode('/Users/antoinerousseau/Downloads/LGPi.csv', len(DataSha_LGPI))
# DRS_LSTN_label = BarCode('/Users/antoinerousseau/Downloads/LSTN.csv', len(DataSha_LSTN))
# DRS_LOFF_label = BarCode('/Users/antoinerousseau/Downloads/Loff.csv', len(DataSha_LOFF))
#
#
# #get rid of crap data in Right side
# DataSha_RSTN = np.delete(DataSha_RSTN, np.s_[1::2], 0)
# DRS_RSTN_label = np.delete(DRS_RSTN_label, np.s_[1::2], 0)
# DataSha_RGPI = np.delete(DataSha_RGPI, np.s_[1::2], 0)
# DRS_RGPI_label = np.delete(DRS_RGPI_label, np.s_[1::2], 0)
# DataSha_ROFF = np.delete(DataSha_ROFF, np.s_[1::2], 0)
# DRS_ROFF_label = np.delete(DRS_ROFF_label, np.s_[1::2], 0)
#
#
# #get data as Spectrum objects
# DRS_RGPI = spectrum.ArrayToSpectra(DataSha_RGPI_x, DataSha_RGPI, label=DRS_RGPI_label).asSpectra()
# DRS_RGPI.addAnnotation('RGPI')
# DRS_RSTN = spectrum.ArrayToSpectra(DataSha_RSTN_x, DataSha_RSTN, label=DRS_RSTN_label).asSpectra()
# DRS_RSTN.addAnnotation('RSTN')
# DRS_ROFF = spectrum.ArrayToSpectra(DataSha_ROFF_x, DataSha_ROFF, label=DRS_ROFF_label).asSpectra()
# DRS_ROFF.addAnnotation('ROFF')
# DRS_LGPI = spectrum.ArrayToSpectra(DataSha_LGPI_x, DataSha_LGPI, label=DRS_LGPI_label).asSpectra()
# DRS_LGPI.addAnnotation('LGPI')
# DRS_LSTN = spectrum.ArrayToSpectra(DataSha_LSTN_x, DataSha_LSTN, label=DRS_LSTN_label).asSpectra()
# DRS_LSTN.addAnnotation('LSTN')
# DRS_LOFF = spectrum.ArrayToSpectra(DataSha_LOFF_x, DataSha_LOFF, label=DRS_LOFF_label).asSpectra()
# DRS_LOFF.addAnnotation('LOFF')
#
#
#
# DRS3 = spectrum.ArrayToSpectra(DataLab_DRS3_x, DataLab_DRS3, label=DRS3_label)
# DRS3.cleanLabel([2])
# DRS3 = DRS3.asSpectra()
#
# DRS4 = spectrum.ArrayToSpectra(DataLab_DRS4_x, DataLab_DRS4, label=DRS4_label)
# DRS4.cleanLabel([2])
# DRS4 = DRS4.asSpectra()

#do shit

#get alex's lab data

# GREY = []
# WHITE = []
# MIXED = []
# for dir in os.listdir('/Users/antoinerousseau/Downloads/2022-08-30_DRS-005/Grey/'):
#     if dir[0] == '.':
#         continue
#     GREY.append(spectrum.Acquisition('/Users/antoinerousseau/Downloads/2022-08-30_DRS-005/Grey/' + dir + '/', fileType='USB2000').spectraSum())
# for dir in os.listdir('/Users/antoinerousseau/Downloads/2022-08-30_DRS-005/White/'):
#     if dir[0] == '.':
#         continue
#     WHITE.append(spectrum.Acquisition('/Users/antoinerousseau/Downloads/2022-08-30_DRS-005/White/' + dir + '/', fileType='USB2000').spectraSum())
# for dir in os.listdir('/Users/antoinerousseau/Downloads/2022-08-30_DRS-005/Mixed/'):
#     if dir[0] == '.':
#         continue
#     MIXED.append(spectrum.Acquisition('/Users/antoinerousseau/Downloads/2022-08-30_DRS-005/Mixed/' + dir + '/', fileType='USB2000').spectraSum())
#
# GREY = spectrum.Spectra(GREY)
# GREY.changeLabel('GREY')
# GREY.addAnnotation('GREY')
# WHITE = spectrum.Spectra(WHITE)
# WHITE.changeLabel('WHITE')
# WHITE.addAnnotation('WHITE')
# MIXED = spectrum.Spectra(MIXED)
# MIXED.changeLabel('MIXED')
# MIXED.addAnnotation('MIXED')

# data = GREY
# data.add(WHITE, MIXED)

# get alex and me's DRS data

# GREY = []
# WHITE = []
# blood = spectrum.Acquisition('/Users/antoinerousseau/Downloads/DRS_006/AntoineBLOOD/blood/', fileType='USB2000').spectra()
# blood_ref = spectrum.Acquisition('/Users/antoinerousseau/Downloads/DRS_006/AntoineBLOOD/ref/', fileType='USB2000').spectraSum()
# for dir in os.listdir('/Users/antoinerousseau/Downloads/DRS_006/grey/'):
#     if dir[0] == '.':
#         continue
#     GREY.append(spectrum.Acquisition('/Users/antoinerousseau/Downloads/DRS_006/grey/' + dir + '/', fileType='USB2000').spectra())
# for dir in os.listdir('/Users/antoinerousseau/Downloads/DRS_006/white/'):
#     if dir[0] == '.':
#         continue
#     WHITE.append(spectrum.Acquisition('/Users/antoinerousseau/Downloads/DRS_006/white/' + dir + '/', fileType='USB2000').spectra())
#
# GREY = spectrum.Spectra(GREY)
# GREY.changeLabel('GREY')
# GREY.addAnnotation('GREY')
# WHITE = spectrum.Spectra(WHITE)
# WHITE.changeLabel('WHITE')
# WHITE.addAnnotation('WHITE')

# data = GREY
# data.add(WHITE)
# data.cut()
# # data.normalizeIntegration()
# data.pca()
# data.pcaDisplay(1, 2, 3)




# data._standardizeData(replace=True)
# data.display3Colored(label1='GREY', label2='WHITE', label3='MIXED', WN=False)

# old = DRS3.spectra[1]
# old.normalizeIntegration()
# DRS3.spectra[1].changeXAxisValues(DRS_RGPI.spectra[0], print_info=True)
# DRS4.display()
# viande = DRS3.spectra[1].addSpectra(DRS_LOFF.spectra[0])
# viande.add(old)
# viande.normalizeIntegration()
# viande.display()
# w_sum = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220802/Monkey_brain/blanc/').spectraSum()
# YDK_data = spectrum.Acquisition('/Users/antoinerousseau/Downloads/2022-08-25-AddDataToShavasana/', fileType='USB2000').spectra()
# YDK_data.normalizeIntegration()
# YDK_data.add(DRS_LOFF)
# YDK_data.normalizeIntegration()
# YDK_data.display2Colored(label1='2022-08-25-AddDataToShavasana', label2='MIXED', WN=False, display_label=False)

# DRS_LOFF.normalizeIntegration()
#
# plt.plot(old.wavelenghts, old.counts, label='Our spectro')
# plt.plot(DRS_LOFF.spectra[0].wavelenghts, DRS_LOFF.spectra[0].counts, label='shavasana')
# plt.plot(YDK_data.spectra[0].wavelenghts, YDK_data.spectra[0].counts, label='YDK spectro')
# plt.legend()
# plt.show()


# data = DRS_RGPI
# data.add(DRS_ROFF, DRS_RSTN, DRS_LGPI, DRS_LSTN, DRS_LOFF)
# data = DRS_LGPI
# data.add(DRS_LSTN, DRS_LOFF)
# data.smooth()
# data.cut(450, 650, WL=True)

# data.removeLabel('MIXED')
# data.cut(450, 587, WL=True)
# data.remove(513, 528, WL=True)
# data.cut(528, 587, WL=True)
# data.cut(450, 513, WL=True)
# data.cut(490, 750, WL=True)
# data.polyfit(4, replace=True)
# data.kpca(nbOfComp=1)
# data.kpca(nbOfComp=2)
# data.kpca(nbOfComp=3)
# data.kpca(nbOfComp=4)
# data.kpca(nbOfComp=4)
# data.kpca(nbOfComp=4)
# data.kpca(nbOfComp=4)
# data.kpca(nbOfComp=4)
# data.kpca(nbOfComp=5)
# data.kpca(nbOfComp=7)
# data.kpca(nbOfComp=10)

# data.normalizeIntegration()
# data.removeLabel('MIXED')
# data.cut(400, 850, WL=True)
# data.fixAberations(threshold=0.01, display=True)
# data.polyfit(4, replace=True)
# data.tsne()
# data.umap()

# data.display2Colored('WHITE', 'GREY', WN=False, display_label=True)
# data.pca()
# data.removeLabel('MIXED')
# data.ldaOnPCsScatteredPlot(1, 2)
# data.pcaScatterPlot(1, 2)
# data.pcaScatterPlot(3, 4)
# data.pcaScatterPlot(5, 6)
# data.pcaScatterPlot(7, 8)
# data.pcaScatterPlot(9, 10)
# data.pcaDisplay(1, 2)
# data.pcaDisplay(4, 5, 6)
# data.pcaDisplay(7, 8, 9)
# data.umap()

# blood.cut(50, None)
# blood_ref.cut(50, None)
# blood.display(WN=False)
# blood_abs = blood.getAbsorbance(blood_ref)
# blood_abs.display(WN=False)
# blooderini = blood_abs.sumSpec()
# blooderini.display(WN=False)

def RamanBrainGrayWhite():
    grey = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220802/Monkey_brain/gris/').spectraSum()
    # grey.changeLabel('Grey')
    white = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220802/Monkey_brain/blanc/').spectraSum()
    # white.changeLabel('White')
    data = grey
    data = data.addSpectra(white)
    data.removeThermalNoise(nbg)
    data.cut(2800, 3075, WN=True)

    # data.fixAberations()
    # data.cut(2800, 2950, WN=True)
    # data.normalizeIntegration()
    # data.spectra[0].display()
    # data.polyfit(3, replace=True)
    # data.smooth()
    data.display(WN=True)
    # data.pca()
    # data.ldaScatterPlot(1)
    # data.pcaScatterPlot(1, 2)

    # data.pcaScatterPlot(3, 4)

    # data.pcaScatterPlot(5, 6)

    # data.pcaDisplay(1, 2, 3, WN=True)
    # data.pcaDisplay(4, 5, 6)


def NaClDataDisplayAndPCA():
    water = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220913/water/').spectra()
    waterSum = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220913/water/').spectraSum()
    NaCl_max = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220913/NaCl_max/').spectra()
    NaCl_few = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220913/NaCl_few/').spectra()
    waterSum.removeThermalNoise(nbg)
    waterSum.cut(20, None)
    waterSum.normalizeIntegration()

    data = NaCl_few
    data.add(NaCl_max, water)
    data.removeThermalNoise(nbg)
    data.normalizeIntegration()
    data.displayMeanSTD(WN=False)
    # waterSum.integrationTime = data.spectra[0].integrationTime
    # data.removeThermalNoise(nbg)
    # data.cut(20, None)
    # data.normalizeIntegration()
    # data.subtract(waterSum)
    #
    # data.displayMeanSTD(WN=True)


# NaClDataDisplayAndPCA()
# RamanBrainGrayWhite()

# methanol = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20220830/ethanol/').spectraSum()
# methanol.display()



def figure_seminaire():
    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/20211001/data/'):
        if dir[0] == '.':
            continue
        data.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/maitrise/DATA/20211001/data/' + dir + '/').spectra())
    data = spectrum.Spectra(data)
    data.displayMeanSTD()

def ashVSobj():
    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/20220921/'):
        if dir[0] == '.':
            continue
        data.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/maitrise/DATA/20220921/' + dir + '/').spectra())
    data = spectrum.Spectra(data)
    data.displayMeanSTD()

def l4():
    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/20211119/toplot/'):
        if dir[0] == '.':
            continue
        data.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/maitrise/DATA/20211119/toplot/' + dir + '/').spectra())
    data = spectrum.Spectra(data)
    data.displayMeanSTD()


def ldatest():
    bg = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20220802/bg/').spectraSum()
    iso = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20220929/morning_verif/iso_verif/').spectra()
    water = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20220913/water/').spectra()
    dopamine = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/neuro_con/dopamine60s/dopamine500/').spectra()
    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/20220830/'):
        if dir[0] == '.':
            continue
        data.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/maitrise/DATA/20220830/' + dir + '/').spectra())
    data = spectrum.Spectra(data)
    data.add(iso, water, dopamine)
    data.removeThermalNoise(bg)
    data.normalizeIntegration()
    # data = data.combineSpectra(add=5)
    data.smooth(n=5)
    # data.ORPL(min_bubble_widths=60, display=True)
    data.cut(800, 1300, WN=True)
    data.displayMeanSTD(WN=True)
    data.lda(display=True)
    data.ldaScatterPlot(LDx=1)
    # data.ldaScatterPlot(LDx=3)


def monomere():
    bg = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20230421/dn/').spectraSum()
    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/20230421/data/'):
        if dir[0] == '.':
            continue
        data.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/maitrise/DATA/20230421/data/' + dir + '/').spectra())
    data = spectrum.Spectra(data)
    data.removeThermalNoise(bg)
    data.combineSpectra(add=5)
    data.butterworthFilter(cutoff_frequency=8, order=3)
    data.cut(400, 1900, WN=True)
    data.displayMeanSTD()
    data.pca()

    # data.pcaScatterPlot(1, 2)
    # data.pcaScatterPlot(3, 4)
    # data.pcaScatterPlot(5, 6)
    data.pcaDisplay(1, 2)
    # data.pcaDisplay(3, 4)
    # data.pcaDisplay(5, 6)


def monomereVis():
    dn = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20230421/dn/').spectraSum()
    mono = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20230421/data/monomeres/').spectraSum()
    pbs1 = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20230421/data/pbs_quartz_2/').spectraSum()
    pbs2 = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20230421/data/pbs_quartz_3/').spectraSum()

    pbs = pbs1.addSpectra(pbs2)

    mono.smooth(n=5)
    pbs1.smooth(n=5)
    pbs2.smooth(n=5)

    # mono.butterworthFilter()
    # pbs1.butterworthFilter()
    # pbs2.butterworthFilter()

    # mono.normalizeIntegration()
    # pbs1.normalizeIntegration()
    # pbs2.normalizeIntegration()


    # mono.removeThermalNoise(dn)
    # pbs1.removeThermalNoise(dn)
    # pbs2.removeThermalNoise(dn)


    mono.subtract(pbs1)
    mono.cut(400, 1900, WN=True)

    mono.display()


def monoVSfib():
    dn = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20230421/dn/').spectraSum()
    pbs = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/alpha_syn/pbs/').spectra()
    monomere = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/alpha_syn/monomeres/').spectra()
    fibril = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/alpha_syn/fibril/').spectra()


    fibril = fibril.combineSpectra(add=10)
    pbs.add(monomere, fibril)
    pbs.removeThermalNoise(dn)
    pbs.butterworthFilter(cutoff_frequency=8, order=3)
    pbs.cut(400, 1900, WN=True)
    pbs.displayMeanSTD()

    pbs.pca()

    # pbs.pcaScatterPlot(1, 2)
    # pbs.pcaScatterPlot(3, 4)
    # pbs.pcaScatterPlot(5, 6)

    pbs.pcaDisplay(1, 2)
    pbs.pcaDisplay(3, 4)
    pbs.pcaDisplay(5, 6)






def fibril():
    dn = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20230421/dn/').spectraSum()
    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/20230323/fibrilPM/'):
        if dir[0] == '.':
            continue
        data.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/maitrise/DATA/20230323/fibrilPM/' + dir + '/').spectra())
    data = spectrum.Spectra(data)
    data.removeThermalNoise(dn)
    data.shortenLabels()
    # data.removeLabel(label='pbs')

    data.picRatio(l1=900, l2=1400)
    data.picRatio(l1=2850, l2=2900)
    data.PRScatterPlot(1, 2)

def compareSum():
    dn = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20230421/dn/').spectraSum()
    pbs = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/alpha_syn/pbs/').spectraSum()
    mono = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/alpha_syn/monomeres/').spectraSum()

    # data = pbs.addSpectra(mono)
    # data.removeThermalNoise(dn)
    # data.cut(400, 3000, WN=True)
    # data.normalizeIntegration()
    #
    # data.display()
    mono.removeThermalNoise(dn)
    pbs.removeThermalNoise(dn)
    mono.cut(400, 3000, WN=True)
    pbs.cut(400, 3000, WN=True)
    mono.normalizeIntegration()
    pbs.normalizeIntegration()
    mono.subtract(pbs)
    mono.display()

# figure_seminaire()
# ashVSobj()
# l4()


# ldatest()
# monomere( )
# monomereVis()
# monoVSfib()
# fibril()
# compareSum()

def test():
    dopamine = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/neuro_con/dopamine60s/dopamine500/').spectraSum()
    GABA = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/neuro_con/GABA60s/GABA500/').spectraSum()
    glut = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/neuro_con/glut60s/glut50/').spectraSum()

    dn = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20221124/darkPM/').spectraSum()
    data = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/neuro_con/dopamine60s/dopamine250/').spectra()
    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/neuro_con/dopamine60s/'):
        if dir[0] == '.':
            continue
        data.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/maitrise/DATA/neuro_con/dopamine60s/' + dir + '/').spectra())
    data = spectrum.Spectra(data)

    GABA_data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/neuro_con/GABA60s/'):
        if dir[0] == '.':
            continue
        GABA_data.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/maitrise/DATA/neuro_con/GABA60s/' + dir + '/').spectra())
    GABA_data = spectrum.Spectra(GABA_data)

    glut_data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/neuro_con/glut60s/'):
        if dir[0] == '.':
            continue
        glut_data.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/maitrise/DATA/neuro_con/glut60s/' + dir + '/').spectra())
    glut_data = spectrum.Spectra(glut_data)

    data.add(GABA_data, glut_data)
    data.removeThermalNoise(dn)
    data.smooth(n=5)
    data.ORPL(min_bubble_widths=70)

    dopamine.removeThermalNoise(dn)
    dopamine.smooth(n=5)
    dopamine.ORPL(min_bubble_widths=70, display=False)
    dopamine.cut(350, 1800, WN=True)
    dopamine.normalizeIntegration()
    GABA.removeThermalNoise(dn)
    GABA.smooth(n=5)
    GABA.ORPL(min_bubble_widths=70)
    GABA.cut(350, 1800, WN=True)
    GABA.normalizeIntegration()
    glut.removeThermalNoise(dn)
    glut.smooth(n=5)
    glut.ORPL(min_bubble_widths=70)
    glut.cut(350, 1800, WN=True)
    # glut.factor(10)
    glut.normalizeIntegration()
    #
    dude = dopamine.addSpectra(GABA)
    dude.add(glut)
    dude.display()

    data.cut(350, 1800, WN=True)
    data.PCAExternalComposit(composit=[dopamine.counts, GABA.counts, glut.counts], labels=['Dopamine', 'GABA', 'glut'])
    # data.PCAExternalComposit(composit=[glut.counts], labels=['glut'])
    data.pcaecScatterPlot('Dopamine')
    data.pcaecScatterPlot('GABA')
    data.pcaecScatterPlot('glut')
    # data.displayMeanSTD()
    # data.pca()
    # data.pcaDisplay(1, 2)
    data.PCAEC_KNNIndividualLabel()


def hanu():
    bg = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20220929/morning_verif/darknoise/').spectraSum()

    dopamine = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/neuro_con/dopamine60s/dopamine500/').spectraSum()
    GABA = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/neuro_con/GABA60s/GABA500/').spectraSum()
    glut = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/neuro_con/glut60s/glut50/').spectraSum()

    dn = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20221124/darkPM/').spectraSum()

    dopamine.removeThermalNoise(dn)
    dopamine.cut(2810, 3024, WN=True)
    # dopamine.ORPL(min_bubble_widths=40, display=True)
    # dopamine.cut(350, 1800, WN=True)
    # dopamine.smooth(n=5)
    dopamine.smooth(n=3)
    dopamine.normalizeIntegration()
    GABA.removeThermalNoise(dn)
    # GABA.ORPL(min_bubble_widths=100)
    # GABA.cut(350, 1800, WN=True)
    # GABA.smooth(n=5)
    GABA.cut(2810, 3024, WN=True)
    GABA.smooth(n=3)
    GABA.normalizeIntegration()
    glut.removeThermalNoise(dn)
    # glut.ORPL(min_bubble_widths=100)
    # glut.cut(350, 1800, WN=True)
    # glut.smooth(n=5)
    glut.cut(2810, 3024, WN=True)
    glut.smooth(n=3)
    # glut.factor(10)
    glut.normalizeIntegration()

    # dude = dopamine.addSpectra(GABA)
    # dude.add(glut)
    # dude.display()

    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/Walter/all_hanu_data/'):
        if dir[0] == '.':
            continue
        data.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/maitrise/DATA/Walter/all_hanu_data/' + dir + '/').spectra())
    data = spectrum.Spectra(data)
    data.removeThermalNoise(bg)
    data.smooth(n=5)

    # data.butterworthFilter(cutoff_frequency=5, order=4)
    data.ORPL(min_bubble_widths=20)
    data.cut(350, 1800, WN=True)
    # data.shortenLabels()
    # data.displayMeanSTD()
    # data.umap(n_components=2)
    # data.UMAP_KNNIndividualLabel()
    data.pca(nbOfComp=3)
    data.PCA_KNNIndividualLabel()
    # data.tsne()
    # data.pca(nbOfComp=20)
    # data.pcaDisplay(1, 2)
    # data.pcaDisplay(3, 4)
    # data.pcaDisplay(5, 6)
    # data.pcaDisplay(7, 8)
    # data.pcaDisplay(15, 20)
    # data.pcaScatterPlot(1, 2)
    # data.pcaScatterPlot(3, 4)
    # data.pcaScatterPlot(5, 6)
    # data.pcaScatterPlot(7, 8)
    # data.pcaScatterPlot(9, 10)
    # data.pcaScatterPlot(11, 12)
    # data.pcaScatterPlot(13, 14)
    # data.pcaScatterPlot(15, 16)
    # data.pcaScatterPlot(17, 18)


    # data.PCA_KNNIndividualLabel(n_comp=10)


    # data.ORPL(min_bubble_widths=40)
    # data.cut(400, 1800, WN=True)
    # data.CRRemoval()
    # data.smooth(n=5)
    # data.shortenLabels()
    # data.displayMeanSTD()

    # data.picRatio(l1=869, l2=488)
    # data.picRatio(l1=688, l2=740)
    # data.picRatio(l1=984, l2=488)
    # data.picRatio(l1=1087, l2=1044)
    # data.picRatio(l1=1087, l2=488)
    # data.picRatio(l1=1288, l2=1274)
    # data.picRatio(l1=1283, l2=488)
    # data.picRatio(l1=1438, l2=488)
    # data.picRatio()
    # data.PRScatterPlot(1, 2)
    # data.PRScatterPlot(3, 4)
    # data.PRScatterPlot(5, 6)
    # data.PRScatterPlot(7, 8)

    # data.PR_KNNIndividualLabel()


    # data.pca()
    # data.PCA_KNNIndividualLabel()

    # data.PCAExternalComposit(composit=[dopamine.counts, GABA.counts, glut.counts], labels=['Dopamine', 'GABA', 'glut'])
    # data.pcaecScatterPlot('Dopamine', shorten_labels=True)
    # data.pcaecScatterPlot('glut', shorten_labels=True)
    # data.pcaecScatterPlot('GABA', shorten_labels=True)
    # data.PCAEC_KNNIndividualLabel()

    # data.shortenLabels()
    # data.displayMeanSTD()
    # data.pca()
    # data.prob_classifier()
    # data.R2_classifier()

    # data.pcaDisplay(1, 2)
    # data.pcaDisplay(3, 4)
    # data.pcaDisplay(5, 6)
    #
    # data.pcaScatterPlot(1, 2)
    # data.pcaScatterPlot(3, 4)
    # data.pcaScatterPlot(5, 6)

def neuro_con():
    import spectrum

    bg = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20221124/darkPM/').spectraSum()
    GABA = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/neuro_con/GABA60s/GABA500/').spectraSum()
    glut = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/neuro_con/glut60s/glut50/').spectraSum()
    dopamine = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/neuro_con/dopamine60s/dopamine500/').spectraSum()
    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/neuro_con/dopamine60s/'):
        if dir[0] == '.':
            continue
        data.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/maitrise/DATA/neuro_con/dopamine60s/' + dir + '/').spectra())
    data = spectrum.Spectra(data)
    data.removeThermalNoise(bg)

    GABA.removeThermalNoise(bg)
    glut.removeThermalNoise(bg)
    dopamine.removeThermalNoise(bg)

    GABA.cut(2800, 3020, WN=True)
    glut.cut(2800, 3020, WN=True)
    dopamine.cut(2800, 3020, WN=True)

    array_2937 = []
    concentrations = []
    array_2800 = []
    array_2870 = []
    array_2976 = []
    array_3007 = []

    for spectrum in data.spectra:
        concentrations.append(float(spectrum.label[8:]))
    #     array_2800.append(spectrum.counts[900] * 4.8)
    #     array_2870.append(spectrum.counts[965] * 1.53)
    #     array_2976.append(spectrum.counts[1016] * 0.9)
    #     array_3007.append(spectrum.counts[1031] * 0.82)
    #     array_2937.append(spectrum.counts[996])
    #
    # plt.plot(concentrations, array_2800, 'o', label='2800 cm-1')
    # plt.plot(concentrations, array_2937, 'o', label='2937 cm-1')
    # plt.plot(concentrations, array_2870, 'o', label='2870 cm-1')
    # plt.plot(concentrations, array_2976, 'o', label='2976 cm-1')
    # plt.plot(concentrations, array_3007, 'o', label='3007 cm-1')
    # plt.legend()
    # plt.show()
    dopamine.smooth(n=3)
    dopamine.display()


    data.smooth(n=3)
    data.cut(2810, 3024, WN=True)
    data.pca()
    data._getPCAdf()
    for i in range(3):
        plt.plot(concentrations, data.pca_df.to_numpy().T[i], 'o', label='PC {0}'.format(i + 1))
    plt.legend()
    plt.show()

    data.pcaDisplay(1, 2, 3)







    # plt.plot(dopamine.counts)
    # plt.show()


    # GABA.normalizeIntegration()
    # glut.normalizeIntegration()
    # dopamine.normalizeIntegration()
    # GABA.ORPL(min_bubble_widths=200, display=False)
    # glut.ORPL(min_bubble_widths=200, display=False)
    # dopamine.ORPL(min_bubble_widths=200, display=False)

    # neuro = GABA.addSpectra(glut)
    # neuro.add(dopamine)
    # neuro.display()
    # GABA.display()
    # glut.display()
    # dopamine.display()

def PCA_dan():
    def test40WithSubclassExampleGraphsAndData(self):
        # Final example with the subclass for cleaner code

        basisSet_bj = self.createBasisSet(self.X, N=5, maxPeaks=5, maxAmplitude=1, maxWidth=30, minWidth=5)
        plt.plot(basisSet_bj.T)
        plt.legend()
        plt.show()
        dataSet_ij, concentration_ik = self.createDatasetFromBasisSet(100, basisSet_bj)
        # dataSet_ij is now a simulated dataset of 100 spectra coming from 5 analytes mixed in various concentrations
        # basisSet_bj is their individual spectra

        pca = LabPCA(n_components=5)
        pca.fit(dataSet_ij)

        # Look at non-centered components
        plt.plot(pca.components_noncentered_.T)
        plt.title("Principal components (non-centered)")
        plt.show()

        # To avoid confusion, indices (i,b,j,k,p) represent:
        # i = sample #
        # b = basis #
        # j = feature #
        # k = concentration #
        # p = principal coefficient #
        b_bp = pca.transform_noncentered(np.array([basisSet_bj[0]]))
        s_ip = pca.transform_noncentered(dataSet_ij)
        s_pi = s_ip.T
        invb_pb = np.linalg.pinv(b_bp)
        invb_bp = invb_pb.T

        recoveredConcentrations_ki = (invb_bp@s_pi).T
        expectedConcentrations_ki = concentration_ik.T
        print("Expected concentrations (first four only):\n", expectedConcentrations_ki[0:3])
        print("Recovered concentrations (first four only):\n", recoveredConcentrations_ki[0:3])

        everythingBelowThreshold = ((expectedConcentrations_ki-recoveredConcentrations_ki) ).all() < 1e-2
        self.assertTrue(everythingBelowThreshold )
        print("Minimal differences: ", everythingBelowThreshold)



def testeroni():
    bg = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20221124/darkPM/').spectraSum()
    # bg.display(WN=False)
    # plt.plot(bg.counts)
    # plt.show()


    caudate_WM = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20230428/caudate_WM_l/').spectra()
    caudate_WM.removeThermalNoise(bg)
    caudate_WM.cut(None, -4)
    caudate_WM.ORPL(min_bubble_widths=150, display=False)
    # caudate_WM.display()
    putamen_WM = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20230428/Putamen_WM_l/').spectra()
    putamen_WM.removeThermalNoise(bg)
    putamen_WM.cut(None, -4)
    putamen_WM.ORPL(min_bubble_widths=150, display=False)
    STN_SN = spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20230428/STN_SN_l/').spectra()
    STN_SN.removeThermalNoise(bg)
    # STN_SN.smooth(n=3)
    STN_SN.cut(None, -4)
    # STN_SN.ORPL(min_bubble_widths=40, display=False)

    STN_SN.display()



    caudate_WM.tile(x=6, y=6, WN_to_display=2848)
    putamen_WM.tile(x=6, y=6, WN_to_display=2848)
    STN_SN.tile(x=16, y=6, WN_to_display=1434)
    STN_SN.tile(x=16, y=6, WN_to_display=1360)
    STN_SN.tile(x=16, y=6, WN_to_display=2847)
    STN_SN.tile(x=16, y=6, WN_to_display=2880)
    STN_SN.tile(x=16, y=6, WN_to_display=2933)

testeroni()



# neuro_con()
# hanu()
# test()
# PCA_dan()
# testeroni()




import os
import json


def txt_to_json(folder_path, output_file):

    # Iterate over files in the folder
    n = 0
    for filename in os.listdir(folder_path):
        n += 1
        current_y_values = []
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            # Read the contents of the text file
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Remove metadata (assuming it is present in the first few lines)
            lines = lines[14:]  # Adjust this line based on the number of metadata lines

            # Extract y values from each line and append to the y_values list
            n_lines = 0
            background = []
            for line in lines:
                n_lines += 1
                background.append(1)

                values = line.strip().split()
                if len(values) >= 2:
                    y = float(values[1])
                    current_y_values.append(round(y))

            # Create the dictionary
            thisdict = {
                "TimeStamp": "2023/6/7-11h10 52.329",
                "Comment": "Dinde",
                "Wavelength": 785,
                "Power": 50,
                "ExpTime": 30,
                "SpecPerAcqui": 1,
                "CamTemp": 0,
                "CamStab": "STABILIZED",
                "Background": background,
                "RawSpectra": current_y_values
            }

            # Save y_values as JSON
            with open('/Users/antoinerousseau/Downloads/test/test{0}.json'.format(n), 'w') as json_file:
                json.dump(thisdict, json_file)



# Usage example
# folder_path = '/Users/antoinerousseau/Desktop/maitrise/DATA/20230428/Thalamus6_1'  # Replace with the actual folder path
# output_file = '/Users/antoinerousseau/Downloads/test/test.json'  # Replace with the desired output file path

# txt_to_json(folder_path, output_file)



