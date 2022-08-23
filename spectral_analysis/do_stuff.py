import numpy as np
from dcclab.database import *
import spectrum
import matplotlib.pyplot as plt
import os


# A NEW ERA HAS BEGUN!!!!!!!!!!!!!!!!!!!!!!!

bg = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220223/backgrounds/10min_0light/').spectraSum()
nbg = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220802/bg/').spectraSum()


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

# iso100full = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220413/100ms_iso/').spectra()
# iso300full = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220413/300ms_iso/').spectra()
# iso100half = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220413/100ms_iso_halfpower/').spectra()

# iso300full.removeThermalNoise(bg)
# iso100full.cut(100, None)
# iso300full.getSTD()


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
iso_new = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220627/').spectra()
# iso_new.display()
# iso_new.normalizeIntegration()

iso_good = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220413/100ms_iso/').spectra()
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
iso_bad = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220422/iso/').spectra()
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
db = SpectraDB()
DataSha_RGPI = db.getSpectralDataFrame(datasetId="SHAVASANA-001", id1="DRS", id2="RGPI")
DataSha_RGPI_x = np.array(DataSha_RGPI.index)
DataSha_RGPI = DataSha_RGPI.to_numpy().T

DataSha_RSTN = db.getSpectralDataFrame(datasetId="SHAVASANA-001", id1="DRS", id2="RSTN")
DataSha_RSTN_x = np.array(DataSha_RSTN.index)
DataSha_RSTN = DataSha_RSTN.to_numpy().T

DataSha_ROFF = db.getSpectralDataFrame(datasetId="SHAVASANA-001", id1="DRS", id2="ROFF")
DataSha_ROFF_x = np.array(DataSha_ROFF.index)
DataSha_ROFF = DataSha_ROFF.to_numpy().T

#get labels
DRS_RGPI_label =
DRS_RSTN_label =
DRS_ROFF_label =



DRS_RGPI = spectrum.ArrayToSpectra(DataSha_RGPI_x, DataSha_RGPI, label='DRS_RGPI').asSpectra()
DRS_RSTN = spectrum.ArrayToSpectra(DataSha_RSTN_x, DataSha_RSTN, label='DRS_RSTN').asSpectra()
DRS_ROFF = spectrum.ArrayToSpectra(DataSha_ROFF_x, DataSha_ROFF, label='DRS_ROFF').asSpectra()

DRS_RGPI.add(DRS_ROFF, DRS_RSTN)
DRS_RGPI.cut(450, 587, WL=True)
DRS_RGPI.display3Colored('DRS_RGPI', 'DRS_RSTN', 'DRS_ROFF', WN=False, display_label=False)
DRS_RGPI.pca()
DRS_RGPI.pcaScatterPlot(1, 2)
DRS_RGPI.pcaScatterPlot(3, 4)
DRS_RGPI.pcaScatterPlot(5, 6)
DRS_RGPI.pcaScatterPlot(7, 8)
DRS_RGPI.pcaScatterPlot(9, 10)
DRS_RGPI.pcaDisplay(1, 2, 3)
DRS_RGPI.pcaDisplay(4, 5, 6)
DRS_RGPI.pcaDisplay(7, 8, 9)
