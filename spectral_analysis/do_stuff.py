import numpy as np

import spectrum
import matplotlib.pyplot as plt
import os


# A NEW ERA HAS BEGUN!!!!!!!!!!!!!!!!!!!!!!!

bg = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220223/backgrounds/10min_0light/').spectraSum()

# watersum = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220223/eau_30sec/').spectraSum()
# watersum.removeThermalNoise(bg)
# watersum.normalizeCounts()
# watersum.polyfit(6)

# water = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220223/eau_30sec/').spectra()
# water.removeThermalNoise(bg)
# water.normalizeCounts()
# water.display()

# GABA = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220315/GABA/').spectra()
# GABA.removeThermalNoise(bg)
# GABA.polyfit(6)
# GABA.subtract(watersum)
# GABAsum.display()
# GABAsum.normalizeCounts()
# GABAsum.subtract(water)
# GABAsum.fftFilter()
# GABAsum.display()




# dopamine = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220315/dopamine/').spectra()
# dopamine.removeThermalNoise(bg)
# dopamine.polyfit(6)
# dopamine.subtract(watersum)
# dopamine.subtract(water)
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

# iso100full.removeThermalNoise(bg)
# iso300full.getSTD()


# Erika data
# path = '/Users/antoinerousseau/Desktop/PM/'
#
# data = []
# for dir in os.listdir(path):
#     if dir[0] == '.':
#         continue
#     data.append(spectrum.Acquisition(path + dir + '/').spectra())
#
# data = spectrum.Spectra(data)
# data.removeThermalNoise(bg)
# data.pca()
# data.pcaDisplay()
# data.pcaScatterPlot(PCx=1, PCy=2)


# GABA = spectrum.Acquisition('/Users/antoinerousseau/Desktop/PM/cm_30sec_lightoff_1A_2/').spectraSum()
# water = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220420/water/').spectraSum()
#
# x = GABA.counts
# GABA.removeThermalNoise(bg)
# GABA.normalizeIntegration()
# water.removeThermalNoise(bg)
# water.normalizeCounts()

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

D4 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220421/D4/').spectraSum()
D3 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220421/D3/').spectraSum()
D2 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220421/D2/').spectraSum()
D5 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220421/D5/').spectraSum()
D1 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220421/D1/').spectraSum()
D6 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220421/D6/').spectraSum()

bg1 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220421/background/').spectra()

iso1 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220413/100ms_iso/').spectra()
iso2 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220421/iso/').spectraSum()
iso3 = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220422/iso/').spectraSum()

iso2.fixSpec()

iso2.setZero(20)

plt.plot(iso1.spectra[0].wavenumbers, iso1.spectra[0].counts, label='old')
plt.plot(iso2.wavenumbers, iso2.counts, label='new')
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
plt.legend()
plt.show()