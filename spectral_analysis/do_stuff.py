import spectrum
import matplotlib.pyplot as plt

# A NEW ERA HAS BEGUN!!!!!!!!!!!!!!!!!!!!!!!

bg = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220223/backgrounds/10min_0light/').spectraSum()
watersum = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220223/eau_30sec/').spectraSum()
watersum.removeThermalNoise(bg)

water = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220223/eau_30sec/').spectra()
water.removeThermalNoise(bg)
# water.normalizeCounts()
# water.display()

GABA = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220315/GABA/').spectra()
GABA.removeThermalNoise(bg)
# GABA.subtract(watersum)
# GABAsum.display()
# GABAsum.normalizeCounts()
# GABAsum.subtract(water)
# GABAsum.fftFilter()
# GABAsum.display()




dopamine = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220315/dopamine/').spectra()
dopamine.removeThermalNoise(bg)
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

new_spectra = dopamine.addSpectra(GABA, water)
# new_spectra.display()
new_spectra.pca()
new_spectra.pcaScatterPlot(PCx=1, PCy=2)


