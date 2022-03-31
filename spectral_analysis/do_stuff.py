import spectrum
import matplotlib.pyplot as plt

# A NEW ERA HAS BEGUN!!!!!!!!!!!!!!!!!!!!!!!

bg = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220223/backgrounds/10min_0light/').spectraSum()

water = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220223/eau_30sec/').spectraSum()
water.removeThermalNoise(bg)
# water.normalizeCounts()
# water.display()

GABA = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220315/GABA/').spectra()
GABA.removeThermalNoise(bg)
# GABAsum.display()
# GABAsum.normalizeCounts()
# GABAsum.subtract(water)
# GABAsum.fftFilter()
# GABAsum.display()




dopamine = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220315/dopamine/').spectraSum()
dopamine.removeThermalNoise(bg)
dopamine.subtract(water)
# dopamine.display()

# plt.plot(dopamine.wavenumbers, y1, label='dopamine raw')
# plt.plot(dopamine.wavenumbers, y2, label='dopamine not noisy')
# plt.plot(dopamine.wavenumbers, dopamine.counts, label='dopamine not noisy, no background')
# plt.xlabel('Wavenumber [cm-1]')
# plt.ylabel('Counts [-]')
# plt.legend()
# plt.show()

new_spectra = dopamine.addSpectra(GABA)
new_spectra.display()




