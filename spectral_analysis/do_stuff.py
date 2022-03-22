import spectrum
import matplotlib.pyplot as plt

# A NEW ERA HAS BEGUN!!!!!!!!!!!!!!!!!!!!!!!

bg = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220223/backgrounds/10min_0light/').spectraSum()

water = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220223/eau_30sec/').spectraSum()
water.removeThermalNoise(bg)
# water.normalizeCounts()
# water.display()

dopamine = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220315/dopamine/').spectra()
dopamine.removeThermalNoise(bg)
dopamine.subtract(water)
dopamine.normalizeCounts()
dopamine.display()

GABA = spectrum.Acquisition('/Users/antoinerousseau/Desktop/20220315/GABA/')
GABAsum = GABA.spectra()
# GABAsum.removeThermalNoise(bg)
# GABAsum.display()
# GABAsum.normalizeCounts()
GABAsum.subtract(water)
# GABAsum.display()


# plt.plot(dopamine.wavenumber, dopamine.counts, label='GABA')
# plt.plot(dopamine.wavenumber, y, label='GABA - water')
# plt.legend()
# plt.show()





