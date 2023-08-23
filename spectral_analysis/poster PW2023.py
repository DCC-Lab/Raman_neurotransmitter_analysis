import spectrum
import os

def monkeyGreyWhite():
    bg = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20220929/morning_verif/darknoise/').spectraSum()
    pbs = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20220802/pbs1/').spectraSum()
    water = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20220913/water/').spectraSum()
    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/20220802/Monkey_brain/'):
        if dir[0] == '.':
            continue
        data.append(
            spectrum.Acquisition('/Users/antoinerousseau/Desktop/maitrise/DATA/20220802/Monkey_brain/' + dir + '/').spectra())
    data = spectrum.Spectra(data)
    pbs.removeThermalNoise(bg)
    water.removeThermalNoise(bg)
    data.removeThermalNoise(bg)
    # data.normalizeIntegration()
    data.butterworthFilter(6)
    # data.subtract(water)
    data.cut(400, 3025, WN=True)
    # data.normalizeIntegration()

    # data.displayMeanSTD(WN=True)
    data.pca()
    data.pcaScatterPlot(1, 2)

monkeyGreyWhite()