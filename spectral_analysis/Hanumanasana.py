import spectrum
import numpy as np
import matplotlib.pyplot as plt
import os



def verify_WR():
    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/Hanumanasana/'):
        if dir[0] == '.':
            continue
        data.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/Hanumanasana/' + dir + '/').spectra())
    data = spectrum.Spectra(data)

    data.displayMeanSTD()


verify_WR()