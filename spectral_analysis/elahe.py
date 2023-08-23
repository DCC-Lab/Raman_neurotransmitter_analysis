import spectrum
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
from dcclab.database import *
import random
import csv


def elahe():
    bg = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20230317/thermal/').spectraSum()
    iso = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20230317/iso/').spectra()
    methanol = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20230317/methanol/').spectra()
    ethanol = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20230317/ethanol/').spectra()

    # iso.removeThermalNoise(bg)
    # methanol.removeThermalNoise(bg)
    # ethanol.removeThermalNoise(bg)
    iso.add(methanol, ethanol)
    iso.cut(2700, 3000, WN=True)
    # methanol.cut(2700, 3000, WN=True)
    # ethanol.cut(2700, 3000, WN=True)
    iso.displayMeanSTD()
    # methanol.display()
    # ethanol.display()




elahe()