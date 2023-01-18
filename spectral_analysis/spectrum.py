import os
import glob
import fnmatch
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
from plotly.offline import plot
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from umap import UMAP
from sklearn.manifold import TSNE
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from scipy import signal
from sklearn.metrics import accuracy_score
import itertools
import random


class QEproSpectralFile:

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.x = []
        self.y = []
        self.date = None
        self.trig_mode = None
        self.integration_time = None
        self.dark_corr = None
        self.nonlin_corr = None
        self.x_axis_unit = None
        self.pixel_nb = None
        self.fileName = filepath.split('/')[-1]
        self.spectrum = None
        self.README = None

        self._loadSpectrumValues()
        self.isValid()
        self._loadAcquisitionDetails()

    def isValid(self, pixelNb=1044):
        # Is it a .txt file
        fileType = self.filepath.split('/')[-1][-4:]
        assert fileType == '.txt', 'file type should be .txt'

        # Does it contain expected strings at the right place
        fich = open(self.filepath, "r")
        fich_list = list(fich)
        date = fich_list[2]
        trig_mode = fich_list[5]
        integration_time = fich_list[6]
        dark_corr = fich_list[8]
        nonlin_corr = fich_list[9]
        x_axis_unit = fich_list[11]
        pixel_nb = fich_list[12]
        fich.close()

        assert date.split(': ')[
                   0] == 'Date', "Was expecting the .txt file\'s 3rd lane to be the date of the acquisition"
        assert trig_mode.split(': ')[
                   0] == 'Trigger mode', "Was expecting the .txt file\'s 6th lane to be the trigger mode used for the acquisition"
        assert integration_time.split(': ')[
                   0] == 'Integration Time (sec)', "Was expecting the .txt file\'s 7th lane to be the integration time used (in sec) for the acquisition"
        assert dark_corr.split(': ')[
                   0] == 'Electric dark correction enabled', "Was expecting the .txt file\'s 9th lane to be the electric dark correction state for the acquisition"
        assert nonlin_corr.split(': ')[
                   0] == 'Nonlinearity correction enabled', "Was expecting the .txt file\'s 10th lane to be the nonlinearity correction state for the acquisition"
        assert x_axis_unit.split(': ')[
                   0] == 'XAxis mode', "Was expecting the .txt file\'s 12th lane to be the x axis units for the acquisition"
        assert pixel_nb.split(': ')[
                   0] == 'Number of Pixels in Spectrum', "Was expecting the .txt file\'s 13th lane to be the number of pixels used for the acquisition"

        # Does it have the right len(x), len(y)
        assert len(self.x) == pixelNb, "Was expecting {0} x values, {1} were given".format(str(pixelNb),
                                                                                           str(len(self.x)))
        assert len(self.y) == pixelNb, "Was expecting {0} y values, {1} were given".format(str(pixelNb),
                                                                                           str(len(self.y)))
        assert len(self.x) == len(self.y), "x and y values for a spectrum should have the same amount of elements"

        # print('File is valid')

    def _loadSpectrumValues(self):
        fich = open(self.filepath, "r")
        test_str = list(fich)[14:]
        fich.close()

        # Nettoyer les informations
        spectral_data = []
        for j in test_str:
            elem_str = j.replace(",", ".").replace("\n", "").replace("\t", ",")
            elem = elem_str.split(",")
            self.x.append(float(elem[0]))
            self.y.append(float(elem[1]))
            spectral_data.append([float(elem[0]), float(elem[1])])
        self.spectrum = np.transpose(spectral_data)

    def _loadAcquisitionDetails(self):
        fich = open(self.filepath, "r")
        fich_list = list(fich)
        date = fich_list[2]
        trig_mode = fich_list[5]
        integration_time = fich_list[6]
        dark_corr = fich_list[8]
        nonlin_corr = fich_list[9]
        x_axis_unit = fich_list[11]
        pixel_nb = fich_list[12]
        fich.close()

        year = (date.split(' ')[-1]).replace('\n', '')
        day = date.split(' ')[-4]
        if len(day) == 1:
            day = '0' + day
        month = date.split(' ')[-5]
        if month == 'Jan':
            month = '01'
        if month == 'Feb':
            month = '02'
        if month == 'Mar':
            month = '03'
        if month == 'Apr':
            month = '04'
        if month == 'May':
            month = '05'
        if month == 'Jun':
            month = '06'
        if month == 'Jul':
            month = '07'
        if month == 'Aug':
            month = '08'
        if month == 'Sep':
            month = '09'
        if month == 'Oct':
            month = '10'
        if month == 'Nov':
            month = '11'
        if month == 'Dec':
            month = '12'

        integration_time = integration_time.split(': ')[-1]
        integration_time = integration_time.replace('\t', '')
        multiplier = int(integration_time.split('E')[-1])
        integration_time = (float(integration_time.split('E')[0])) * (10 ** multiplier)

        dark_corr = dark_corr.split(' ')[-1]
        if dark_corr == 'true':
            dark_corr = True
        if dark_corr == 'false':
            dark_corr = False

        nonlin_corr = nonlin_corr.split(' ')[-1]
        if nonlin_corr == 'true':
            nonlin_corr = True
        if nonlin_corr == 'false':
            nonlin_corr = False

        self.date = year + month + day
        self.trig_mode = int(trig_mode.split(' ')[-1])
        self.integration_time = integration_time
        self.dark_corr = dark_corr.replace('\n', '')
        self.nonlin_corr = nonlin_corr.replace('\n', '')
        self.x_axis_unit = x_axis_unit.split(': ')[-1].replace('\n', '')
        self.pixel_nb = int(pixel_nb.split(': ')[-1])


class USB2000SpectralFile:

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.x = []
        self.y = []
        self.date = None
        self.trig_mode = None
        self.integration_time = None
        self.dark_corr = None
        self.x_axis_unit = None
        self.pixel_nb = None
        self.fileName = filepath.split('/')[-1]
        self.spectrum = None
        self.README = None

        self._loadSpectrumValues()
        self.isValid()
        self._loadAcquisitionDetails()

    def isValid(self, pixelNb=2048):
        # Is it a .txt file
        fileType = self.filepath.split('/')[-1][-4:]
        assert fileType == '.txt', 'file type should be .txt'

        # Does it contain expected strings at the right place
        fich = open(self.filepath, "r")
        fich_list = list(fich)
        date = fich_list[2]
        trig_mode = fich_list[6]
        integration_time = fich_list[7]
        dark_corr = fich_list[8]
        x_axis_unit = fich_list[10]
        pixel_nb = fich_list[11]
        fich.close()

        assert date.split(': ')[
                   0] == 'Date', "Was expecting the .txt file\'s 3rd lane to be the date of the acquisition"
        assert trig_mode.split(': ')[
                   0] == 'Trigger mode', "Was expecting the .txt file\'s 6th lane to be the trigger mode used for the acquisition"
        assert integration_time.split(': ')[
                   0] == 'Integration Time (sec)', "Was expecting the .txt file\'s 7th lane to be the integration time used (in sec) for the acquisition"
        assert dark_corr.split(': ')[
                   0] == 'Electric dark correction enabled', "Was expecting the .txt file\'s 9th lane to be the electric dark correction state for the acquisition"
        assert x_axis_unit.split(': ')[
                   0] == 'XAxis mode', "Was expecting the .txt file\'s 12th lane to be the x axis units for the acquisition"
        assert pixel_nb.split(': ')[
                   0] == 'Number of Pixels in Spectrum', "Was expecting the .txt file\'s 13th lane to be the number of pixels used for the acquisition"

        # Does it have the right len(x), len(y)
        assert len(self.x) == pixelNb, "Was expecting {0} x values, {1} were given".format(str(pixelNb),
                                                                                           str(len(self.x)))
        assert len(self.y) == pixelNb, "Was expecting {0} y values, {1} were given".format(str(pixelNb),
                                                                                           str(len(self.y)))
        assert len(self.x) == len(self.y), "x and y values for a spectrum should have the same amount of elements"

        # print('File is valid')

    def _loadSpectrumValues(self):
        fich = open(self.filepath, "r")
        test_str = list(fich)[13:]
        fich.close()

        # Nettoyer les informations
        spectral_data = []
        for j in test_str:
            elem_str = j.replace(",", ".").replace("\n", "").replace("\t", ",")
            elem = elem_str.split(",")
            self.x.append(float(elem[0]))
            self.y.append(float(elem[1]))
            spectral_data.append([float(elem[0]), float(elem[1])])
        self.spectrum = np.transpose(spectral_data)

    def _loadAcquisitionDetails(self):
        fich = open(self.filepath, "r")
        fich_list = list(fich)
        date = fich_list[2]
        trig_mode = fich_list[6]
        integration_time = fich_list[7]
        dark_corr = fich_list[8]
        x_axis_unit = fich_list[10]
        pixel_nb = fich_list[11]
        fich.close()

        year = (date.split(' ')[-1]).replace('\n', '')
        day = date.split(' ')[-4]
        if len(day) == 1:
            day = '0' + day
        month = date.split(' ')[-5]
        if month == 'Jan':
            month = '01'
        if month == 'Feb':
            month = '02'
        if month == 'Mar':
            month = '03'
        if month == 'Apr':
            month = '04'
        if month == 'May':
            month = '05'
        if month == 'Jun':
            month = '06'
        if month == 'Jul':
            month = '07'
        if month == 'Aug':
            month = '08'
        if month == 'Sep':
            month = '09'
        if month == 'Oct':
            month = '10'
        if month == 'Nov':
            month = '11'
        if month == 'Dec':
            month = '12'

        integration_time = integration_time.split(': ')[-1]
        integration_time = integration_time.replace('\t', '')
        multiplier = int(integration_time.split('E')[-1])
        integration_time = (float(integration_time.split('E')[0])) * (10 ** multiplier)

        dark_corr = dark_corr.split(' ')[-1]
        if dark_corr == 'true':
            dark_corr = True
        if dark_corr == 'false':
            dark_corr = False

        self.date = year + month + day
        self.trig_mode = int(trig_mode.split(' ')[-1])
        self.integration_time = integration_time
        self.dark_corr = dark_corr.replace('\n', '')
        self.x_axis_unit = x_axis_unit.split(': ')[-1].replace('\n', '')
        self.pixel_nb = int(pixel_nb.split(': ')[-1])


class VictoriaFiles:

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.x = []
        self.y = []
        self.fileName = filepath.split('/')[-1]
        self.spectrum = None
        self.README = None
        self.integration_time = 1
        self.label = 'victo'

        self._loadSpectrumValues()
        # self.isValid()
        # self._loadAcquisitionDetails()

    def _loadSpectrumValues(self):
        fich = open(self.filepath, "r", errors='ignore')
        test_str = list(fich)[3:123]

        # Nettoyer les informations
        spectral_data = []
        for j in test_str:
            elem_str = j.replace("\n", "").replace("\t", ",")
            elem = elem_str.split(",")
            self.x.append(float(elem[41]))
            self.y.append(float(elem[58]))
            spectral_data.append([float(elem[41]), float(elem[58])])
        self.spectrum = np.transpose(spectral_data)

        fich.close()

        # Nettoyer les informations
        # spectral_data = []
        # for j in test_str:
        #     elem_str = j.replace(",", ".").replace("\n", "").replace("\t", ",")
        #     elem = elem_str.split(",")
        #     self.x.append(float(elem[0]))
        #     self.y.append(float(elem[1]))
        #     spectral_data.append([float(elem[0]), float(elem[1])])
        # self.spectrum = np.transpose(spectral_data)


class DeapoliLiveMonkeySpectralFile:

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.x = []
        self.y = []
        self.fileName = filepath.split('/')[-1]
        self.spectrum = None
        self.integration_time = 1

        self._loadSpectrumValues()
        self.isValid()

    def isValid(self, pixelNb=2048):
        # Is it a .txt file
        fileType = self.filepath.split('/')[-1][-4:]
        assert fileType == '.txt', 'file type should be .txt'

        # Does it have the right len(x), len(y)
        assert len(self.x) == pixelNb, "Was expecting {0} x values, {1} were given".format(str(pixelNb),
                                                                                           str(len(self.x)))
        assert len(self.y) == pixelNb, "Was expecting {0} y values, {1} were given".format(str(pixelNb),
                                                                                           str(len(self.y)))
        assert len(self.x) == len(self.y), "x and y values for a spectrum should have the same amount of elements"

        # print('File is valid')

    def _loadSpectrumValues(self):
        fich = open(self.filepath, "r")
        test_str = list(fich)
        fich.close()

        # Nettoyer les informations
        spectral_data = []
        x = []
        y = []
        for j in test_str:
            elem_str = j.replace(",", ".").replace("\n", "").replace("\t", ",")
            elem = elem_str.split(",")
            x.append(elem[0])
            y.append(elem[1])
        for i in x:
            val, exp = i.split('e')
            new_x = float(val) * 10 ** float(exp)
            self.x.append(new_x)
        for i in y:
            val, exp = i.split('e')
            new_y = float(val) * 10 ** float(exp)
            self.y.append(new_y)
        self.spectrum = [self.x, self.y]


class Acquisition:
    def __init__(self, directory, fileType='OVSF', extension='.txt'):
        self.directory = directory
        self.spectralFiles = []
        self.directoryName = directory.split('/')[-2]
        self.extension = extension

        filePaths = self._listNameOfFiles()

        if fileType == 'OVSF':
            for filepath in filePaths:
                spectralFile = QEproSpectralFile(directory + filepath)
                self.spectralFiles.append(spectralFile)
        if fileType == 'VF':
            for filepath in filePaths:
                spectralFile = VictoriaFiles(directory + filepath)
                self.spectralFiles.append(spectralFile)
        if fileType == 'USB2000':
            for filepath in filePaths:
                spectralFile = USB2000SpectralFile(directory + filepath)
                self.spectralFiles.append(spectralFile)
        if fileType == 'Depaoli':
            for filepath in filePaths:
                spectralFile = DeapoliLiveMonkeySpectralFile(directory + filepath)
                self.spectralFiles.append(spectralFile)

    def _listNameOfFiles(self) -> list:
        foundFiles = []
        for file in os.listdir(self.directory):
            if file[0] == '.':
                continue
            if file == 'README.txt':
                # should do stuff like reading it and taking info from it
                RM = open(self.directory + file, "r")
                self.README = list(RM)
                continue
            if fnmatch.fnmatch(file, f'*{self.extension}'):
                foundFiles.append(file)

        foundFiles = sorted(foundFiles)
        foundFiles = sorted(foundFiles, key=len)
        return foundFiles

    def spectra(self):
        spectra = []
        for file in self.spectralFiles:
            spectrum = Spectrum(file.x, file.y, file.integration_time, self.directoryName)
            spectra.append(spectrum)

        return Spectra(spectra)

    def spectraSum(self):
        count_sum = np.zeros(len(self.spectralFiles[0].x))
        integrationTime = 0
        for i in self.spectralFiles:
            count_sum += np.array(i.y)
            integrationTime += i.integration_time

        return Spectrum(self.spectralFiles[0].x, count_sum, integrationTime, self.directoryName)

    def _sorted_alphanumeric(self, data):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(data, key=alphanum_key)


class ArrayToSpectra:
    def __init__(self, x, array, label=None, integrationTime=None):
        self.x = x
        self.data = array
        self.labelList = label
        self.integrationTimeList = integrationTime
        self.spectra = []

        if type(self.labelList) == type(None):
            self.labelList = []
            for i in range(len(array)):
                self.labelList.append('No Label Given')

        if type(self.labelList) == str:
            self.labelList = []
            for i in range(len(array)):
                self.labelList.append(label)

        if self.integrationTimeList == None:
            self.integrationTimeList = []
            for i in range(len(array)):
                self.integrationTimeList.append(1)

        assert len(array) == len(self.labelList) == len(
            self.integrationTimeList), 'Arguments given must all be the same length'
        assert len(x) == len(array[0]), 'x must be an array containing as many element as each array element'

        self._prepareSpectraList()

    def _prepareSpectraList(self):
        self.spectra = []
        for i in range(len(self.data)):
            spectrum = Spectrum(self.x, self.data[i], self.integrationTimeList[i], self.labelList[i])
            self.spectra.append(spectrum)

    def asSpectra(self):
        self._prepareSpectraList()
        spectra = []
        for spectrum in self.spectra:
            spectra.append(spectrum)
        return Spectra(spectra)

    def cleanLabel(self, keyPos):
        # keyPos is the position of the element of interest in original label
        newLabels = []
        for label in self.labelList:
            key = ''
            for i in keyPos:
                key += label.split('-')[i]
            newLabels.append(key)
        self.labelList = newLabels


class Spectrum:
    def __init__(self, wavelenghts, counts, integrationTime, label, annotation=None):
        self.wavelenghts = wavelenghts

        self.counts = counts
        self.integrationTime = integrationTime
        self.label = label
        self.annotation = annotation

        exist = 0 in self.wavelenghts
        if exist == False:
            self.wavenumbers = ((10 ** 7) * ((1 / 785) - (1 / np.array(self.wavelenghts))))
        if exist == True:
            self.wavenumbers = None

    def getSNR(self, bgStart=550, bgEnd=800):
        if len(self.counts) <= bgStart:
            return None, None, None
        else:
            bg_AVG = 0
            for i in range(bgStart, bgEnd):
                bg_AVG += self.counts[i] / (bgEnd - bgStart)
            return (np.amax(self.counts) - bg_AVG) / np.sqrt(bg_AVG), np.amax(self.counts), (
                        np.amax(self.counts) - bg_AVG)

    def removeMean(self):
        mean = np.mean(self.counts)
        self.counts = self.counts - mean

    def display(self, WN=True, NoX=False, xlabel='Wavelenght [nm]', ylabel='Counts [-]'):
        # snrString = ', SNR= '+str(self.getSNR()[0])[:6] + ', peak = '+str(self.getSNR()[1])[:7] + ', IT: {0} s'.format(self.integrationTime)
        snrString = "todo"
        if WN == True and NoX == False:
            xlabel = 'Wavenumber [cm-1]'
            plt.plot(self.wavenumbers, self.counts, label=self.label + snrString)
        if WN == False and NoX == False:
            plt.plot(self.wavelenghts, self.counts, label=self.label + snrString)
        if NoX == True:
            plt.plot(self.counts, label=self.label + snrString)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()

    def subtract(self, specToSub, specToSub_AcqTime=1):  # kwargs expected: time and factor
        # le specToSubAcqTime est à changer pour mettre le bon temps d'acquisition du spectre à soustraire si le spectre a soustraire n'est as de type Spectrum
        acqTime = self.integrationTime

        if type(specToSub) != Spectrum:
            self.counts = list(np.array(self.counts) - (acqTime * np.array(specToSub) / specToSub_AcqTime))

        if type(specToSub) == Spectrum:
            self.counts = list(
                np.array(self.counts) - (acqTime * np.array(specToSub.counts)) / specToSub.integrationTime)

    def removeThermalNoise(self, TNToRemove):
        assert type(TNToRemove) == Spectrum, 'Passed argument should be of type Spectrum'
        acqTime = self.integrationTime
        specToSub_AcqTime = TNToRemove.integrationTime

        self.counts = list(np.array(self.counts) - (acqTime * np.array(TNToRemove.counts) / specToSub_AcqTime))

    def factor(self, factor):
        self.counts = list(np.array(self.counts) * factor)

    def normalizeIntegration(self):
        integration = 0
        for i in self.counts:
            integration += abs(i)

        self.counts = list(np.array(self.counts) / integration)

    def normalizeCounts(self):
        self.counts = list(np.array(self.counts) / np.amax(self.counts))

    def normalizeTime(self):
        self.counts = list(np.array(self.counts) / self.integrationTime)

    def addSpectra(self, spectraToAdd):
        # Returns a spectra object containing this spectrum + the spectra or spectrum given as argument
        assert type(spectraToAdd) == Spectra or Spectrum, 'Expecting a Spectra or Spectrum type argument'
        newSpectra = Spectra([Spectrum(self.wavelenghts, self.counts, self.integrationTime, self.label)])

        if type(spectraToAdd) == Spectra:
            for spectrum in spectraToAdd.spectra:
                newSpectra.spectra.append(spectrum)

        if type(spectraToAdd) == Spectrum:
            newSpectra.spectra.append(spectraToAdd)

        else:
            raise ValueError('The given value is neither a Spectra or a Spectrum object')

        return newSpectra

    @staticmethod
    def polyFunc(x, fit_coefs):
        assert type(fit_coefs) == list
        fit_coefs.reverse()

        coefs = list(np.zeros(31))

        for i in range(len(fit_coefs)):
            coefs[i] = fit_coefs[i]

        y = []
        for i in x:
            y.append(coefs[0] + coefs[1] * i + coefs[2] * i ** 2 + coefs[3] * i ** 3 + coefs[4] * i ** 4 + coefs[
                5] * i ** 5 + coefs[6] * i ** 6 + coefs[7] * i ** 7 + coefs[8] * i ** 8 + coefs[9] * i ** 9 + coefs[
                         10] * i ** 10 + coefs[11] * i ** 11 + coefs[12] * i ** 12 + coefs[13] * i ** 13 + coefs[
                         14] * i ** 14 + coefs[15] * i ** 15 + coefs[16] * i ** 16 + coefs[17] * i ** 17 + coefs[
                         18] * i ** 18 + coefs[19] * i ** 19 + coefs[20] * i ** 20 + coefs[21] * i ** 21 + coefs[
                         22] * i ** 22 + coefs[23] * i ** 23 + coefs[24] * i ** 24 + coefs[25] * i ** 25 + coefs[
                         26] * i ** 26 + coefs[27] * i ** 27 + coefs[28] * i ** 28 + coefs[29] * i ** 29 + coefs[
                         30] * i ** 30)

        return y

    def polyfit(self, poly_order, show=False, replace=False, return_fit=False):
        # should return a spectum object ? i think so....
        # should i raise expection for it to work without any integration time?
        fit_coefs = list(np.polyfit(self.wavelenghts, self.counts, poly_order))
        y_fit = self.polyFunc(self.wavelenghts, fit_coefs)

        if show == True:
            plt.plot(self.wavelenghts, self.counts, 'k-', label='Data curve')
            plt.plot(self.wavelenghts, y_fit, 'r--', label='{0}th order polynomial fit'.format(poly_order))
            plt.plot(self.wavelenghts, (np.array(self.counts) - np.array(y_fit)), 'r', label='subtraction')
            plt.legend()
            plt.show()

        if replace == True:
            self.counts = (np.array(self.counts) - np.array(y_fit))

        label = self.label + '_fit'
        if return_fit == True:
            return Spectrum(self.wavelenghts, y_fit, self.integrationTime, label)

    def fft(self, show=False, replace=False, return_fit=False, fc=0.001, b=0.04, shift=54):

        # b = 0.08
        N = int(np.ceil((4 / b)))
        if not N % 2: N += 1
        n = np.arange(N)

        sinc_func = np.sinc(2 * fc * (n - (N - 1) / 2.))
        window = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
        sinc_func = sinc_func * window
        sinc_func = sinc_func / np.sum(sinc_func)

        self.fft_fit = np.convolve(self.counts, sinc_func)[shift:]
        self.fftFilteredCounts = np.array(self.counts) - np.array(self.fft_fit)[:len(self.counts)]

        trace1 = go.Scatter(
            # x=list(range(len(self.fft_fit))),
            x=self.wavelenghts,
            y=self.counts,
            mode='lines',
            name='Raw',
            line=dict(
                color='#000000',
                # dash='dash'
            )
        )
        trace2 = go.Scatter(
            # x=list(range(len(self.fft_fit))),
            x=self.wavelenghts,
            y=self.fft_fit,
            mode='lines',
            name='FFT Fit',
            marker=dict(
                color='#0000FF'
            )
        )

        layout = go.Layout(
            title='Low-Pass Filter',
            showlegend=True
        )

        trace_data = [trace1, trace2]
        fig = go.Figure(data=trace_data, layout=layout)

        if show == True:
            fig.show()
            plt.plot(self.wavelenghts, self.fftFilteredCounts)
            plt.show()

        if replace == True:
            self.counts = self.fftFilteredCounts

        if return_fit == True:
            return Spectrum(self.wavelenghts[:len(self.counts)], self.fft_fit[:len(self.counts)], self.integrationTime,
                            'fft_fit')

    def setZero(self, val):
        self.counts[0:val] = 0

    # works for data acquired on 20220802 for 10 sec integration in monkey brain samples in PBS
    def fixAberations(self, threshold=3.5, display=False):
        count = 0
        for i in range(2, len(self.counts) - 2):
            bef = self.counts[i - 1]
            at = self.counts[i]
            aft = self.counts[i + 1]
            edge_avg = (bef + aft) / 2

            if (at - edge_avg) > (threshold * np.sqrt(edge_avg)) and at > (bef and aft) and abs(at - bef) > abs(
                    bef - self.counts[i - 2]):
                count += 1
                self.counts[i] = edge_avg
        if display == True:
            print('{0} pixel values have been changed'.format(count))

    def smooth(self):
        for i in range(1, len(self.counts) - 1):
            val = (self.counts[i - 1] + self.counts[i] + self.counts[i + 1]) / 3
            self.counts[i] = val

    def cut(self, start, end, WN=False, WL=False):
        if WN == False and WL == False:
            self.wavenumbers = self.wavenumbers[start: end]
            self.wavelenghts = self.wavelenghts[start: end]
            self.counts = self.counts[start: end]

        if WN == True and WL == True:
            raise TypeError('WN and WL must not be True at the same time. Only one or none of them should be True')

        if WN == True and WL == False:
            if end == None:
                end = self.wavenumbers[-1]
            if start == None:
                start = self.wavenumbers[0]
            WN = list(self.wavenumbers)

            ADF_s = lambda list_value: abs(list_value - start)
            ADF_e = lambda list_value: abs(list_value - end)

            CV_s = min(WN, key=ADF_s)
            CV_e = min(WN, key=ADF_e)

            WN_start = WN.index(CV_s)
            WN_end = WN.index(CV_e)

            self.wavenumbers = self.wavenumbers[WN_start: WN_end]
            self.wavelenghts = self.wavelenghts[WN_start: WN_end]
            self.counts = self.counts[WN_start: WN_end]

        if WN == False and WL == True:
            if end == None:
                end = self.wavelenghts[-1]
            if start == None:
                start = self.wavelenghts[0]
            WL = list(self.wavelenghts)

            ADF_s = lambda list_value: abs(list_value - start)
            ADF_e = lambda list_value: abs(list_value - end)

            CV_s = min(WL, key=ADF_s)
            CV_e = min(WL, key=ADF_e)

            WL_start = WL.index(CV_s)
            WL_end = WL.index(CV_e) + 1

            self.wavenumbers = self.wavenumbers[WL_start: WL_end]
            self.wavelenghts = self.wavelenghts[WL_start: WL_end]
            self.counts = self.counts[WL_start: WL_end]

    def remove(self, start, end, WN=False, WL=False):
        if WN == False and WL == False:
            self.wavenumbers = np.delete(self.wavenumbers, np.s_[start: end])
            self.wavelenghts = np.delete(self.wavelenghts, np.s_[start: end])
            self.counts = np.delete(self.counts, np.s_[start: end])

        if WN == True and WL == True:
            raise TypeError('WN and WL must not be True at the same time. Only one or none of them should be True')

        if WN == True and WL == False:
            WN = list(self.wavenumbers)

            ADF_s = lambda list_value: abs(list_value - start)
            ADF_e = lambda list_value: abs(list_value - end)

            CV_s = min(WN, key=ADF_s)
            CV_e = min(WN, key=ADF_e)

            WN_start = WN.index(CV_s)
            WN_end = WN.index(CV_e)

            self.wavenumbers = np.delete(self.wavenumbers, np.s_[WN_start: WN_end])
            self.wavelenghts = np.delete(self.wavelenghts, np.s_[WN_start: WN_end])
            self.counts = np.delete(self.counts, np.s_[WN_start: WN_end])

        if WN == False and WL == True:
            WL = list(self.wavelenghts)

            ADF_s = lambda list_value: abs(list_value - start)
            ADF_e = lambda list_value: abs(list_value - end)

            CV_s = min(WL, key=ADF_s)
            CV_e = min(WL, key=ADF_e)

            WL_start = WL.index(CV_s)
            WL_end = WL.index(CV_e) + 1

            self.wavenumbers = np.delete(self.wavenumbers, range(WL_start, WL_end), 0)
            self.wavelenghts = np.delete(self.wavelenghts, range(WL_start, WL_end), 0)
            self.counts = np.delete(self.counts, range(WL_start, WL_end), 0)

    def changeXAxisValues(self, ref, print_info=False):
        if type(ref) == list:
            ref = ref
        if type(ref) == Spectrum:
            ref = list(ref.wavelenghts)

        new_counts = []
        init_to_remove = 0
        end_to_remove = 0
        for i in ref:
            ADF = lambda list_value: abs(list_value - i)
            CV = min(self.wavelenghts, key=ADF)
            index = list(self.wavelenghts).index(CV)
            # if index == 0:
            #     init_to_remove += 1
            #     continue
            # if index == len(self.wavelenghts):
            #     end_to_remove += 1
            #     continue
            #
            # if self.wavelenghts[index] > i:
            #     index_up = index
            #     index_down = index - 1
            # if self.wavelenghts[index] < i:
            #     index_up = index + 1
            #     index_down = index

            # val_up = self.counts[index_up]
            # val_down = self.counts[index_down]
            # x_up = self.wavelenghts[index_up]
            # x_down = self.wavelenghts[index_down]
            # m = (val_up - val_down) / (x_up - x_down)
            # b = (val_down) / (m * x_down)
            # new_count = (m * i) + b
            # new_counts.append(new_count)

            new_counts.append(self.counts[index])

        if print_info == True:
            print('{0} values were skipped due to index 0'.format(init_to_remove))
            print('{0} values were skipped due to index max'.format(end_to_remove))

        new_x = np.array(ref)
        new_x = np.delete(new_x, range(init_to_remove))
        new_x = new_x[0: new_x.size - end_to_remove]
        self.wavelenghts = new_x
        self.counts = new_counts
        self.wavenumbers = np.delete(self.wavenumbers, range(init_to_remove))
        self.wavenumbers = self.wavenumbers[0: new_x.size - end_to_remove]

    def getAbsorbance(self, ref):  # ref needs to be a Spectrum object
        relative_spec = (np.array(self.counts) / self.integrationTime) / (np.array(ref.counts) / ref.integrationTime)
        for i, value in enumerate(relative_spec):
            if value <= 0:
                relative_spec[i] = 0.001
        A = -1 / (np.log10(1 / relative_spec))
        return Spectrum(self.wavelenghts, A, 1, self.label)

    def savgolFilter(self, window_length=40, order=2):
        self.counts = savgol_filter(self.counts, window_length, order, mode="nearest")

    def butterworthFilter(self, cutoff_frequency=5, order=2):
        nyquist_frequency = 0.5 * len(self.counts)
        cutoff_frequency = cutoff_frequency / nyquist_frequency
        b, a = signal.butter(order, cutoff_frequency, btype='low', analog=False)
        filtered_intensities = signal.filtfilt(b, a, self.counts)
        self.counts =  self.counts - filtered_intensities


class Spectra:
    def __init__(self, items):
        self.spectra = []
        self.annotations = None
        for item in items:
            self.add(item)

        self.PC = []
        self.EVR = []
        self.SV = []
        self.PCA = None
        self.data = []
        self.labelList = []

        self._loadData()

    def _loadData(self):
        self.data = []
        self.labelList = []

        for spectrum in self.spectra:
            spectrum_features = np.array(spectrum.counts)
            spectrum_label = spectrum.label
            self.data.append(spectrum_features)
            self.labelList.append(spectrum_label)

    def removeLabel(self, label):
        label_index_list = []
        for i in range(len(self.labelList)):
            if self.labelList[i] == label:
                label_index_list.append(i)

        spectra = self.spectra
        spectra = np.delete(spectra, label_index_list, 0)
        self.spectra = spectra
        self._loadData()
        if self.annotations != None:
            self.annotations = np.delete(self.annotations, label_index_list, 0)

    def changeLabel(self, new_label):
        if type(new_label) == str:
            for spectrum in self.spectra:
                spectrum.label = new_label
        if type(new_label) == list or type(new_label) == np.ndarray:
            for i in range(len(self.spectra)):
                self.spectra[i].label = new_label[i]
        else:
            print(
                'Could not change the labels with an argument of type {0}. It needs to be a string, a list, or a np.array'.format(
                    type(new_label)))
        self._loadData()

    def getLabelSpectra(self, label):
        spectra = []
        for spectrum in self.spectra:
            if spectrum.label == label:
                spectra.append(spectrum)
        return Spectra(spectra)

    def addAnnotation(self, annotation):
        self.annotations = []
        for i in range(len(self.spectra)):
            self.annotations.append(annotation)

    def sumSpec(self):
        new_counts = np.zeros(np.shape(self.spectra[0].counts))
        integration_time = 0
        label = ''
        for spectrum in self.spectra:
            new_counts += spectrum.counts
            integration_time += spectrum.integrationTime
            label += spectrum.label + ', '

        return Spectrum(self.spectra[0].wavelenghts, new_counts, integration_time, label)

    def display(self, WN=True, label=True):
        if WN == False:
            for spectrum in self.spectra:
                plt.plot(spectrum.wavelenghts, spectrum.counts,
                         label=spectrum.label + ', integration = ' + str(spectrum.integrationTime)[:5] + ' s')
                plt.xlabel('Wavelenghts [nm]')

        if WN == True:
            for spectrum in self.spectra:
                plt.plot(spectrum.wavenumbers, spectrum.counts,
                         label=spectrum.label + ', integration = ' + str(spectrum.integrationTime)[:5] + ' s')
                plt.xlabel('Wavenumbers [cm-1]')

        plt.ylabel('Counts [-]')
        if label == True:
            plt.legend()
        plt.show()

    def display2Colored(self, label1, label2, WN=True, display_label=True):
        if WN == False:
            for spectrum in self.spectra:
                if spectrum.label == label1:
                    plt.plot(spectrum.wavelenghts, spectrum.counts, 'k')
                    # label=spectrum.label + ', integration = ' + str(spectrum.integrationTime)[:5] + ' s')
                    plt.xlabel('Wavelenghts [nm]')
                if spectrum.label == label2:
                    plt.plot(spectrum.wavelenghts, spectrum.counts, 'r')
                    # label=spectrum.label + ', integration = ' + str(spectrum.integrationTime)[:5] + ' s')
                    plt.xlabel('Wavelenghts [nm]')

        if WN == True:
            for spectrum in self.spectra:
                if spectrum.label == label1:
                    plt.plot(spectrum.wavenumbers, spectrum.counts, 'k')
                    # label=spectrum.label + ', integration = ' + str(spectrum.integrationTime)[:5] + ' s')
                    plt.xlabel('Wavenumbers [cm-1]')
                if spectrum.label == label2:
                    plt.plot(spectrum.wavenumbers, spectrum.counts, 'r')
                    # label=spectrum.label + ', integration = ' + str(spectrum.integrationTime)[:5] + ' s')
                    plt.xlabel('Wavenumbers [cm-1]')

        plt.ylabel('Counts [-]')

        if display_label == True:
            plt.plot([], [], 'k', label=label1)
            plt.plot([], [], 'r', label=label2)
            plt.legend()

        plt.show()

    def display3Colored(self, label1, label2, label3, WN=True, display_label=True):
        if WN == False:
            for spectrum in self.spectra:
                if spectrum.label == label1:
                    plt.plot(spectrum.wavelenghts, spectrum.counts, 'k')
                    # label=spectrum.label + ', integration = ' + str(spectrum.integrationTime)[:5] + ' s')
                    plt.xlabel('Wavelenghts [nm]')
                if spectrum.label == label2:
                    plt.plot(spectrum.wavelenghts, spectrum.counts, 'r')
                    # label=spectrum.label + ', integration = ' + str(spectrum.integrationTime)[:5] + ' s')
                    plt.xlabel('Wavelenghts [nm]')
                if spectrum.label == label3:
                    plt.plot(spectrum.wavelenghts, spectrum.counts, 'b')
                    # label=spectrum.label + ', integration = ' + str(spectrum.integrationTime)[:5] + ' s')
                    plt.xlabel('Wavelenghts [nm]')

        if WN == True:
            for spectrum in self.spectra:
                if spectrum.label == label1:
                    plt.plot(spectrum.wavenumbers, spectrum.counts, 'k')
                    # label=spectrum.label + ', integration = ' + str(spectrum.integrationTime)[:5] + ' s')
                    plt.xlabel('Wavenumbers [cm-1]')
                if spectrum.label == label2:
                    plt.plot(spectrum.wavenumbers, spectrum.counts, 'r')
                    # label=spectrum.label + ', integration = ' + str(spectrum.integrationTime)[:5] + ' s')
                    plt.xlabel('Wavenumbers [cm-1]')
                if spectrum.label == label3:
                    plt.plot(spectrum.wavenumbers, spectrum.counts, 'b')
                    # label=spectrum.label + ', integration = ' + str(spectrum.integrationTime)[:5] + ' s')
                    plt.xlabel('Wavenumbers [cm-1]')

        plt.ylabel('Counts [-]')
        if display_label == True:
            plt.plot([], [], 'k', label=label1)
            plt.plot([], [], 'r', label=label2)
            plt.plot([], [], 'b', label=label3)
            plt.legend()
        plt.show()

    def display2ColoredMeanSTD(self, label1, label2, WN=False, display_label=True):
        label1List = []
        label2List = []
        for spectrum in self.spectra:
            if spectrum.label == label1:
                label1List.append(spectrum.counts)
            if spectrum.label == label2:
                label2List.append(spectrum.counts)
        mean1 = np.mean(np.array(label1List), axis=0)
        std1 = np.std(np.array(label1List), axis=0)
        mean2 = np.mean(np.array(label2List), axis=0)
        std2 = np.std(np.array(label2List), axis=0)

        if WN == False:
            plt.plot(self.spectra[0].wavelenghts, mean1, 'k')
            plt.fill_between(self.spectra[0].wavelenghts, mean1 - std1, mean1 + std1, facecolor='k', alpha=0.5)
            plt.plot(self.spectra[0].wavelenghts, mean2, 'r')
            plt.fill_between(self.spectra[0].wavelenghts, mean2 - std2, mean2 + std2, facecolor='r', alpha=0.5)

        if WN == True:
            plt.plot(self.spectra[0].wavenumbers, mean1, 'k')
            plt.fill_between(self.spectra[0].wavenumbers, mean1 - std1, mean1 + std1, facecolor='k', alpha=0.5)
            plt.plot(self.spectra[0].wavenumbers, mean2, 'r')
            plt.fill_between(self.spectra[0].wavenumbers, mean2 - std2, mean2 + std2, facecolor='r', alpha=0.5)

        plt.ylabel('Counts [-]')

        if display_label == True:
            plt.plot([], [], 'k', label=label1)
            plt.plot([], [], 'r', label=label2)
            plt.legend()

        plt.show()

    def display3ColoredMeanSTD(self, label1, label2, label3, WN=False, display_label=True):
        label1List = []
        label2List = []
        label3List = []
        for spectrum in self.spectra:
            if spectrum.label == label1:
                label1List.append(spectrum.counts)
            if spectrum.label == label2:
                label2List.append(spectrum.counts)
            if spectrum.label == label3:
                label3List.append(spectrum.counts)
        mean1 = np.mean(np.array(label1List), axis=0)
        std1 = np.std(np.array(label1List), axis=0)
        mean2 = np.mean(np.array(label2List), axis=0)
        std2 = np.std(np.array(label2List), axis=0)
        mean3 = np.mean(np.array(label3List), axis=0)
        std3 = np.std(np.array(label3List), axis=0)

        if WN == False:
            plt.plot(self.spectra[0].wavelenghts, mean1, 'k')
            plt.fill_between(self.spectra[0].wavelenghts, mean1 - std1, mean1 + std1, facecolor='k', alpha=0.5)
            plt.plot(self.spectra[0].wavelenghts, mean2, 'r')
            plt.fill_between(self.spectra[0].wavelenghts, mean2 - std2, mean2 + std2, facecolor='r', alpha=0.5)
            plt.plot(self.spectra[0].wavelenghts, mean3, 'b')
            plt.fill_between(self.spectra[0].wavelenghts, mean3 - std3, mean3 + std3, facecolor='b', alpha=0.5)

        if WN == True:
            plt.plot(self.spectra[0].wavenumbers, mean1, 'k')
            plt.fill_between(self.spectra[0].wavenumbers, mean1 - std1, mean1 + std1, facecolor='k', alpha=0.5)
            plt.plot(self.spectra[0].wavenumbers, mean2, 'r')
            plt.fill_between(self.spectra[0].wavenumbers, mean2 - std2, mean2 + std2, facecolor='r', alpha=0.5)
            plt.plot(self.spectra[0].wavenumbers, mean3, 'b')
            plt.fill_between(self.spectra[0].wavenumbers, mean3 - std3, mean3 + std3, facecolor='b', alpha=0.5)
        plt.ylabel('Counts [-]')

        if display_label == True:
            plt.plot([], [], 'k', label=label1)
            plt.plot([], [], 'r', label=label2)
            plt.plot([], [], 'b', label=label3)
            plt.legend()

        plt.show()

    @staticmethod
    def _Unique(liste):
        unique_list = []
        for elem in liste:
            if elem not in unique_list:
                unique_list.append(elem)
        return unique_list

    def displayMeanSTD(self, WN=False, display_label=True):
        labels = self._Unique(self.labelList)
        # color_list = ["#" + ''.join([random.choice('0123456789ABCDEF') for i in range(6)]) for j in range(len(labels))]
        color_list = ['k', 'red', 'green', 'blue', '#332288', 'orange', '#AA4499', '#88CCEE', '#117733', '#999933',
                      '#44AA99', '#DDCC77', '#805E2B', 'yellow']
        empty_lists = [[] for _ in range(len(labels))]
        WL_values = [[] for _ in range(len(labels))]
        WN_values = [[] for _ in range(len(labels))]
        for spectrum in self.spectra:
            for i in range(len(labels)):
                if spectrum.label == labels[i]:  # ca va pt chier ici
                    empty_lists[i].append(spectrum.counts)
                    if len(WL_values[i]) == 0:
                        WL_values[i] = spectrum.wavelenghts
                        WN_values[i] = spectrum.wavenumbers

        data_ordered = empty_lists
        data_means = []
        data_STD = []
        for i in range(len(labels)):
            data_means.append(np.mean(data_ordered[i], axis=0))
            data_STD.append(np.std(data_ordered[i], axis=0))

        if WN == False:
            for i in range(len(labels)):
                plt.plot(WL_values[i], data_means[i], color=color_list[i])
                plt.fill_between(WL_values[i], data_means[i] - data_STD[i], data_means[i] + data_STD[i],
                                 color=color_list[i], alpha=0.5)
            plt.xlabel('Wavelengths [nm]')

        if WN == True:
            for i in range(len(labels)):
                plt.plot(WN_values[i], data_means[i], color=color_list[i])
                plt.fill_between(WN_values[i], data_means[i] - data_STD[i], data_means[i] + data_STD[i],
                                 color=color_list[i], alpha=0.5)
            plt.xlabel('Wavenumbers [cm-1]')
        plt.ylabel('Counts [-]')

        if display_label == True:
            for i in range(len(labels)):
                plt.plot([], [], label=labels[i], color=color_list[i])
            plt.legend()

        plt.show()

    def removeThermalNoise(self, TNToRemove):
        for spectrum in self.spectra:
            spectrum.removeThermalNoise(TNToRemove)

    def subtract(self, specToSub, specToSub_AcqTime=1):
        if type(specToSub) != Spectrum:
            for spectrum in self.spectra:
                spectrum.counts = list(
                    np.array(spectrum.counts) - (spectrum.integrationTime * np.array(specToSub) / specToSub_AcqTime))

        if type(specToSub) == Spectrum:
            for spectrum in self.spectra:
                spectrum.counts = list(np.array(spectrum.counts) - (
                            spectrum.integrationTime * np.array(specToSub.counts) / specToSub.integrationTime))

    def smooth(self):
        for spectrum in self.spectra:
            spectrum.smooth()

    def fixAberations(self, threshold=3.5, display=False):
        for spectrum in self.spectra:
            spectrum.fixAberations(threshold=threshold, display=display)
        self._loadData()

    def normalizeIntegration(self):
        for spectrum in self.spectra:
            spectrum.normalizeIntegration()

    def normalizeCounts(self):
        for spectrum in self.spectra:
            spectrum.normalizeCounts()

    def normalizeTime(self):
        for spectrum in self.spectra:
            spectrum.normalizeTime()

    def _getSpectraSum(self):
        datashape = np.shape(self.spectra[0].counts)
        sum = np.zeros(datashape)
        for spectrum in self.spectra:
            sum += spectrum.counts
        self.spectraSum = sum

    def _getSpectraAVG(self):
        datashape = np.shape(self.spectra[0].counts)
        AVG = np.zeros(datashape)
        for spectrum in self.spectra:
            AVG += (np.array(spectrum.counts) / len(self.spectra))
        self.spectraAVG = AVG

    def add(self, *items):
        for item in items:
            assert type(item) == Spectrum or Spectra, 'Expecting a Spectra or Spectrum type argument'

            if type(item) == Spectra:
                before = len(self.spectra)
                for spectrum in item.spectra:
                    self.spectra.append(spectrum)

                if self.annotations != None and item.annotations != None:
                    for i in item.annotations:
                        self.annotations.append(i)

                after = len(self.spectra)
                assert after == (before + len(
                    item.spectra)), 'The spectra that were supposed to be added to this object have not been properly added'

            if type(item) == Spectrum:
                before = len(self.spectra)
                self.spectra.append(item)

                after = len(self.spectra)
                assert after == before + 1, 'The spectrum that was supposed to be added to this object has not been properly added'

        self._loadData()

    def cut(self, start, end, WN=False, WL=False):
        for Spectrum in self.spectra:
            Spectrum.cut(start, end, WN, WL)
        self._loadData()

    def remove(self, start, end, WN=False, WL=False):
        for Spectrum in self.spectra:
            Spectrum.remove(start, end, WN, WL)
        self._loadData()

    def pca(self, nbOfComp=10, SC=False):
        self._loadData()
        data = self.data
        if SC == True:
            self._standardizeData()
            data = self.SCData

        self.PCA = PCA(n_components=nbOfComp)
        self.PCA.fit(data)

        for i in range(nbOfComp):
            self.SV.append(self.PCA.singular_values_[i])
            self.EVR.append(self.PCA.explained_variance_ratio_[i])
            self.PC.append(Spectrum(self.spectra[0].wavelenghts, self.PCA.components_[i], 1,
                                    'PC{0}, val propre = {1}'.format(i + 1, self.PCA.explained_variance_ratio_[i])))

    def subtractPCToData(self, PC):
        self._getSpectraAVG()
        self._getPCAdf()
        PCSpectrum = self.PC[PC - 1].counts
        PCEigVal = []
        print(self.pca_df)

        for EV in range(len(self.data)):
            PCEigVal.append(self.pca_df.iat[EV, PC - 1])
        print(PCEigVal)

        newData = []
        for i in range(len(self.data)):
            newData.append(self.data[i] - (PCEigVal[i] * PCSpectrum) - self.spectraAVG)
            # plt.plot(self.data[i], label='A spectrum')
            # plt.plot(newData[i], label="A spectrum - it's PC1")
            # plt.legend()
            # plt.show()

        return newData

    def pcaDisplay(self, *PCs, WN=False):
        if WN == True:
            for PC in PCs:
                plt.plot(self.spectra[0].wavenumbers, self.PC[PC - 1].counts, label=self.PC[PC - 1].label)
            plt.xlabel('Wavenumbers [cm-1]')
        if WN == False:
            for PC in PCs:
                plt.plot(self.spectra[0].wavelenghts, self.PC[PC - 1].counts, label=self.PC[PC - 1].label)
            plt.xlabel('Wavelengths [nm]')
        plt.legend()
        plt.show()

    def _getPCAdf(self, SC=False):
        self._loadData()
        data = self.data

        if SC == True:
            self._standardizeData()
            data = self.SCData

        pca_data = self.PCA.fit_transform(data)
        self.PCAlabels = []
        self.PCAcolumns = []

        for spectrum in self.spectra:
            self.PCAlabels.append(spectrum.label)

        for PC in range(len(self.PC)):
            self.PCAcolumns.append('PC{0}'.format(PC + 1))

        self.pca_df = pd.DataFrame(pca_data, index=self.PCAlabels, columns=self.PCAcolumns)

    def pcaScatterPlot(self, PCx, PCy=None, PCz=None, AnnotationToDisplay=None, show_annotations=False):
        self._loadData()
        self._getPCAdf()

        if AnnotationToDisplay == None:
            if show_annotations == False:
                if PCy == None and PCz == None:
                    fig = px.scatter(self.pca_df, x='PC{0}'.format(PCx), color=self.PCAlabels)

                if PCz == None and PCy != None:
                    fig = px.scatter(self.pca_df, x='PC{0}'.format(PCx), y='PC{0}'.format(PCy), color=self.PCAlabels)

                if PCy != None and PCz != None:
                    fig = px.scatter_3d(self.pca_df, x='PC{0}'.format(PCx), y='PC{0}'.format(PCy),
                                        z='PC{0}'.format(PCz), color=self.PCAlabels)

            if show_annotations == True:
                if PCy == None and PCz == None:
                    fig = px.scatter(self.pca_df, x='PC{0}'.format(PCx), color=self.PCAlabels, text=self.annotations)

                if PCz == None and PCy != None:
                    fig = px.scatter(self.pca_df, x='PC{0}'.format(PCx), y='PC{0}'.format(PCy), color=self.PCAlabels,
                                     text=self.annotations)

                if PCy != None and PCz != None:
                    fig = px.scatter_3d(self.pca_df, x='PC{0}'.format(PCx), y='PC{0}'.format(PCy),
                                        z='PC{0}'.format(PCz),
                                        color=self.PCAlabels, text=self.annotations)

        if AnnotationToDisplay != None:
            toDisplayList = []
            for annotation in AnnotationToDisplay:
                for i in range(len(self.annotations)):
                    if self.annotations[i] == annotation:
                        toDisplayList.append(i)
            temp_pca_df = self.pca_df.iloc[toDisplayList]
            temp_PCAlabels = []
            for i in toDisplayList:
                temp_PCAlabels.append(self.PCAlabels[i])

            if show_annotations == False:
                if PCy == None and PCz == None:
                    fig = px.scatter(temp_pca_df, x='PC{0}'.format(PCx), color=temp_PCAlabels)

                if PCz == None and PCy != None:
                    fig = px.scatter(temp_pca_df, x='PC{0}'.format(PCx), y='PC{0}'.format(PCy), color=temp_PCAlabels)

                if PCy != None and PCz != None:
                    fig = px.scatter_3d(temp_pca_df, x='PC{0}'.format(PCx), y='PC{0}'.format(PCy),
                                        z='PC{0}'.format(PCz), color=temp_PCAlabels)

            if show_annotations == True:

                temp_annotations = []
                for i in toDisplayList:
                    temp_annotations.append(self.annotations[i])
                if PCy == None and PCz == None:
                    fig = px.scatter(temp_pca_df, x='PC{0}'.format(PCx), color=temp_PCAlabels, text=temp_annotations)
                    print('ok')

                if PCz == None and PCy != None:
                    fig = px.scatter(temp_pca_df, x='PC{0}'.format(PCx), y='PC{0}'.format(PCy), color=temp_PCAlabels,
                                     text=temp_annotations)
                    print('ok')

                if PCy != None and PCz != None:
                    fig = px.scatter_3d(temp_pca_df, x='PC{0}'.format(PCx), y='PC{0}'.format(PCy),
                                        z='PC{0}'.format(PCz),
                                        color=temp_PCAlabels, text=temp_annotations)
                    print('ok')

        plot(fig)

    def getRatioPhotonPerCount(self):
        ratios = []
        means = []
        for val in range(len(self.spectra[0].wavelenghts)):
            data = []
            for spectrum in self.spectra:
                data.append(spectrum.counts[val])
                WL = spectrum.wavelenghts[val]
            STD = np.std(data)
            mean = np.mean(data)
            means.append(mean)
            ratio = mean / (STD ** 2)
            ratios.append(ratio)

            print('WL: {3},mean: {0}, STD: {1}, ratio: {2}'.format(mean, STD, ratio, WL))

        plt.plot(self.spectra[0].wavelenghts, ratios)
        plt.plot(self.spectra[0].wavelenghts, np.array(self.spectra[0].counts) / 600)
        plt.xlabel('Wavelenghts [cm-1]')
        plt.ylabel('Ratio [photons / counts]')
        plt.show()

        plt.plot(means, ratios, 'o')
        plt.xlabel('Mean [counts]')
        plt.ylabel('Ratio [Photon/count]')
        plt.show()

    def setZero(self, val):
        for spectrum in self.spectra:
            spectrum.setZero(val)

    # def fixSpec(self):
    #     for spectrum in self.spectra:
    #         spectrum.fixSpec()

    def _standardizeData(self, replace=False):
        self.SCData = StandardScaler().fit_transform(self.data)
        if replace == True:
            for i in range(len(self.spectra)):
                self.spectra[i].counts = self.SCData[i]
        self._loadData()

    def fft(self, show=False, replace=False, return_fit=False, fc=0.001, b=0.04, shift=54):
        for spectrum in self.spectra:
            spectrum.fft(show, replace, return_fit, fc, b, shift)

    def polyfit(self, poly_order, show=False, replace=False, return_fit=False):
        for spectrum in self.spectra:
            spectrum.polyfit(poly_order, show, replace, return_fit)

    def lda(self, n_components=1, SC=False, WN=True):
        # might be usless since ldaScatterplot is doing this again
        # I dont really understand the n_component stuff going on
        self._loadData()

        assert len(self.data) == len(self.labelList), "'data' and 'label' arrays must be the same lenght"

        data = self.data

        # if SC == True:
        #     self._standardizeData()
        #     assert len(self.SCData) == len(self.labelList), "'SCdata' and 'label' arrays must be the same lenght"
        #     data = self.SCData

        self.LDA = LinearDiscriminantAnalysis(n_components=n_components)
        LDA_comp = self.LDA.fit(data, self.labelList)

        if WN == False:
            plt.plot(self.spectra[0].wavelenghts, abs(LDA_comp.coef_[0]))
            plt.xlabel('Wavelengths [nm]')
        if WN == True:
            plt.plot(self.spectra[0].wavenumbers, abs(LDA_comp.coef_[0]))
            plt.xlabel('Wavenumbers [cm-1]')
        plt.show()

    def ldaScatterPlot(self, LDx, LDy=None, LDz=None, SC=False):
        self._loadData()
        self.lda()

        data = self.data
        if SC == True:
            self._standardizeData()
            data = self.SCData

        lda_data = self.LDA.fit_transform(data, self.labelList)
        labels = []
        columns = []

        for spectrum in self.spectra:
            labels.append(spectrum.label)

        for LD in range(self.LDA.n_components):
            columns.append('LD{0}'.format(LD + 1))

        self.lda_df = pd.DataFrame(lda_data, index=labels, columns=columns)

        if LDy == None and LDz == None:
            fig = px.scatter(self.lda_df, x='LD{0}'.format(LDx), color=labels)

        if LDz == None and LDy != None:
            fig = px.scatter(self.lda_df, x='LD{0}'.format(LDx), y='LD{0}'.format(LDy), color=labels)

        if LDy != None and LDz != None:
            fig = px.scatter_3d(self.lda_df, x='LD{0}'.format(LDx), y='LD{0}'.format(LDy), z='LD{0}'.format(LDz),
                                color=labels)

        plot(fig)

    def ldaOnPCsScatteredPlot(self, LDx, LDy=None, n_components=2):
        self._loadData()
        self._getPCAdf()

        # assert len(self.data) == len(self.labelList), "'data' and 'label' arrays must be the same lenght"

        pc1 = self.pca_df['PC1']
        pc2 = self.pca_df['PC2']
        pc3 = self.pca_df['PC3']
        pc4 = self.pca_df['PC4']
        pc5 = self.pca_df['PC5']
        pc6 = self.pca_df['PC6']
        pc7 = self.pca_df['PC7']
        pc8 = self.pca_df['PC8']
        pc9 = self.pca_df['PC9']
        pc10 = self.pca_df['PC10']
        data = []
        for i in range(len(pc1)):
            val = []
            # val.append(pc1[i])
            val.append(pc2[i])
            val.append(pc3[i])
            # val.append(pc4[i])
            # val.append(pc5[i])
            # val.append(pc6[i])
            # val.append(pc7[i])
            # val.append(pc8[i])
            # val.append(pc9[i])
            # val.append(pc10[i])
            data.append(val)

        # index_to_del = []
        # for i in range(len(self.spectra)):
        #     if self.spectra[i].label == 'MIXED':
        #         index_to_del.append(i)
        # self.spectra = np.delete(self.spectra, index_to_del, 0)
        # self._loadData()
        # data = np.delete(data, index_to_del, 0)

        self.LDA = LinearDiscriminantAnalysis(n_components=n_components)
        lda_data = self.LDA.fit_transform(data, self.labelList)

        labels = []
        columns = []

        for spectrum in self.spectra:
            labels.append(spectrum.label)

        for LD in range(self.LDA.n_components):
            columns.append('LD{0}'.format(LD + 1))

        self.lda_df = pd.DataFrame(lda_data, index=labels, columns=columns)

        if LDy == None:
            fig = px.scatter(self.lda_df, x='LD{0}'.format(LDx), color=labels)

        if LDy != None:
            fig = px.scatter(self.lda_df, x='LD{0}'.format(LDx), y='LD{0}'.format(LDy), color=labels)

        plot(fig)

    def changeXAxisValues(self, ref, print_info=False):
        for spectrum in self.spectra:
            spectrum.changeXAxisValues(ref, print_info)

    def kpca(self, nbOfComp=10):
        self._loadData()
        data = self.data
        df = pd.DataFrame(data, columns=self.spectra[0].wavelenghts)
        df['labels'] = self.labelList
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        X_train = StandardScaler().fit_transform(X_train)
        X_test = StandardScaler().fit_transform(X_test)

        kpca_test = KernelPCA()
        kpca_test = kpca_test.fit_transform(X_train)
        explained_variance = np.var(kpca_test, axis=0)
        explained_variance_ratio = explained_variance / np.sum(explained_variance)

        evr = explained_variance_ratio
        cvr = np.cumsum(explained_variance_ratio)

        kpca_df = pd.DataFrame()
        kpca_df['Cumulative Variance Ratio'] = cvr
        kpca_df['Explained Variance Ratio'] = evr

        kpca = KernelPCA(n_components=nbOfComp)
        X_train = kpca.fit_transform(X_train)
        X_test = kpca.transform(X_test)

        param_grid = [{'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10], 'kernel': ['rbf']}]
        grid = GridSearchCV(SVC(), param_grid, verbose=2)
        grid.fit(X_train, y_train)

        y_pred = grid.predict(X_test)
        print(grid.best_params_)
        print(nbOfComp)
        print('Confusion matrix:\n', metrics.confusion_matrix(y_test, y_pred))
        print('Accuracy:', np.round(metrics.accuracy_score(y_test, y_pred), 4))
        print(metrics.classification_report(y_test, y_pred))
        # classifier = grid

        # visualise train set
        # X_set, y_set = X_train, y_train
        # X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
        #                      np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
        # print(X1)
        # print(X2)
        # plt.contourf(X1, X2, grid.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
        #                                         alpha=0.75, cmap=ListedColormap(('red', 'green', 'blue')))
        # plt.xlim(X1.min(), X1.max())
        # plt.ylim(X2.min(), X2.max())
        # for i, j in enumerate(np.unique(y_set)):
        #     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
        #                 c=ListedColormap(('red', 'green', 'blue'))(i), label=j)
        # plt.title('SVC (training set)')
        # plt.xlabel('PC1')
        # plt.ylabel('PC2')
        # plt.legend()
        # plt.show()

        # visualise test set

    def umap(self, n_components=2, n_neighbors=15, metric='euclidean', min_dist=0.1, title=None, display=True):
        self._loadData()
        if title == None:
            title = '{0} neighbors'.format(n_neighbors)

        if n_components == 2:
            self.umap_2d = UMAP(n_components=n_components, n_neighbors=n_neighbors, metric=metric, min_dist=min_dist, init='random', random_state=0)
            self.UMAP_proj_2d = self.umap_2d.fit_transform(self.data)

            if display == True:
                fig_2d = px.scatter(
                    self.UMAP_proj_2d, x=0, y=1,
                    color=self.labelList, labels=self.labelList, title=title)

                fig_2d.show()

        if n_components == 3:
            self.umap_3d = UMAP(n_components=n_components, n_neighbors=n_neighbors, metric=metric, min_dist=min_dist,
                           init='random', random_state=0)
            self.UMAP_proj_3d = self.umap_3d.fit_transform(self.data)

            if display == True:
                fig_3d = px.scatter_3d(
                    self.UMAP_proj_3d, x=0, y=1, z=2,
                    color=self.labelList, labels=self.labelList, title=title)

                fig_3d.show()

    def tsne(self, n_components=2, perplexity=30, metric='euclidean', title=None, display=True):
        self._loadData()
        if title == None:
            title = 'perplexity = {0}'.format(perplexity)

        if n_components == 2:
            self.tsne_2d = TSNE(n_components=n_components, perplexity=perplexity, metric=metric,
                           init='random', random_state=0)

            self.TSNE_proj_2d = self.tsne_2d.fit_transform(self.data)

            if display == True:
                fig_2d = px.scatter(
                    self.TSNE_proj_2d, x=0, y=1,
                    color=self.labelList, labels=self.labelList, title=title)

                fig_2d.show()

        if n_components == 3:
            self.tsne_3d = TSNE(n_components=n_components, perplexity=perplexity, metric=metric,
                           init='random', random_state=0)

            self.TSNE_proj_3d = self.tsne_3d.fit_transform(self.data)

            if display == True:
                fig_3d = px.scatter_3d(
                    self.TSNE_proj_3d, x=0, y=1, z=2,
                    color=self.labelList, labels=self.labelList, title=title)

                fig_3d.show()

    def getAbsorbance(self, ref):  # ref needs to be a Spectrum object
        abs_spectra = []
        for spectrum in self.spectra:
            abs_spectra.append(spectrum.getAbsorbance(ref))
        return Spectra(abs_spectra)

    def removeSpectra(self, start, end):
        del self.spectra[start: end]
        self._loadData()

    def plotPCOnBarCode(self, PC, shift=0):
        self._loadData()
        self._getPCAdf()
        BarCode = np.array(self.labelList)
        assert len(BarCode) == len(
            self.pca_df), 'You should have as many elements in the PCs dataframe then in the BarCode provided. Right now, you have {0} elements in the PCs dataframe and {1} elements in the BarCode'.format(
            len(self.pca_df), len(BarCode))
        digital_BarCode = []
        for i in range(len(BarCode)):
            if BarCode[i] == 'WHITE':
                digital_BarCode.append(1)
            if BarCode[i] == 'MIXED':
                digital_BarCode.append(0.5)
            if BarCode[i] == 'GREY':
                digital_BarCode.append(0)
        PCdata = self.pca_df['PC{0}'.format(PC)].to_numpy()
        PCdata_max = np.amax(PCdata)
        PCdata_min = np.amin(PCdata)
        expansion = abs(PCdata_max) + abs(PCdata_min)
        PCdata = PCdata / expansion
        new_min = np.amin(PCdata)
        PCdata = PCdata - new_min
        assert len(digital_BarCode) == len(
            PCdata), 'Some labels in the BarCode are probably wrong! Make sure that all elements in the BarCod are either "WHITE", "GREY" or "MIXED"'
        x = np.array(range(len(PCdata)))
        x = x / 10  # now in mm
        x_shift = x + shift

        plt.plot(x, PCdata, 'r', label='PC{0}'.format(PC))
        plt.plot(x_shift, digital_BarCode, 'ko', label='BarCode')
        plt.xlabel('Distance [mm]')
        plt.legend()
        plt.show()

    def shiftSpectra(self, shift):  # shift in nb of data
        # shift positiv will shift the data to the left
        # shift negative will shift self.labelList to the left
        if shift > 0:
            self.spectra = np.delete(self.spectra, np.s_[:shift:])
            for i in range(len(self.spectra)):
                self.spectra[i].label = self.labelList[i]
            self._loadData()
        if shift < 0:
            for i in range(len(self.spectra) - abs(shift)):
                self.spectra[i].label = self.labelList[i + abs(shift)]
            self.spectra = np.delete(self.spectra, np.s_[len(self.spectra) - abs(shift)::])
            self._loadData()

    def removeMean(self):
        for spectrum in self.spectra:
            spectrum.removeMean()
        self._loadData()

    def savgolFilter(self, window_length=51, order=2):
        for spectrum in self.spectra:
            spectrum.savgolFilter(window_length=window_length, order=order)
        self._loadData()

    def combineSpectra(self, add):
        # returns a new spectra object
        spectra_list = []
        for i in range(0, len(self.spectra) - add, add):
            new_spec = self.spectra[i]
            int_time = new_spec.integrationTime
            for j in range(i + 1, i + add):
                if new_spec.label == self.spectra[j].label:

                    new_spec = new_spec.addSpectra(self.spectra[j])
                    new_spec = new_spec.sumSpec()
                    new_spec.label = self.spectra[j].label
            if new_spec.integrationTime == int_time * add:
                spectra_list.append(new_spec)
            else:
                print('For label "{0}", some data were left behind due to a number of spectra that does not divide by {1}'.format(new_spec.label, add))

        return Spectra(spectra_list)

    def kmeanSHAV(self, data, n_clusters=2, graph=False, barcode=False, barplot=False, title=None):
        # barcode should be used for SHAV like data when labels are 'WHITE', 'GREY' and 'MIXTED'

        if data == 'PCA':
            kmeans = KMeans(n_clusters=n_clusters).fit(self.PCAcolumns.astype("double"))
            raw = self.PCAcolumns

        elif data == 'UMAP2d':
            kmeans = KMeans(n_clusters=n_clusters).fit(self.UMAP_proj_2d.astype("double"))
            raw = self.UMAP_proj_2d

        elif data == 'UMAP3d':
            kmeans = KMeans(n_clusters=n_clusters).fit(self.UMAP_proj_3d.astype("double"))
            raw = self.UMAP_proj_3d

        elif data == 'TSNE2d':
            kmeans = KMeans(n_clusters=n_clusters).fit(self.TSNE_proj_2d.astype("double"))
            raw = self.TSNE_proj_2d

        elif data == 'TSNE3d':
            kmeans = KMeans(n_clusters=n_clusters).fit(self.TSNE_proj_3d.astype("double"))
            raw = self.TSNE_proj_3d

        else:
            raise ValueError('"{0}" does not correspond to an expected str.'.format(data))

        if barcode == True:
            BarCode = np.array(self.labelList)
            assert len(BarCode) == len(
                kmeans.labels_), 'You should have as many elements in the PCs dataframe then in the BarCode provided. Right now, you have {0} elements in the PCs dataframe and {1} elements in the BarCode'.format(
                len(self.pca_df), len(BarCode))
            digital_BarCode = []
            for i in range(len(BarCode)):
                if BarCode[i] == 'WHITE':
                    digital_BarCode.append(1)
                if BarCode[i] == 'MIXED':
                    digital_BarCode.append(0.5)
                if BarCode[i] == 'GREY':
                    digital_BarCode.append(0)

            assert len(digital_BarCode) == len(
                kmeans.labels_), 'Some labels in the BarCode are probably wrong! Make sure that all elements in the BarCod are either "WHITE", "GREY" or "MIXED"'
            x = np.array(range(len(kmeans.labels_)))
            x = x / 2  # now in mm

            #compute accuracy
            right = 0
            wrong = 0
            for i in range(len(kmeans.labels_)):
                if kmeans.labels_[i] == digital_BarCode[i]:
                    right += 1
                else:
                    wrong += 1
            self.kmean_accuracy_ratio = right / (right + wrong)
            if self.kmean_accuracy_ratio < 0.5:
                self.kmean_accuracy_ratio = 1 - self.kmean_accuracy_ratio

            if barplot == True:
                plt.plot(x, kmeans.labels_, 'ro', label=data + 'accuracy = {0}'.format(self.kmean_accuracy_ratio))
                plt.plot(x, digital_BarCode, 'ko', label='BarCode')
                plt.xlabel('Distance [mm]')
                plt.title(title)
                plt.legend()
                plt.show()

        if graph == True:
            centers = kmeans.cluster_centers_

            mid_point_x = np.mean(centers[:, 0])
            mid_point_y = np.mean(centers[:, 1])
            slope = (centers[1][1] - centers[0][1]) / (centers[1][0] - centers[0][1])
            inverse_slope = (-1)/slope
            b = mid_point_y - (inverse_slope * mid_point_x)
            # y = inverse_slope * x + b

            # Step size of the mesh. Decrease to increase the quality of the VQ.
            h = 0.1  # point in the mesh [x_min, x_max]x[y_min, y_max].

            # Plot the decision boundary. For that, we will assign a color to each
            x_min, x_max = raw[:, 0].min(), raw[:, 0].max()
            y_min, y_max = raw[:, 1].min(), raw[:, 1].max()
            xx, yy = np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)

            # Points to draw the line between:
            y_slope = []
            x_slope = []
            # Calculate y for x_min:
            Y_xmin = inverse_slope * x_min + b
            if Y_xmin >= y_max:
                y = y_max
                x = (y_max - b) / inverse_slope
                y_slope.append(y)
                x_slope.append(x)
            if Y_xmin <= y_min:
                y = y_min
                x = (y_min - b) / inverse_slope
                y_slope.append(y)
                x_slope.append(x)
            if y_min < Y_xmin < y_max:
                y = Y_xmin
                x = x_min
                y_slope.append(y)
                x_slope.append(x)
            # Calculate y for x_max:
            Y_xmax = inverse_slope * x_max + b
            if Y_xmax >= y_max:
                y = y_max
                x = (y_max - b) / inverse_slope
                y_slope.append(y)
                x_slope.append(x)
            if Y_xmax <= y_min:
                y = y_min
                x = (y_min - b) / inverse_slope
                y_slope.append(y)
                x_slope.append(x)
            if y_min < Y_xmax < y_max:
                y = Y_xmax
                x = x_min
                y_slope.append(y)
                x_slope.append(x)

            assert len(x_slope) == 2, 'There has been a problem calculating the slope of the kmean decision matrix'
            assert len(y_slope) == 2, 'There has been a problem calculating the slope of the kmean decision matrix'

            cluster_array = []
            for i in xx:
                for j in yy:
                    cluster_array.append([i, j])
            cluster_array = np.array(cluster_array)
            cluster_labels = kmeans.predict(cluster_array)

            verif_dim = str(raw)
            if '3d' in verif_dim:
                fig_3d = px.scatter_3d(
                    raw, x=0, y=1, z=2,
                    color=self.labelList, labels=self.labelList)
                fig_3d.add_trace(go.Scatter3d(mode='markers',
                    x=centers[:, 0], y=centers[:, 1], z=centers[:, 2], marker=dict(
                    color='black', size=20, opacity=0.5), name='Cluster center'))
                fig_3d.show()

            else:
                fig_2d = px.scatter(
                    raw, x=0, y=1,
                    color=self.labelList, labels=self.labelList, title=title)
                fig_2d.add_trace(go.Scatter(mode='markers',
                    x=centers[:, 0], y=centers[:, 1], marker=dict(
                    color='black', size=20, opacity=0.5), name='Cluster center'))
                fig_2d.add_trace(go.Scatter(mode='markers',
                    x=cluster_array[:, 0], y=cluster_array[:, 1], marker=dict(
                    color=cluster_labels, opacity=0.1), name='kmean clusters colored'))
                fig_2d.add_trace(go.Scatter(mode='lines',
                    x=x_slope, y=y_slope, marker=dict(color='black'), name='kmean separator'))

                fig_2d.show()

    def getMeanSTD(self):
        sums = []
        for spectrum in self.spectra:
            sums.append(np.sum(spectrum.counts))

        means = np.mean(sums)
        STD = np.std(sums)
        return means, STD

    def butterworthFilter(self, cutoff_frequency=5, order=2):
        for spectrum in self.spectra:
            spectrum.butterworthFilter(cutoff_frequency=cutoff_frequency, order=order)
        self._loadData()

    @staticmethod
    def createLabelPermutations(labels):
        unique_labels = np.unique(labels)
        permutations = list(itertools.permutations(unique_labels))
        new_permutations = []
        for perm in permutations:
            new_perm = [perm[np.where(unique_labels == label)[0][0]] for label in labels]
            new_permutations.append(new_perm)
        return np.array(new_permutations)

    def cluster(self, type):
        if type == "raw":
            # Translate string labels to int labels
            unique_labels = np.unique(self.labelList)
            label_to_int = {label: i for i, label in enumerate(unique_labels, start=1)}
            labels = np.array([label_to_int[label] for label in self.labelList])


            kmeans = KMeans(n_clusters=len(np.unique(labels)))

            # Fit the model to the spectra data
            kmeans.fit(self.data)

            # Make predictions for each spectrum
            predictions = kmeans.predict(self.data)

            # Calculate the accuracy of the predictions
            best_accuracy = 0
            for prediction in self.createLabelPermutations(predictions):
                accuracy = accuracy_score(labels, prediction)
                print(accuracy)
                if accuracy >= best_accuracy:
                    best_accuracy = accuracy
                    print('Best accuracy calculated yet = {0}'.format(best_accuracy))
                    print('The correct labels are: {0}'.format(list(prediction)))
            print('Best accuracy calculated yet = {0}'.format(best_accuracy))
            return best_accuracy

    def getCopy(self):
        return Spectra(self.spectra)

