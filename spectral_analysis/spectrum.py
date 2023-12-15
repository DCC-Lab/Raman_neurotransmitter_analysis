import os
import fnmatch
import re
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from scipy import sparse
from scipy.sparse.linalg import spsolve
import seaborn as sns
import random
import scipy.stats as stats
import ORPL
import LabPCA
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)




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
        self.PR = []

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
        plt.clf()
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

    def smooth(self, n=3):
        n_ = int((n - 1) / 2)
        new_counts = []
        for i in range(n_, len(self.counts) - n_):
            val_int = self.counts[i - n_: i + n_ + 1]
            val = np.mean(val_int)
            new_counts.append(val)
        self.counts = new_counts
        self.wavenumbers = self.wavenumbers[n_: -n_]
        self.wavelenghts = self.wavelenghts[n_: -n_]

    def bin(self, n=3):
        n_ = int((n - 1) / 2)
        new_counts = []
        for i in range(n_, len(self.counts) - n_, n):
            val_int = self.counts[i - n_: i + n_ + 1]
            val = np.mean(val_int)
            new_counts.append(val)
        self.counts = new_counts
        self.wavenumbers = self.wavenumbers[n_: -n_: n]
        self.wavelenghts = self.wavelenghts[n_: -n_: n]

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

    def butterworthFilter(self, cutoff_frequency=5, order=2, display=False):
        nyquist_frequency = 0.5 * len(self.counts)
        cutoff_frequency = cutoff_frequency / nyquist_frequency
        b, a = signal.butter(order, cutoff_frequency, btype='low', analog=False)
        filtered_intensities = signal.filtfilt(b, a, self.counts)
        if display == True:
            plt.plot(self.wavenumbers, self.counts, 'b')
            plt.plot(self.wavenumbers, filtered_intensities, '--r')
            plt.title('Cut off freq = {0} , Order = {1}'.format(cutoff_frequency, order))
            plt.show()
        self.counts =  self.counts - filtered_intensities

    def getLocalMin(self):
        # Subtract the minimum value of the spectra from the spectra to center it around zero
        spectra_centered = self.counts - np.min(self.counts)

        # Compute the first and second derivatives of the spectra
        d1 = np.gradient(spectra_centered)
        d2 = np.gradient(d1)

        # Find the indexes of the local minima in the second derivative
        minima_indexes = np.where(d2 > 0)[0]

        # Return the indexes of the local minima in the original spectra
        return minima_indexes

    def ALS(self, lam=10**2, p=0.1, niter=10, display=False):
        L = len(self.counts)
        y = np.array(self.counts)
        D = sparse.csc_matrix(np.diff(np.eye(L), 2))
        w = np.ones(L)
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        # return z

        if display == True:
            plt.plot(self.wavelenghts, self.counts, 'black')
            plt.plot(self.wavelenghts, z, 'r--')
            plt.title('lam = {0} , p = {1}'.format(lam, p))
            plt.show()

        self.counts = self.counts - z

    def picRatio(self, l1, l2, n=3, WN=True):
        # returns the ratio l1 / l2
        assert n % 2 == 1, "'n' must be an odd number"
        if WN == True:
            WN = list(self.wavenumbers)

            ADF_s = lambda list_value: abs(list_value - l1)
            ADF_e = lambda list_value: abs(list_value - l2)

            CV_s = min(WN, key=ADF_s)
            CV_e = min(WN, key=ADF_e)

            WN_l1 = WN.index(CV_s)
            WN_l2 = WN.index(CV_e)

            I_l1 = 0
            I_l2 = 0
            for i in range(n):
                I_l1 += self.counts[int(WN_l1 - ((n - 1) / 2) + i)] / n
                I_l2 += self.counts[int(WN_l2 - ((n - 1) / 2) + i)] / n

            ratio = I_l1 / I_l2
        self.PR.append(ratio)
        return ratio

    def ORPL(self, min_bubble_widths=10, display=False):
        filtered, baseline = ORPL.bubblefill(np.array(self.counts), min_bubble_widths=min_bubble_widths)

        if display == True:
            plt.plot(self.wavenumbers, self.counts, color='black')
            plt.plot(self.wavenumbers, baseline, '--r')
            plt.plot(self.wavenumbers, filtered, color='red')
            plt.show()

        self.counts = filtered

    def CRRemoval(self, width=3, std_factor=15, display=False):
        filtered = ORPL.crfilter_single(np.array(self.counts), width=width, std_factor=std_factor)
        if display == True:
            plt.plot(self.wavenumbers, self.counts)
            plt.plot(self.wavenumbers, filtered)
            plt.show()
        self.counts = filtered


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
        self.picRatioList = []

        self._loadData()

    def _loadData(self):
        self.data = []
        self.labelList = []
        picRatioList = []

        for spectrum in self.spectra:
            spectrum_features = np.array(spectrum.counts)
            spectrum_label = spectrum.label
            self.data.append(spectrum_features)
            self.labelList.append(spectrum_label)
            picRatioList.append(spectrum.PR)
        self.picRatioList = list(np.array(picRatioList).T)

    def removeLabel(self, label):
        label_index_list = []
        for i in range(len(self.labelList)):
            if self.labelList[i] == label:
                label_index_list.append(i)

        spectra = self.spectra
        spectra = np.delete(spectra, label_index_list, 0)
        self.spectra = list(spectra)
        self._loadData()
        if self.annotations is not None:
            self.annotations = np.delete(self.annotations, label_index_list, 0)

    def removelown(self, n):
        # Removes spectrum objects from the spectra object that has less then n spectrum object with same label
        unique_array = np.unique(np.array(self.labelList))

        for label in unique_array:
            x = [i for i, j in enumerate(self.labelList) if j == label]
            if len(x) < n:
                for i in sorted(x, reverse=True):
                    del self.labelList[i]
                    del self.spectra[i]
                    if self.annotations != None:
                        del self.annotations[i]
                        assert len(self.labelList) == len(self.annotations), 'There should be as many annotation as the number of label there is. Now, there is {0} label and {1} annotation'.format(len(self.labelList), len(self.annotations))

        self._loadData()

    def changeLabel(self, new_label):
        if type(new_label) == str:
            for spectrum in self.spectra:
                spectrum.label = new_label
        elif type(new_label) == list or type(new_label) == np.ndarray:
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

        if type(annotation) == str:
            for i in range(len(self.spectra)):
                self.annotations.append(annotation)

        if type(annotation) == list:
            self.annotations = annotation

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

    def displayMeanSTD(self, WN=True, display_label=True):
        labels = self._Unique(self.labelList)
        color_list = ['k', 'red', 'green', 'blue', '#332288', 'orange', '#AA4499', '#88CCEE', 'cyan', '#999933',
                      '#44AA99', '#DDCC77', '#805E2B', 'yellow']
        if len(labels) >= len(color_list):
            color_list = ["#" + ''.join([random.choice('0123456789ABCDEF') for i in range(6)]) for j in range(len(labels))]

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

    def smooth(self, n=3):
        for spectrum in self.spectra:
            spectrum.smooth(n=n)
        self._loadData()

    def ALS(self, lam=10**2, p=0.1, niter=10, display=False):
        for spectrum in self.spectra:
            spectrum.ALS(lam=lam, p=p, niter=niter, display=display)
        self._loadData()

    def fixAberations(self, threshold=3.5, display=False):
        for spectrum in self.spectra:
            spectrum.fixAberations(threshold=threshold, display=display)
        self._loadData()

    def CRRemoval(self, width=3, std_factor=15, display=False):
        for spectrum in self.spectra:
            spectrum.CRRemoval( width=width, std_factor=std_factor, display=display)
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

    def pca(self, nbOfComp=10, SC=False, normalize_PCs=False):
        self._loadData()
        self.normalize_PCs = False
        data = self.data
        if SC == True:
            self._standardizeData()
            data = self.SCData

        if normalize_PCs == True:
            self.normalize_PCs = True

        self.PCA = PCA(n_components=nbOfComp)
        self.PCA.fit(data)

        self.SV = []
        self.EVR = []
        self.PC = []
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

    def pcaDisplay(self, *PCs, WN=True):
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

        if self.normalize_PCs == True:
            normalized_array = np.zeros_like(pca_data)
            num_rows, num_cols = pca_data.shape

            for col in range(num_cols):
                column_data = pca_data[:, col]
                min_value = np.min(column_data)
                max_value = np.max(column_data)

                if abs(min_value) >= abs(max_value):
                    normalized_col = column_data / abs(min_value)

                if abs(min_value) <= abs(max_value):
                    normalized_col = column_data / abs(max_value)

                normalized_array[:, col] = normalized_col

            pca_data = normalized_array

        # print(len(pca_data))
        self.PCAlabels = []
        self.PCAcolumns = []

        for spectrum in self.spectra:
            self.PCAlabels.append(spectrum.label)

        for PC in range(len(self.PC)):
            self.PCAcolumns.append('PC{0}'.format(PC + 1))

        # print(np.shape(pca_data))
        self.pca_df = pd.DataFrame(pca_data, index=self.PCAlabels, columns=self.PCAcolumns)

    def _getUMAPdf(self, normalize_Cs=False):
        self._loadData()
        data = self.data

        umap_data = self.UMAP.fit_transform(data)

        if normalize_Cs == True:
            normalized_array = np.zeros_like(umap_data)
            num_rows, num_cols = umap_data.shape

            for col in range(num_cols):
                column_data = umap_data[:, col]
                min_value = np.min(column_data)
                max_value = np.max(column_data)

                if abs(min_value) >= abs(max_value):
                    normalized_col = column_data / abs(min_value)

                if abs(min_value) <= abs(max_value):
                    normalized_col = column_data / abs(max_value)

                normalized_array[:, col] = normalized_col

            pca_data = normalized_array

        # print(len(pca_data))
        self.UMAPlabels = []
        self.UMAPcolumns = []

        for spectrum in self.spectra:
            self.UMAPlabels.append(spectrum.label)

        # print("df_dataframe first data: ", umap_data[0])
        for C in range(len(umap_data[0])):
            self.UMAPcolumns.append('C{0}'.format(C + 1))

        # print(np.shape(pca_data))
        self.umap_df = pd.DataFrame(umap_data, index=self.UMAPlabels, columns=self.UMAPcolumns)

    def _getTSNEdf(self, n_comp=3, normalize_Cs=False):
        self._loadData()
        data = self.data

        tsne_data = self.TSNE.fit_transform(data)

        if normalize_Cs == True:
            normalized_array = np.zeros_like(tsne_data)
            num_rows, num_cols = tsne_data.shape

            for col in range(num_cols):
                column_data = tsne_data[:, col]
                min_value = np.min(column_data)
                max_value = np.max(column_data)

                if abs(min_value) >= abs(max_value):
                    normalized_col = column_data / abs(min_value)

                if abs(min_value) <= abs(max_value):
                    normalized_col = column_data / abs(max_value)

                normalized_array[:, col] = normalized_col

            pca_data = normalized_array

        # print(len(pca_data))
        self.TSNElabels = []
        self.TSNEcolumns = []

        for spectrum in self.spectra:
            self.TSNElabels.append(spectrum.label)

        for C in range(n_comp):
            self.TSNEcolumns.append('C{0}'.format(C + 1))

        # print(np.shape(pca_data))
        self.tsne_df = pd.DataFrame(tsne_data, index=self.TSNElabels, columns=self.TSNEcolumns)

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

    def lda(self, n_components=1 , WN=True, display=False):
        # I dont really understand the n_component stuff going on
        self._loadData()

        assert len(self.data) == len(self.labelList), "'data' and 'label' arrays must be the same lenght"

        nb_class = len(list(set(self.labelList)))
        # self.LDA = LinearDiscriminantAnalysis(n_components=n_components)
        self.LDA = LinearDiscriminantAnalysis(n_components=nb_class-1)
        self.lda_data = self.LDA.fit_transform(self.data, self.labelList)
        print('New dimension space : ', np.shape(self.lda_data))
        print('Scaling? : ', np.shape(self.LDA.scalings_))
        print('shape coef_ : ', np.shape(self.LDA.coef_))
        print('np.of labels', list(set(self.labelList)))
        print('nb de comp : ', self.LDA.n_components)

#____________________________
        lda = LinearDiscriminantAnalysis()
        lda.fit(self.data, self.labelList)

        # Obtenez les composantes LDA
        composantes_lda = lda.scalings_

        # Visualisation des composantes LDA
        plt.figure(figsize=(8, 6))
        for i in range(composantes_lda.shape[1]):
            plt.arrow(0, 0, composantes_lda[:, i][0], composantes_lda[:, i][1],
                      head_width=0.05, head_length=0.1, fc='blue', ec='blue')
            plt.text(composantes_lda[:, i][0], composantes_lda[:, i][1], f'Composante {i + 1}')
        plt.xlabel('Caractéristique 1')
        plt.ylabel('Caractéristique 2')
        plt.title('Composantes LDA')
        plt.grid(True)
        plt.show()

        # Obtenez les coefficients des vecteurs discriminants
        coefs = lda.coef_

        # Calculez l'importance globale pour chaque caractéristique (somme des coefficients)
        importance = np.sum(np.abs(coefs), axis=0)

        # Triez les caractéristiques par ordre d'importance décroissante
        sorted_indices = np.argsort(importance)[::-1]
        sorted_importance = importance[sorted_indices]

        features = []
        for WN in self.spectra[0].wavenumbers:
            features.append(round(WN))
        features = np.array(features)
        sorted_features = features[sorted_indices]  # Remplacez "features" par votre tableau de noms de caractéristiques

        # Créez la visualisation
        plt.figure(figsize=(8, 6))
        plt.bar(range(len(sorted_importance)), sorted_importance)
        plt.xticks(range(len(sorted_importance)), sorted_features, rotation=90)
        plt.xlabel('Caractéristiques')
        plt.ylabel('Importance')
        plt.title('Importance des caractéristiques dans la séparation des données')
        plt.tight_layout()
        plt.show()
# _______________________
        if display == True:
            if WN ==False:
                for i in range(len(self.LDA.coef_)):
                    plt.plot(self.spectra[0].wavelenghts, abs(self.LDA.coef_[i]), label='LD{0}'.format(i + 1))
                plt.xlabel('Wavelengths [nm]')
            if WN == True:
                for i in range(len(self.LDA.coef_)):
                    plt.plot(self.spectra[0].wavenumbers, abs(self.LDA.coef_[i]), label='LD{0}'.format(i + 1))
                    # plt.plot(self.spectra[0].wavenumbers, self.LDA.scalings_.T[i], label='Scaling{0}'.format(i + 1))
                    # plt.plot(self.spectra[0].wavenumbers, (LDA_comp.coef_[i]), label='LD{0}'.format(i + 1))
                plt.xlabel('Wavenumbers [cm-1]')
            plt.legend()
            plt.show()

    def ldaScatterPlot(self, LDx, LDy=None, LDz=None):
        self._loadData()
        print('shape of self.data : ', np.shape(self.data))
        labels = []
        columns = []
        lda_data_T = self.lda_data.T

        for spectrum in self.spectra:
            labels.append(spectrum.label)

        for LD in range(len(lda_data_T)):
            columns.append('LD{0}'.format(LD + 1))
        print('label shape : ', np.shape(labels))
        print('column shape : ', np.shape(columns))

        print('shape df data: ', np.shape(self.lda_data))
        self.lda_df = pd.DataFrame(self.lda_data, index=labels, columns=columns)


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

        self.UMAP = UMAP(n_components=n_components, n_neighbors=n_neighbors, metric=metric, min_dist=min_dist,
                         init='random', random_state=0)
        self.UMAP_array = self.UMAP.fit_transform(self.data)

        if n_components == 2 and display == True:
            fig_2d = px.scatter(
                self.UMAP_array, x=0, y=1,
                color=self.labelList, labels=self.labelList, title=title)

            fig_2d.show()

        if n_components == 3 and display == True:
            fig_3d = px.scatter_3d(
                self.UMAP_array, x=0, y=1, z=2,
                color=self.labelList, labels=self.labelList, title=title)

            fig_3d.show()

    def tsne(self, n_components=2, perplexity=30, metric='euclidean', title=None, display=True):
        self._loadData()
        if title == None:
            title = 'perplexity = {0}'.format(perplexity)

        self.TSNE = TSNE(n_components=n_components, perplexity=perplexity, metric=metric,
                       init='random', random_state=0)
        self.TSNE_array = self.TSNE.fit_transform(self.data)

        if n_components == 2 and display == True:
            fig_2d = px.scatter(
                self.TSNE_array, x=0, y=1,
                color=self.labelList, labels=self.labelList, title=title)

            fig_2d.show()

        if n_components == 3 and display == True:
            fig_3d = px.scatter_3d(
                self.TSNE_array, x=0, y=1, z=2,
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
        #combine 'add' spectra together that have the same label
        # returns a new spectra object
        spectra_list = []
        for i in range(0, len(self.spectra) - add + 1, add):
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

    def ORPL(self, min_bubble_widths=10, display=False):
        for spectrum in self.spectra:
            spectrum.ORPL(min_bubble_widths=min_bubble_widths, display=display)
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

    @staticmethod
    def _getLabelsAsInts(labelList):
        unique_labels = np.unique(labelList)
        label_to_int = {label: i for i, label in enumerate(unique_labels, start=1)}
        labels = np.array([label_to_int[label] for label in labelList])
        return labels, label_to_int

    def kmeanCluster(self, type):
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

    def PCA_KNNIndividualSpec(self, save=False, return_accuracy=False):
        nb_of_good_pred = 0
        prediction_list = []

        for i in range(len(self.spectra)):
            spectrum_to_classify = self.spectra[0]
            spec_to_class_label = spectrum_to_classify.label

            # Remove the spectrum to test from the training dataset
            del self.spectra[0]
            self._loadData()
            # Create a label list that replace str with ints
            int_label_list, label_dict = self._getLabelsAsInts(self.labelList)
            label_str = list(label_dict.keys())
            label_int = list(label_dict.values())
            spec_to_class_label_as_int = label_str.index(spec_to_class_label)

            # Compute PCA and prepare the data in the PCA 'space' for KNN
            self.pca()
            spectrum_to_classify_in_RD_space = self.PCA.transform([spectrum_to_classify.counts])
            self._getPCAdf()
            data = self.pca_df.to_numpy()

            # Make the KNN prediction for the train data
            neigh = KNeighborsClassifier(n_neighbors=5)
            neigh.fit(data, int_label_list)
            pred = neigh.predict(spectrum_to_classify_in_RD_space)[0]
            prediction_list.append(pred)
            # Compute the accuracy
            if int(pred) == int(spec_to_class_label_as_int) + 1:
                nb_of_good_pred += 1

            # Add back the spectrum removed at the begining
            self.spectra.append(spectrum_to_classify)
            self._loadData()

        # Make the matrix to display nicely the results
        label_dict = self._getLabelsAsInts(self.labelList)[1]
        label_str = list(label_dict.keys())
        label_int = list(label_dict.values())
        prediction_list_str = self.int_to_strings(prediction_list, label_int, label_str)
        print(nb_of_good_pred / len(self.spectra))
        # label_str = ['Caudate', 'GPe', 'GPi', 'STN', 'SN', 'Putamen', 'WM', 'Thalamus']
        mat = confusion_matrix(self.labelList, prediction_list_str, labels=label_str)
        cmn = np.round(mat.astype('float') * 100 / mat.sum(axis=1)[:, np.newaxis])
        sns.heatmap(cmn, annot=True, fmt='.0f', xticklabels=label_str, yticklabels=label_str)
        # sns.heatmap(mat, annot=True, fmt='.0f', xticklabels=label_str, yticklabels=label_str)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        if save == True:
            plt.savefig('/Users/antoinerousseau/Desktop/confusion_matrix.eps')
        plt.show()

        if return_accuracy == True:
            return nb_of_good_pred / len(self.spectra)

    def PR_KNNIndividualSpec(self, save=False, return_accuracy=False):
        nb_of_good_pred = 0
        prediction_list = []

        for i in range(len(self.spectra)):
            spectrum_to_classify = self.spectra[0]
            spec_to_class_label = spectrum_to_classify.label

            # Remove the spectrum to test from the training dataset
            del self.spectra[0]
            self._loadData()

            # Create a label list that replace str with ints
            int_label_list, label_dict = self._getLabelsAsInts(self.labelList)
            label_str = list(label_dict.keys())
            label_int = list(label_dict.values())
            spec_to_class_label_as_int = label_str.index(spec_to_class_label)

            spectrum_to_classify_in_RD_space = spectrum_to_classify.PR
            data = np.array(self.picRatioList).T

            # Make the KNN prediction for the train data
            neigh = KNeighborsClassifier(n_neighbors=5)
            neigh.fit(data, int_label_list)
            pred = neigh.predict([spectrum_to_classify_in_RD_space])[0]
            prediction_list.append(pred)
            # Compute the accuracy
            if int(pred) == int(spec_to_class_label_as_int) + 1:
                nb_of_good_pred += 1

            # Add back the spectrum removed at the begining
            self.spectra.append(spectrum_to_classify)
            self._loadData()

        # Make the matrix to display nicely the results
        label_dict = self._getLabelsAsInts(self.labelList)[1]
        label_str = list(label_dict.keys())
        label_int = list(label_dict.values())
        prediction_list_str = self.int_to_strings(prediction_list, label_int, label_str)
        print(nb_of_good_pred / len(self.spectra))
        # label_str = ['Caudate', 'GPe', 'GPi', 'STN', 'SN', 'Putamen', 'WM', 'Thalamus']
        mat = confusion_matrix(self.labelList, prediction_list_str, labels=label_str)
        cmn = np.round(mat.astype('float') * 100 / mat.sum(axis=1)[:, np.newaxis])
        sns.heatmap(cmn, annot=True, fmt='.0f', xticklabels=label_str, yticklabels=label_str)
        # sns.heatmap(mat, annot=True, fmt='.0f', xticklabels=label_str, yticklabels=label_str)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        if save == True:
            plt.savefig('/Users/antoinerousseau/Desktop/confusion_matrix.eps')
        plt.show()

        if return_accuracy == True:
            return nb_of_good_pred / len(self.spectra)

    def PCA_KNNIndividualLabel(self, save=False, return_accuracy=False, return_details=False, display=True, keep_single_label=False, nn=5, n_comp=10):
        #This method requires that all the acquisitions have different labels.
        nb_of_good_pred = 0
        prediction_list = []
        int_label_list_init, label_dict_init = self._getLabelsAsInts(self.labelList)
        label_str_init = list(label_dict_init.keys())
        label_int_init = list(label_dict_init.values())

        for label in label_str_init:

            test_spectra = []
            label_indexes = [i for i, x in enumerate(self.labelList) if x == label]
            # Get les listes d"index et tout pour la current iteration
            int_label_list, label_dict = self._getLabelsAsInts(self.labelList)
            int_label_list = list(int_label_list)
            label_str = list(label_dict.keys())
            label_int = list(label_dict.values())

            # flip la liste pour que le "pop" fonctionne comme il se doit
            label_indexes.sort(reverse=True)

            # Get the test_spectra list and delete them from the self.spectra_thing
            for i in label_indexes:
                test_spectra.append(self.spectra[i])
                self.spectra.pop(i)
                int_label_list.pop(i)
            self._loadData()

            # Compute PCA and prepare the data in the PCA 'space' for KNN
            self._getPCAdf()
            data = self.pca_df.to_numpy()

            spectra_to_classify_in_PCA_space = []
            for spectrum in test_spectra:
                spectrum_to_classify_in_PCA_space = self.PCA.transform([spectrum.counts])
                spectra_to_classify_in_PCA_space.append(spectrum_to_classify_in_PCA_space)

            # Make the KNN prediction for the train data
            neigh = KNeighborsClassifier(n_neighbors=nn)
            neigh.fit(data, int_label_list)
            k = 0
            for spectrum in spectra_to_classify_in_PCA_space:
                true_label = test_spectra[k].label
                pred = neigh.predict(spectrum)[0]

                for key, value in label_dict.items():
                    if value == pred:
                        pred_str = key
                prediction_list.append(pred_str)

                # Compute the accuracy
                # print('Prediction : ' + str(pred_str) + ', Real : ' + str(true_label))
                if self.enlever_chiffres(str(pred_str)) == self.enlever_chiffres(str(true_label)):
                    nb_of_good_pred += 1

                k += 1

            for spectrum in test_spectra:
                self.spectra.append(spectrum)
            self._loadData()

        # Make the matrix to display nicely the results
        new_label_list = []
        new_pred_list = []
        for i in self.labelList:
            new_label_list.append(self.enlever_chiffres(i))
        for i in prediction_list:
            new_pred_list.append(self.enlever_chiffres(i))

        label_dict = self._getLabelsAsInts(new_label_list)[1]
        label_str = list(label_dict.keys())
        label_int = list(label_dict.values())
        prediction_list_str = new_pred_list
        # print(nb_of_good_pred / len(self.spectra))
        # print(new_label_list)
        # print(prediction_list_str)
        # print(label_str)

        # To overwrite the labels:
        # label_str = ['GPe', 'GPi', 'SN', 'STN', 'Putamen', 'Thalamus']

        plt.clf()
        # print(new_label_list)
        # print(prediction_list_str)
        mat = confusion_matrix(new_label_list, prediction_list_str, labels=label_str)
        cmn = np.round(mat.astype('float') * 100 / mat.sum(axis=1)[:, np.newaxis])
        sns.heatmap(cmn, annot=True, fmt='.0f', xticklabels=label_str, yticklabels=label_str)
        # sns.heatmap(mat, annot=True, fmt='.0f', xticklabels=label_str, yticklabels=label_str)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        if save == True:
            plt.savefig('/Users/antoinerousseau/Desktop/confusion_matrix.eps')
        if display == True:
            plt.show()

        if return_accuracy == True:
            return nb_of_good_pred / len(self.spectra)
        # print(nb_of_good_pred / len(self.spectra))

        if return_details == True:
            # [0] : total accuracy
            # [1] : accuracy per label (diagonale)
            # [2] : Valeurs maxs par rangée
            # [3] : String correspondant à la valeur max de chaque rangée ([2])
            # [4] : label list (as str)
            # [5] : Matrice de confusion
            tot_accuracy = nb_of_good_pred / len(self.spectra)

            accuracy_per_label = []
            max_val_list = []
            max_str_list = []
            for i in range(len(label_str)):
                current_list = list(cmn[i])
                accuracy_per_label.append(current_list[i])

                max_index = current_list.index(max(current_list))
                if max_index == i:
                    current_list[i] = 0
                    max_index = current_list.index(max(current_list))

                max_val = current_list[max_index]
                max_str = label_str[max_index]
                max_val_list.append(max_val)
                max_str_list.append(max_str)
            return tot_accuracy, accuracy_per_label, max_val_list, max_str_list, label_str, mat

    def UMAP_KNNIndividualLabel(self, save=False, return_accuracy=False, return_details=False, keep_single_label=False, nn=5, n_comp=3, display=True):
        #This method requires that all the acquisitions have different labels.
        # TODO change the "pca" term by "umap"
        nb_of_good_pred = 0
        prediction_list = []
        int_label_list_init, label_dict_init = self._getLabelsAsInts(self.labelList)
        label_str_init = list(label_dict_init.keys())
        label_int_init = list(label_dict_init.values())

        for label in label_str_init:

            test_spectra = []
            label_indexes = [i for i, x in enumerate(self.labelList) if x == label]
            # Get les listes d"index et tout pour la current iteration
            int_label_list, label_dict = self._getLabelsAsInts(self.labelList)
            int_label_list = list(int_label_list)
            label_str = list(label_dict.keys())
            label_int = list(label_dict.values())

            # flip la liste pour que le "pop" fonctionne comme il se doit
            label_indexes.sort(reverse=True)

            # Get the test_spectra list and delete them from the self.spectra_thing
            for i in label_indexes:
                test_spectra.append(self.spectra[i])
                self.spectra.pop(i)
                int_label_list.pop(i)
            self._loadData()

            # Compute UMAP and prepare the data in the UMAP 'space' for KNN
            self._getUMAPdf()
            data = self.umap_df.to_numpy()
            # print(np.shape(data))

            spectra_to_classify_in_PCA_space = []
            for spectrum in test_spectra:
                spectrum_to_classify_in_PCA_space = self.UMAP.transform([spectrum.counts])
                spectra_to_classify_in_PCA_space.append(spectrum_to_classify_in_PCA_space)

            # Make the KNN prediction for the train data
            neigh = KNeighborsClassifier(n_neighbors=nn)
            neigh.fit(data, int_label_list)
            k = 0
            for spectrum in spectra_to_classify_in_PCA_space:
                true_label = test_spectra[k].label
                pred = neigh.predict(spectrum)[0]

                for key, value in label_dict.items():
                    if value == pred:
                        pred_str = key
                prediction_list.append(pred_str)

                # Compute the accuracy
                # print('Prediction : ' + str(pred_str) + ', Real : ' + str(true_label))
                if self.enlever_chiffres(str(pred_str)) == self.enlever_chiffres(str(true_label)):
                    nb_of_good_pred += 1

                k += 1

            for spectrum in test_spectra:
                self.spectra.append(spectrum)
            self._loadData()

        # Make the matrix to display nicely the results
        new_label_list = []
        new_pred_list = []
        for i in self.labelList:
            new_label_list.append(self.enlever_chiffres(i))
        for i in prediction_list:
            new_pred_list.append(self.enlever_chiffres(i))

        label_dict = self._getLabelsAsInts(new_label_list)[1]
        label_str = list(label_dict.keys())
        label_int = list(label_dict.values())
        prediction_list_str = new_pred_list
        # print(nb_of_good_pred / len(self.spectra))
        # print(new_label_list)
        # print(prediction_list_str)
        # print(label_str)

        # To overwrite the labels:
        # label_str = ['GPe', 'GPi', 'SN', 'STN', 'Putamen', 'Thalamus']

        plt.clf()
        mat = confusion_matrix(new_label_list, prediction_list_str, labels=label_str)
        cmn = np.round(mat.astype('float') * 100 / mat.sum(axis=1)[:, np.newaxis])
        sns.heatmap(cmn, annot=True, fmt='.0f', xticklabels=label_str, yticklabels=label_str)
        # sns.heatmap(mat, annot=True, fmt='.0f', xticklabels=label_str, yticklabels=label_str)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        if save == True:
            plt.savefig('/Users/antoinerousseau/Desktop/confusion_matrix.eps')
        if display == True:
            plt.show()

        if return_accuracy == True:
            return nb_of_good_pred / len(self.spectra)

        if return_details == True:
            # [0] : total accuracy
            # [1] : accuracy per label (diagonale)
            # [2] : Valeurs maxs par rangée
            # [3] : String correspondant à la valeur max de chaque rangée ([2])
            # [4] : label list (as str)
            # [5] : Matrice de confusion
            tot_accuracy = nb_of_good_pred / len(self.spectra)

            accuracy_per_label = []
            max_val_list = []
            max_str_list = []
            for i in range(len(label_str)):
                current_list = list(cmn[i])
                accuracy_per_label.append(current_list[i])

                max_index = current_list.index(max(current_list))
                if max_index == i:
                    current_list[i] = 0
                    max_index = current_list.index(max(current_list))

                max_val = current_list[max_index]
                max_str = label_str[max_index]
                max_val_list.append(max_val)
                max_str_list.append(max_str)
            return tot_accuracy, accuracy_per_label, max_val_list, max_str_list, label_str, mat

    def PCAEC_KNNIndividualLabel(self, save=False, return_accuracy=False, return_details=False, keep_single_label=False, nn=5, display=False):
        nb_of_good_pred = 0
        prediction_list = []
        true_label_list = []
        # Run PCAEF sur tout les données pour avoir les concentrations des composits pour chaque spectre
        data = self.PCAEC_df
        labels = self.labelList

        int_label_list_init, label_dict_init = self._getLabelsAsInts(self.labelList)
        label_str_init = list(label_dict_init.keys())
        label_int_init = list(label_dict_init.values())

        # print(label_str_init)
        # print(label_int_init)
        # print(labels)
        # print(self.PCAEC_df)

        # enlever les spectres qu'on veut evaluer avec KNN

        for label in label_str_init:
            # Get the data to classify
            data_to_class = data.loc[[label]]

            # Remove the data to classify from the training dataframe
            data = data.drop(label)
            # Get current label list
            current_labels = list(data.index.values)

            # fit KNN sur les training spectra
            neigh = KNeighborsClassifier(n_neighbors=nn)
            neigh.fit(data.values, current_labels)

            # Evaluer KNN sur les test spectra
            data_to_class_as_np = data_to_class.to_numpy()
            for spectrum in data_to_class_as_np:
                pred = neigh.predict([spectrum])[0]

                # print(pred)
                prediction_list.append(pred)
                true_label_list.append(label)

                # Compute the accuracy
                # print('Prediction : ' + str(pred) + ', Real : ' + str(label))
                if self.enlever_chiffres(str(pred)) == self.enlever_chiffres(str(label)):
                    nb_of_good_pred += 1


            # Add back the data classified to the dataframe
            data = data.append(data_to_class)
        # print(data)

        # Make the matrix to display nicely the results
        new_label_list = []
        new_pred_list = []
        for i in true_label_list:
            new_label_list.append(self.enlever_chiffres(i))
        for i in prediction_list:
            new_pred_list.append(self.enlever_chiffres(i))

        label_dict = self._getLabelsAsInts(new_label_list)[1]
        label_str = list(label_dict.keys())
        label_int = list(label_dict.values())
        prediction_list_str = new_pred_list
        # print(nb_of_good_pred / len(self.spectra))
        # print(new_label_list)
        # print(prediction_list_str)
        # print(label_str)

        # To overwrite the labels:
        # label_str = ['GPe', 'GPi', 'SN', 'STN', 'Putamen', 'Thalamus']

        plt.clf()
        mat = confusion_matrix(new_label_list, prediction_list_str, labels=label_str)
        cmn = np.round(mat.astype('float') * 100 / mat.sum(axis=1)[:, np.newaxis])
        sns.heatmap(cmn, annot=True, fmt='.0f', xticklabels=label_str, yticklabels=label_str)
        # sns.heatmap(mat, annot=True, fmt='.0f', xticklabels=label_str, yticklabels=label_str)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        if save == True:
            plt.savefig('/Users/antoinerousseau/Desktop/confusion_matrix.eps')
        if display == True:
            plt.show()

        if return_accuracy == True:
            return nb_of_good_pred / len(self.spectra)

        if return_details == True:
            # [0] : total accuracy
            # [1] : accuracy per label (diagonale)
            # [2] : Valeurs maxs par rangée
            # [3] : String correspondant à la valeur max de chaque rangée ([2])
            # [4] : label list (as str)
            # [5] : Matrice de confusion
            tot_accuracy = nb_of_good_pred / len(self.spectra)

            accuracy_per_label = []
            max_val_list = []
            max_str_list = []
            for i in range(len(label_str)):
                current_list = list(cmn[i])
                accuracy_per_label.append(current_list[i])

                max_index = current_list.index(max(current_list))
                if max_index == i:
                    current_list[i] = 0
                    max_index = current_list.index(max(current_list))

                max_val = current_list[max_index]
                max_str = label_str[max_index]
                max_val_list.append(max_val)
                max_str_list.append(max_str)
            return tot_accuracy, accuracy_per_label, max_val_list, max_str_list, label_str, mat

    def PR_KNNIndividualLabel(self, save=False, return_accuracy=False, return_details=False, keep_single_label=False, nn=5, display=True):
        #This method requires that all the acquisitions have different labels.
        nb_of_good_pred = 0
        prediction_list = []
        int_label_list_init, label_dict_init = self._getLabelsAsInts(self.labelList)
        label_str_init = list(label_dict_init.keys())
        label_int_init = list(label_dict_init.values())
        # print(len(label_str_init))
        # print(label_str_init)


        for label in label_str_init:

            test_spectra = []
            label_indexes = [i for i, x in enumerate(self.labelList) if x == label]

            # Get les liste d"index et tout pour la current iteration
            int_label_list, label_dict = self._getLabelsAsInts(self.labelList)
            int_label_list = list(int_label_list)
            label_str = list(label_dict.keys())
            label_int = list(label_dict.values())
            # print(int_label_list)

            # flip la liste pour que le "pop" fonctionne comme il se doit
            label_indexes.sort(reverse=True)

            # Get the test_spectra list and delete them from the self.spectra_thing
            for i in label_indexes:
                test_spectra.append(self.spectra[i])
                self.spectra.pop(i)
                int_label_list.pop(i)
            self._loadData()

            # prepare the data in the PR 'space' for KNN
            spectra_to_classify_in_PR_space = []
            for spectrum in test_spectra:
                spectrum_to_classify_in_PR_space = spectrum.PR
                spectra_to_classify_in_PR_space.append(spectrum_to_classify_in_PR_space)
            self._getPRdf()
            data = self.PR_df.to_numpy()
            # Make the KNN prediction for the train data
            neigh = KNeighborsClassifier(n_neighbors=nn)
            neigh.fit(data, int_label_list)
            k = 0
            for spectrum in spectra_to_classify_in_PR_space:
                true_label = test_spectra[k].label
                pred = neigh.predict([spectrum])
                # print(pred)
                for key, value in label_dict.items():
                    if value == pred:
                        pred_str = key
                prediction_list.append(pred_str)

                # Compute the accuracy
                # print('Prediction : ' + str(pred_str) + ', Real : ' + str(true_label))
                if self.enlever_chiffres(str(pred_str)) == self.enlever_chiffres(str(true_label)):
                    nb_of_good_pred += 1

                k += 1

            for spectrum in test_spectra:
                self.spectra.append(spectrum)
            self._loadData()

        # Make the matrix to display nicely the results
        new_label_list = []
        new_pred_list = []
        for i in self.labelList:
            new_label_list.append(self.enlever_chiffres(i))
        for i in prediction_list:
            new_pred_list.append(self.enlever_chiffres(i))

        label_dict = self._getLabelsAsInts(new_label_list)[1]
        label_str = list(label_dict.keys())
        label_int = list(label_dict.values())
        prediction_list_str = new_pred_list
        # print(nb_of_good_pred / len(self.spectra))
        # print(new_label_list)
        # print(prediction_list_str)
        # print(label_str)

        # To overwrite the labels:
        # label_str = ['GPe', 'GPi', 'SN', 'STN', 'Putamen', 'Thalamus']

        plt.clf()
        mat = confusion_matrix(new_label_list, prediction_list_str, labels=label_str)
        cmn = np.round(mat.astype('float') * 100 / mat.sum(axis=1)[:, np.newaxis])
        sns.heatmap(cmn, annot=True, fmt='.0f', xticklabels=label_str, yticklabels=label_str)
        # sns.heatmap(mat, annot=True, fmt='.0f', xticklabels=label_str, yticklabels=label_str)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        if save == True:
            plt.savefig('/Users/antoinerousseau/Desktop/confusion_matrix.eps')
        if display == True:
            plt.show()

        if return_accuracy == True:
            return nb_of_good_pred / len(self.spectra)

        if return_details == True:
            # [0] : total accuracy
            # [1] : accuracy per label (diagonale)
            # [2] : Valeurs maxs par rangée
            # [3] : String correspondant à la valeur max de chaque rangée ([2])
            # [4] : label list (as str)
            # [5] : Matrice de confusion
            tot_accuracy = nb_of_good_pred / len(self.spectra)

            accuracy_per_label = []
            max_val_list = []
            max_str_list = []
            for i in range(len(label_str)):
                current_list = list(cmn[i])
                accuracy_per_label.append(current_list[i])

                max_index = current_list.index(max(current_list))
                if max_index == i:
                    current_list[i] = 0
                    max_index = current_list.index(max(current_list))

                max_val = current_list[max_index]
                max_str = label_str[max_index]
                max_val_list.append(max_val)
                max_str_list.append(max_str)
            return tot_accuracy, accuracy_per_label, max_val_list, max_str_list, label_str, mat

    @staticmethod
    def int_to_strings(int_array, item_array, label_array):
        # Create an empty dictionary to store the mapping of integers to labels
        int_to_label = {}

        # Iterate over the item array and the label array
        for i in range(len(item_array)):
            # Map the current item to its corresponding label
            int_to_label[item_array[i]] = label_array[i]

        # Create an empty list to store the new array of labels
        label_list = []

        # Iterate over the integer array
        for i in int_array:
            # Append the corresponding label to the label list
            label_list.append(int_to_label[i])

        # Return the label list
        return label_list

    @staticmethod
    def enlever_chiffres(string):
        for i in range(len(string)):
            if string[i].isnumeric():
            # if not string[i].isalpha():
                return string[:i]
        return string

    def shortenLabels(self):
        for spectrum in self.spectra:
            spectrum.label = self.enlever_chiffres(spectrum.label)
        self._loadData()

    def shuffleSpectraOrder(self):
        random.shuffle(self.spectra)
        self._loadData()

    @staticmethod
    def generate_n_different_numbers(n, l):
        if n > l:
            raise ValueError("n doit être inférieur ou égal à l")
        numbers = set()
        while len(numbers) < n:
            numbers.add(random.randint(0, l - 1))
        return list(numbers)

    def keep(self, n):
        indexes = self.generate_n_different_numbers(n, len(self.spectra))
        new_spectra_list = []
        for i in indexes:
            new_spectra_list.append(self.spectra[i])
        self.spectra = new_spectra_list
        self._loadData()

    def picRatio(self, l1, l2, n=3, WN=True):
        picRatioList = []
        for spectrum in self.spectra:
            picRatioList.append(spectrum.picRatio(l1=l1, l2=l2, n=n, WN=WN))
        self.picRatioList.append(picRatioList)

    def _getPRdf(self):
        PR_data = list(np.array(self.picRatioList).T)
        self.PRlabels = []
        self.PRcolumns = []

        for spectrum in self.spectra:
            self.PRlabels.append(spectrum.label)

        for PR in range(len(self.picRatioList)):
            self.PRcolumns.append('PR{0}'.format(PR + 1))
        self.PR_df = pd.DataFrame(PR_data, index=self.PRlabels, columns=self.PRcolumns)

    def PRScatterPlot(self, PRx, PRy=None, PRz=None, AnnotationToDisplay=None, show_annotations=False):
        self._loadData()
        self._getPRdf()


        if AnnotationToDisplay == None:
            if show_annotations == False:
                if PRy == None and PRz == None:
                    fig = px.scatter(self.PR_df, x='PR{0}'.format(PRx), color=self.PRlabels)

                if PRz == None and PRy != None:
                    fig = px.scatter(self.PR_df, x='PR{0}'.format(PRx), y='PR{0}'.format(PRy), color=self.PRlabels)

                if PRy != None and PRz != None:
                    fig = px.scatter_3d(self.PR_df, x='PR{0}'.format(PRx), y='PR{0}'.format(PRy),
                                        z='PR{0}'.format(PRz), color=self.PRlabels)

            if show_annotations == True:
                if PRy == None and PRz == None:
                    fig = px.scatter(self.PR_df, x='PR{0}'.format(PRx), color=self.PRlabels, text=self.annotations)

                if PRz == None and PRy != None:
                    fig = px.scatter(self.PR_df, x='PR{0}'.format(PRx), y='PR{0}'.format(PRy), color=self.PRlabels,
                                     text=self.annotations)

                if PRy != None and PRz != None:
                    fig = px.scatter_3d(self.PR_df, x='PR{0}'.format(PRx), y='PR{0}'.format(PRy),
                                        z='PR{0}'.format(PRz),
                                        color=self.PRlabels, text=self.annotations)

        if AnnotationToDisplay != None:
            toDisplayList = []
            for annotation in AnnotationToDisplay:
                for i in range(len(self.annotations)):
                    if self.annotations[i] == annotation:
                        toDisplayList.append(i)
            temp_PR_df = self.PR_df.iloc[toDisplayList]
            temp_PRlabels = []
            for i in toDisplayList:
                temp_PRlabels.append(self.PCAlabels[i])

            if show_annotations == False:
                if PRy == None and PRz == None:
                    fig = px.scatter(temp_PR_df, x='PR{0}'.format(PRx), color=temp_PRlabels)

                if PRz == None and PRy != None:
                    fig = px.scatter(temp_PR_df, x='PR{0}'.format(PRx), y='PR{0}'.format(PRy), color=temp_PRlabels)

                if PRy != None and PRz != None:
                    fig = px.scatter_3d(temp_PR_df, x='PR{0}'.format(PRx), y='PR{0}'.format(PRy),
                                        z='PR{0}'.format(PRz), color=temp_PRlabels)

            if show_annotations == True:

                temp_annotations = []
                for i in toDisplayList:
                    temp_annotations.append(self.annotations[i])
                if PRy == None and PRz == None:
                    fig = px.scatter(temp_PR_df, x='PR{0}'.format(PRx), color=temp_PRlabels, text=temp_annotations)
                    print('ok')

                if PRz == None and PRy != None:
                    fig = px.scatter(temp_PR_df, x='PR{0}'.format(PRx), y='PR{0}'.format(PRy), color=temp_PRlabels,
                                     text=temp_annotations)
                    print('ok')

                if PRy != None and PRz != None:
                    fig = px.scatter_3d(temp_PR_df, x='PR{0}'.format(PRx), y='PR{0}'.format(PRy),
                                        z='PR{0}'.format(PRz),
                                        color=temp_PRlabels, text=temp_annotations)
                    print('ok')

        plot(fig)

    def R2_classifier(self, save=False, return_accuracy=False, return_details=False, plot_mean_std=False, display=True, keep_single_label=False):
        #This method requires that all the acquisitions have different labels.
        color_list = ['k', 'red', 'green', 'blue', '#332288', 'orange', '#AA4499', '#88CCEE', 'cyan', '#999933',
                      '#44AA99', '#DDCC77', '#805E2B', 'yellow']


        nb_of_good_pred = 0
        prediction_list = []
        int_label_list_init, label_dict_init = self._getLabelsAsInts(self.labelList)
        label_str_init = list(label_dict_init.keys())
        for label in label_str_init:

            test_spectra = []
            label_indexes = [i for i, x in enumerate(self.labelList) if x == label]

            # Get les listes d"index et tout pour la current iteration
            int_label_list, label_dict = self._getLabelsAsInts(self.labelList)
            int_label_list = list(int_label_list)
            label_str = list(label_dict.keys())
            label_int = list(label_dict.values())

            # flip la liste pour que le "pop" fonctionne comme il se doit
            label_indexes.sort(reverse=True)

            # Get the test_spectra list and delete them from the self.spectra_thing
            for i in label_indexes:
                test_spectra.append(self.spectra[i])
                self.spectra.pop(i)
                int_label_list.pop(i)
            self._loadData()

            #Get the single label list (liste avec 1 fois chaque label qui existe dans la liste)
            shorten_labels = []
            for label in label_str_init:
                short_label = self.enlever_chiffres(label)
                shorten_labels.append(short_label)
            label_list = list(set(shorten_labels))

            # Compute the average spectrum for each label
            average_spectra = []
            std_spectra = []
            for label in label_list:
                current_spectra = []
                for spectrum in self.spectra:
                    if self.enlever_chiffres(spectrum.label) == label:
                        current_spectra.append(spectrum.counts)
                    else:
                        continue
                current_spectra = np.array(current_spectra)
                average = np.mean(current_spectra, axis=0)
                std = np.std(current_spectra, axis=0)
                average_spectra.append(average)
                std_spectra.append(std)

            k = 0
            for spectrum in test_spectra:
                r2_scores = []
                for i in range(len(average_spectra)):
                    r2_score = sklearn.metrics.r2_score(average_spectra[i], spectrum.counts)
                    r2_scores.append(r2_score)

                max_r2 = max(r2_scores)
                max_index = r2_scores.index(max_r2)

                pred_str = label_list[max_index]
                prediction_list.append(pred_str)

                true_label = test_spectra[k].label
                # Compute the accuracy
                # print('Prediction : ' + str(pred_str) + ', Real : ' + str(true_label))
                if self.enlever_chiffres(str(pred_str)) == self.enlever_chiffres(str(true_label)):
                    nb_of_good_pred += 1

                k += 1

                #Plot data
                if plot_mean_std == True:
                    plt.plot(spectrum.wavenumbers, spectrum.counts, color=color_list[0])
                    for i in range(len(average_spectra)):
                        plt.plot(spectrum.wavenumbers, average_spectra[i], color=color_list[i + 1])
                        plt.fill_between(spectrum.wavenumbers, average_spectra[i] - std_spectra[i],
                                         average_spectra[i] + std_spectra[i],
                                         color=color_list[i + 1], alpha=0.5)
                    plt.xlabel("Wavenumber [cm-1]")
                    plt.ylabel("Counts [-]")
                    current_color = 0
                    for i in range(len(label_list)):
                        plt.plot([], [], label=label_list[i] + " : R^2 = {0}".format(r2_scores[i]), color=color_list[i + 1])
                        current_color += 1
                    plt.plot([], [], label="Current Spectrum", color=color_list[0])
                    plt.legend()
                    plt.show()


            for spectrum in test_spectra:
                self.spectra.append(spectrum)
            self._loadData()

        # Make the matrix to display nicely the results
        new_label_list = []
        new_pred_list = []
        for i in self.labelList:
            new_label_list.append(self.enlever_chiffres(i))
        for i in prediction_list:
            new_pred_list.append(self.enlever_chiffres(i))

        label_dict = self._getLabelsAsInts(new_label_list)[1]
        label_str = list(label_dict.keys())
        label_int = list(label_dict.values())
        prediction_list_str = new_pred_list
        # print(nb_of_good_pred / len(self.spectra))
        # print(new_label_list)
        # print(prediction_list_str)
        # print(label_str)

        # To overwrite the labels:
        # label_str = ['GPe', 'GPi', 'SN', 'STN', 'Putamen', 'Thalamus']
        plt.clf()

        mat = confusion_matrix(new_label_list, prediction_list_str, labels=label_str)
        # cmn = np.round(mat.astype('float') * 100 / mat.sum(axis=1)[:, np.newaxis])
        # sns.heatmap(cmn, annot=True, fmt='.0f', xticklabels=label_str, yticklabels=label_str)
        sns.heatmap(mat, annot=True, fmt='.0f', xticklabels=label_str, yticklabels=label_str)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        if save == True:
            plt.savefig('/Users/antoinerousseau/Desktop/confusion_matrix.eps')
        if display == True:
            plt.show()

        if return_accuracy == True:
            return nb_of_good_pred / len(self.spectra)

        if return_details == True:
            # [0] : total accuracy
            # [1] : accuracy per label (diagonale)
            # [2] : Valeurs maxs par rangée
            # [3] : String correspondant à la valeur max de chaque rangée ([2])
            # [4] : label list (as str)
            # [5] : matrice de confusion
            tot_accuracy = nb_of_good_pred / len(self.spectra)

            accuracy_per_label = []
            max_val_list = []
            max_str_list = []
            for i in range(len(label_str)):
                current_list = list(cmn[i])
                accuracy_per_label.append(current_list[i])

                max_index = current_list.index(max(current_list))
                if max_index == i:
                    current_list[i] = 0
                    max_index = current_list.index(max(current_list))

                max_val = current_list[max_index]
                max_str = label_str[max_index]
                max_val_list.append(max_val)
                max_str_list.append(max_str)
            return tot_accuracy, accuracy_per_label, max_val_list, max_str_list, label_str, mat

    def prob_classifier(self, save=False, return_accuracy=False, return_details=False, plot_mean_std=False, display=True, keep_single_label=False):
        # This method requires that all the acquisitions have different labels.
        color_list = ['k', 'red', 'green', 'blue', '#332288', 'orange', '#AA4499', '#88CCEE', 'cyan', '#999933',
                      '#44AA99', '#DDCC77', '#805E2B', 'yellow']

        nb_of_good_pred = 0
        prediction_list = []
        int_label_list_init, label_dict_init = self._getLabelsAsInts(self.labelList)
        label_str_init = list(label_dict_init.keys())
        for label in label_str_init:

            test_spectra = []
            label_indexes = [i for i, x in enumerate(self.labelList) if x == label]

            # Get les listes d"index et tout pour la current iteration
            int_label_list, label_dict = self._getLabelsAsInts(self.labelList)
            int_label_list = list(int_label_list)
            label_str = list(label_dict.keys())
            label_int = list(label_dict.values())

            # flip la liste pour que le "pop" fonctionne comme il se doit
            label_indexes.sort(reverse=True)

            # Get the test_spectra list and delete them from the self.spectra_thing
            for i in label_indexes:
                test_spectra.append(self.spectra[i])
                self.spectra.pop(i)
                int_label_list.pop(i)
            self._loadData()

            # Get the single label list (liste avec 1 fois chaque label qui existe dans la liste)
            shorten_labels = []
            for label in label_str_init:
                short_label = self.enlever_chiffres(label)
                shorten_labels.append(short_label)
            label_list = list(set(shorten_labels))

            # Compute the average spectrum for each label
            average_spectra = []
            std_spectra = []
            for label in label_list:
                current_spectra = []
                for spectrum in self.spectra:
                    if self.enlever_chiffres(spectrum.label) == label:
                        current_spectra.append(spectrum.counts)
                current_spectra = np.array(current_spectra)
                average = np.mean(current_spectra, axis=0)
                std = np.std(current_spectra, axis=0)
                average_spectra.append(average)
                std_spectra.append(std)

            k = 0
            for spectrum in test_spectra:

                prob_scores = []
                for i in range(len(average_spectra)):
                    # Calcul de la prob pour chaque longueur d'onde
                    probs = []
                    for j in range(len(spectrum.counts)):
                        probability = stats.norm.pdf(spectrum.counts[j], average_spectra[i][j], std_spectra[i][j])
                        probs.append(probability)
                    avg_prob = np.mean(np.array(probs))
                    prob_scores.append(avg_prob)

                max_prob = max(prob_scores)
                max_index = prob_scores.index(max_prob)

                pred_str = label_list[max_index]
                prediction_list.append(pred_str)

                true_label = test_spectra[k].label
                # Compute the accuracy
                # print('Prediction : ' + str(pred_str) + ', Real : ' + str(true_label))
                if self.enlever_chiffres(str(pred_str)) == self.enlever_chiffres(str(true_label)):
                    nb_of_good_pred += 1

                k += 1

                # Plot data
                if plot_mean_std == True:
                    plt.plot(spectrum.wavenumbers, spectrum.counts, color=color_list[0])
                    for i in range(len(average_spectra)):
                        plt.plot(spectrum.wavenumbers, average_spectra[i], color=color_list[i + 1])
                        plt.fill_between(spectrum.wavenumbers, average_spectra[i] - std_spectra[i],
                                         average_spectra[i] + std_spectra[i],
                                         color=color_list[i + 1], alpha=0.5)
                    plt.xlabel("Wavenumber [cm-1]")
                    plt.ylabel("Counts [-]")
                    current_color = 0
                    for i in range(len(label_list)):
                        plt.plot([], [], label=label_list[i] + " : prob = {0}".format(prob_scores[i]),
                                 color=color_list[i + 1])
                        current_color += 1
                    plt.plot([], [], label="Current Spectrum", color=color_list[0])
                    plt.legend()
                    plt.show()

            for spectrum in test_spectra:
                self.spectra.append(spectrum)
            self._loadData()

        # Make the matrix to display nicely the results
        new_label_list = []
        new_pred_list = []
        for i in self.labelList:
            new_label_list.append(self.enlever_chiffres(i))
        for i in prediction_list:
            new_pred_list.append(self.enlever_chiffres(i))

        label_dict = self._getLabelsAsInts(new_label_list)[1]
        label_str = list(label_dict.keys())
        label_int = list(label_dict.values())
        prediction_list_str = new_pred_list
        # print(nb_of_good_pred / len(self.spectra))
        # print(new_label_list)
        # print(prediction_list_str)
        # print(label_str)

        # To overwrite the labels:
        # label_str = ['GPe', 'GPi', 'SN', 'STN', 'Putamen', 'Thalamus']
        plt.clf()

        mat = confusion_matrix(new_label_list, prediction_list_str, labels=label_str)
        # cmn = np.round(mat.astype('float') * 100 / mat.sum(axis=1)[:, np.newaxis])
        # sns.heatmap(cmn, annot=True, fmt='.0f', xticklabels=label_str, yticklabels=label_str)
        sns.heatmap(mat, annot=True, fmt='.0f', xticklabels=label_str, yticklabels=label_str)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        if save == True:
            plt.savefig('/Users/antoinerousseau/Desktop/confusion_matrix.eps')
        if display == True:
            plt.show()

        if return_accuracy == True:
            return nb_of_good_pred / len(self.spectra)

        if return_details == True:
            # [0] : total accuracy
            # [1] : accuracy per label (diagonale)
            # [2] : Valeurs maxs par rangée
            # [3] : String correspondant à la valeur max de chaque rangée ([2])
            # [4] : label list (as str)
            # [5] : Matrice de confusion
            tot_accuracy = nb_of_good_pred / len(self.spectra)

            accuracy_per_label = []
            max_val_list = []
            max_str_list = []
            for i in range(len(label_str)):
                current_list = list(cmn[i])
                accuracy_per_label.append(current_list[i])

                max_index = current_list.index(max(current_list))
                if max_index == i:
                    current_list[i] = 0
                    max_index = current_list.index(max(current_list))

                max_val = current_list[max_index]
                max_str = label_str[max_index]
                max_val_list.append(max_val)
                max_str_list.append(max_str)
            return tot_accuracy, accuracy_per_label, max_val_list, max_str_list, label_str, mat

    def PCAExternalComposit(self, composit=None, labels=None, nb_of_comp=10, print_con=False, display=False):
        # TODO make sure that all composit have the same len
        self._loadData()
        if composit != None:
            self.composit = composit
        if labels != None:
            self.PCA_composit_labels = labels

        assert len(self.composit[0]) == len(self.spectra[0].wavenumbers), 'Make sure that the composit has the same x axis as the current spectra object'

        if display == True:
            for i in range(len(self.composit)):
                plt.plot(self.spectra[0].wavenumbers, np.array(self.composit)[i], label=self.PCA_composit_labels[i])
            plt.legend()
            plt.title('Composits')
            plt.show()
        # dataSet_ij is now a simulated dataset of 100 spectra coming from 5 analytes mixed in various concentrations
        # basisSet_bj is their individual spectra

        pca = LabPCA.LabPCA(n_components=nb_of_comp)
        pca.fit(self.data)


        # Look at non-centered components
        if display == True:
            plt.plot(pca.components_noncentered_[:5].T)
            plt.legend(['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])
            plt.title("Principal components (non-centered) 1 to 5")
            plt.show()
            plt.plot(pca.components_noncentered_[5:10].T)
            plt.legend(['PC6', 'PC7', 'PC8', 'PC9', 'PC10'])
            plt.title("Principal components (non-centered) 6 to 10")
            plt.show()


        # To avoid confusion, indices (i,b,j,k,p) represent:
        # i = sample #
        # b = basis #
        # j = feature #
        # k = concentration #
        # p = principal coefficient #
        b_bp = pca.transform_noncentered(np.array(self.composit))
        s_ip = pca.transform_noncentered(np.array(self.data))
        s_pi = s_ip.T
        invb_pb = np.linalg.pinv(b_bp)
        invb_bp = invb_pb.T

        recoveredConcentrations_ki = (invb_bp@s_pi).T
        # print(len(recoveredConcentrations_ki))
        # print(len(self.labelList))
        if print_con == True:
            print(self.PCA_composit_labels)
            for i in range(len(recoveredConcentrations_ki)):
                print(self.labelList[i] + ' : ', (recoveredConcentrations_ki[i]))

        # PCAEC : PCA Extracted Concentrations
        self.PCAEC = recoveredConcentrations_ki
        self.PCAEC_df = pd.DataFrame(self.PCAEC, index=self.labelList, columns=self.PCA_composit_labels)

    def pcaecScatterPlot(self, Cx, Cy=None, Cz=None, shorten_labels=False):
        self._loadData()

        if shorten_labels == False:
            if Cy == None and Cz == None:
                fig = px.scatter(self.PCAEC_df, x=Cx, color=self.labelList)

            if Cz == None and Cy != None:
                fig = px.scatter(self.PCAEC_df, x=Cx, y=Cy, color=self.labelList)

            if Cy != None and Cz != None:
                fig = px.scatter_3d(self.PCAEC_df, x=Cx, y=Cy,
                                    z=Cz, color=self.labelList)

        if shorten_labels == True:
            str_label_list = []
            for label in self.labelList:
                str_label_list.append(self.enlever_chiffres(label))

            if Cy == None and Cz == None:
                fig = px.scatter(self.PCAEC_df, x=Cx, color=str_label_list)

            if Cz == None and Cy != None:
                fig = px.scatter(self.PCAEC_df, x=Cx, y=Cy, color=str_label_list)

            if Cy != None and Cz != None:
                fig = px.scatter_3d(self.PCAEC_df, x=Cx, y=Cy,
                                    z=Cz, color=str_label_list)

        plot(fig)

    def tile(self, x, y, WN_to_display, n=3, WN=True):
        # TODO test si je cut avant de tile si ca change de WN_index
        assert x * y == len(self.spectra), 'The shape given (x * y) does not fit the amount of spectra in that Spectra object'

        if WN == True:
            WN = list(self.spectra[0].wavenumbers)

            ADF = lambda list_value: abs(list_value - WN_to_display)

            CV = min(WN, key=ADF)

            data_index = WN.index(CV)

        data = []
        for spectrum in self.spectra:
            value = 0
            for i in range(n):
                value += spectrum.counts[int(data_index - ((n - 1) / 2) + i)] / n

            # data.append(spectrum.counts[data_index])
            data.append(value)
        print('RAW : ', data)
        data = np.reshape(data, (y, x))
        print('Reshaped : ', data)

        for i in range(len(data)):
            if (i % 2) == 1:
                data[i] = data[i][::-1]
        print('flipped : ', data)

        plt.imshow(data)
        plt.title('{} cm-1'.format(WN_to_display))
        plt.show()





