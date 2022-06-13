import os
import fnmatch
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
import plotly.express as px
from plotly.offline import plot
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler



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



    def isValid(self, pixelNb = 1044):
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

        assert date.split(': ')[0] == 'Date', "Was expecting the .txt file\'s 3rd lane to be the date of the acquisition"
        assert trig_mode.split(': ')[0] == 'Trigger mode', "Was expecting the .txt file\'s 6th lane to be the trigger mode used for the acquisition"
        assert integration_time.split(': ')[0] == 'Integration Time (sec)', "Was expecting the .txt file\'s 7th lane to be the integration time used (in sec) for the acquisition"
        assert dark_corr.split(': ')[0] == 'Electric dark correction enabled', "Was expecting the .txt file\'s 9th lane to be the electric dark correction state for the acquisition"
        assert nonlin_corr.split(': ')[0] == 'Nonlinearity correction enabled', "Was expecting the .txt file\'s 10th lane to be the nonlinearity correction state for the acquisition"
        assert x_axis_unit.split(': ')[0] == 'XAxis mode', "Was expecting the .txt file\'s 12th lane to be the x axis units for the acquisition"
        assert pixel_nb.split(': ')[0] == 'Number of Pixels in Spectrum', "Was expecting the .txt file\'s 13th lane to be the number of pixels used for the acquisition"

        # Does it have the right len(x), len(y)
        assert len(self.x) == pixelNb, "Was expecting {0} x values, {1} were given".format(str(pixelNb), str(len(self.x)))
        assert len(self.y) == pixelNb, "Was expecting {0} y values, {1} were given".format(str(pixelNb), str(len(self.y)))
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



    def isValid(self, pixelNb = 2048):
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

        assert date.split(': ')[0] == 'Date', "Was expecting the .txt file\'s 3rd lane to be the date of the acquisition"
        assert trig_mode.split(': ')[0] == 'Trigger mode', "Was expecting the .txt file\'s 6th lane to be the trigger mode used for the acquisition"
        assert integration_time.split(': ')[0] == 'Integration Time (sec)', "Was expecting the .txt file\'s 7th lane to be the integration time used (in sec) for the acquisition"
        assert dark_corr.split(': ')[0] == 'Electric dark correction enabled', "Was expecting the .txt file\'s 9th lane to be the electric dark correction state for the acquisition"
        assert x_axis_unit.split(': ')[0] == 'XAxis mode', "Was expecting the .txt file\'s 12th lane to be the x axis units for the acquisition"
        assert pixel_nb.split(': ')[0] == 'Number of Pixels in Spectrum', "Was expecting the .txt file\'s 13th lane to be the number of pixels used for the acquisition"

        # Does it have the right len(x), len(y)
        assert len(self.x) == pixelNb, "Was expecting {0} x values, {1} were given".format(str(pixelNb), str(len(self.x)))
        assert len(self.y) == pixelNb, "Was expecting {0} y values, {1} were given".format(str(pixelNb), str(len(self.y)))
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


    def _listNameOfFiles(self) -> list:
        foundFiles = []
        for file in os.listdir(self.directory):
            if file[0] == '.':
                continue
            if file == 'README.txt':
                #should do stuff like reading it and taking info from it
                RM = open(self.directory + file, "r")
                self.README = list(RM)
                continue
            if fnmatch.fnmatch(file, f'*{self.extension}'):
                foundFiles.append(file)
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



class ArrayToSpectra:
    def __init__(self, x, array, label=None, integrationTime=None):
        self.x = x
        self.data = array
        self.labelList = label
        self.integrationTimeList = integrationTime
        self.spectra = []

        if self.labelList == None:
            self.labelList = []
            for i in range(len(array)):
                self.labelList.append('No Label Given')

        if self.integrationTimeList == None:
            self.integrationTimeList = []
            for i in range(len(array)):
                self.integrationTimeList.append(1)

        assert len(array) == len(self.labelList) == len(self.integrationTimeList), 'Arguments given must all be the same length'
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
    def __init__(self, wavelenghts, counts, integrationTime, label):
        self.wavelenghts = wavelenghts

        self.counts = counts
        self.integrationTime = integrationTime
        self.label = label

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
            return (np.amax(self.counts) - bg_AVG) / np.sqrt(bg_AVG), np.amax(self.counts), (np.amax(self.counts) - bg_AVG)


    def display(self, WN=True, NoX=False, xlabel='Wavelenght [nm]', ylabel='Counts [-]'):
        if WN == True:
            xlabel = 'Wavenumber [cm-1]'
            # plt.plot(self.wavenumbers, self.counts,  label=self.label + ', SNR= '+str(self.getSNR()[0])[:6] + ', peak = '+str(self.getSNR()[1])[:7] + ', IT: {0} s'.format(self.integrationTime))
            plt.plot(self.wavenumbers, self.counts,  label=self.label)

        if WN == False:
            plt.plot(self.wavelenghts, self.counts, label=self.label + ', SNR= '+str(self.getSNR()[0])[:6] + ', peak = '+str(self.getSNR()[1])[:7] + 'IT: {0} s'.format(self.integrationTime))
        if NoX == True:
            plt.plot(self.counts, label=self.label + ', SNR= '+str(self.getSNR()[0])[:6] + ', peak = ' + str(self.getSNR()[1])[:7] + 'IT: {0} s'.format(self.integrationTime))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()


    def subtract(self, specToSub, specToSub_AcqTime=1): # kwargs expected: time and factor
        # le specToSubAcqTime est à changer pour mettre le bon temps d'acquisition du spectre à soustraire si le spectre a soustraire n'est as de type Spectrum
        acqTime = self.integrationTime

        if type(specToSub) != Spectrum:
            self.counts = list(np.array(self.counts) - (acqTime * np.array(specToSub) / specToSub_AcqTime))

        if type(specToSub) == Spectrum:
            self.counts = list(np.array(self.counts) - (acqTime * np.array(specToSub.counts)) / specToSub.integrationTime)


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


    def addSpectra(self, spectraToAdd):
        # Returns a spectra object containing this spectrum + the spectra or spectrum given as argument
        assert type(spectraToAdd) == Spectra or Spectrum, 'Expecting a Spectra or Spectrum type argument'
        newSpectra = Spectra([Spectrum(self.wavelenghts, self.counts, self.integrationTime, self.label)])

        if type(spectraToAdd) == Spectra:
            for spectrum in spectraToAdd.spectra:
                newSpectra.spectra.append(spectrum)

        if type(spectraToAdd) == Spectrum:
            newSpectra.spectra.append(spectraToAdd)

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


    def polyfit(self, poly_order):
        # should return a spectum object ? i think so....
        # should i raise expection for it to work without any integration time?
        fit_coefs = list(np.polyfit(self.wavenumbers[63:], self.counts[63:], poly_order))
        y_fit = self.polyFunc(self.wavenumbers, fit_coefs)



        plt.plot(self.wavenumbers, self.counts, 'k-', label='Data curve')
        plt.plot(self.wavenumbers[63:], y_fit[63:], 'r--', label='{0}th order polynomial fit'.format(poly_order))
        plt.plot(self.wavenumbers[63:], (np.array(self.counts)-np.array(y_fit))[63:], 'r', label='subtraction')
        plt.legend()
        plt.show()

        label = self.label + '_fit'
        return Spectrum(self.wavenumbers, y_fit, self.integrationTime, label)


    def fftFilter(self):
        # should change self.counts so it
        d = (self.wavenumbers[-1] - self.wavenumbers[0]) / len(self.wavenumbers)
        fhat = np.fft.fft(self.counts, len(self.counts))
        PSD = fhat * np.conj(fhat) / len(self.counts)
        freq = (1 / (d * len(self.counts))) * np.arange(len(self.counts))
        L = np.arange(1, np.floor(len(self.counts)/2), dtype='int')

        PSDtreated = PSD
        val = 515
        #   Highpass
        # PSDtreated[:val] = 0
        # fhat[:val] = 0
        # fhat[-val:] = 0
        #   Lowpass
        PSDtreated[-val:] = 0
        fhat[int(((len(fhat) / 2) - val)): int((len(fhat) / 2))] = 0
        fhat[int((len(fhat) / 2)): int(((len(fhat) / 2) + val))] = 0
        ffilt = np.fft.ifft(fhat)


        plt.plot(freq[L], PSDtreated[L])
        plt.show()

        plt.plot(self.wavenumbers, ffilt, 'r')
        plt.plot(self.wavenumbers, self.counts, 'k')
        plt.show()

        plt.plot(self.wavenumbers, (self.counts - ffilt))
        plt.show()


    def setZero(self, val):
        self.counts[0:val] = 0


    def fixSpec(self):
        x = np.array(self.wavenumbers)
        x = x - 85
        self.wavenumbers = x

        for i in range(len(self.wavenumbers)):
            self.wavenumbers[i] = self.wavenumbers[i] + 10 * ((self.wavenumbers[i]) / 747)


    def smooth(self):
        for i in range(1, len(self.counts) - 1):
            val = (self.counts[i - 1] + self.counts[i] + self.counts[i + 1]) / 3
            self.counts[i] = val



class Spectra:
    def __init__(self, items):
        self.spectra = []
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


    def display(self, WN=True):
        if WN == False:
            for spectrum in self.spectra:
                plt.plot(spectrum.wavelenghts, spectrum.counts, label=spectrum.label + ', integration = ' + str(spectrum.integrationTime) + ' s')

        if WN == True:
            for spectrum in self.spectra:
                plt.plot(spectrum.wavenumbers, spectrum.counts, label=spectrum.label + ', integration = ' + str(spectrum.integrationTime) + ' s')

        plt.legend()
        plt.show()


    def removeThermalNoise(self, TNToRemove):
        for spectrum in self.spectra:
            spectrum.removeThermalNoise(TNToRemove)


    def subtract(self, specToSub, specToSub_AcqTime=None):
        if type(specToSub) != Spectrum:
            for spectrum in self.spectra:
                spectrum.counts = list(np.array(spectrum.counts) - (spectrum.integrationTime * np.array(specToSub) / specToSub_AcqTime))

        if type(specToSub) == Spectrum:
            for spectrum in self.spectra:
                spectrum.counts = list(np.array(spectrum.counts) - (spectrum.integrationTime * np.array(specToSub.counts) / specToSub.integrationTime))


    def normalizeIntegration(self):
        for spectrum in self.spectra:
            spectrum.normalizeIntegration()


    def normalizeCounts(self):
        for spectrum in self.spectra:
            spectrum.normalizeCounts()


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

                after = len(self.spectra)
                assert after == (before + len(
                    item.spectra)), 'The spectra that were supposed to be added to this object have not been properly added'

            if type(item) == Spectrum:
                before = len(self.spectra)
                self.spectra.append(item)

                after = len(self.spectra)
                assert after == before + 1, 'The spectrum that was supposed to be added to this object has not been properly added'



    def pca(self, nbOfComp = 10, SC=False):
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
            self.PC.append(Spectrum(self.spectra[0].wavelenghts, self.PCA.components_[i], 1, 'PC{0}, val propre = {1}'.format(i + 1, self.PCA.explained_variance_ratio_[i])))


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


    def pcaDisplay(self, *PCs):
        WN = False
        if WN == True:
            for PC in PCs:
                plt.plot(self.spectra[0].wavenumbers, self.PC[PC - 1].counts, label=self.PC[PC - 1].label)
        if WN == False:
            for PC in PCs:
                plt.plot(self.spectra[0].wavelenghts, self.PC[PC - 1].counts, label=self.PC[PC - 1].label)
        plt.legend()
        plt.show()


    def _getPCAdf(self, SC=False):
        self._loadData()
        data = self.data

        if SC == True:
            self._standardizeData()
            data = self.SCData

        pca_data = self.PCA.fit_transform(data)
        print(pca_data)
        self.PCAlabels = []
        self.PCAcolumns = []

        for spectrum in self.spectra:
            self.PCAlabels.append(spectrum.label)

        for PC in range(len(self.PC)):
            self.PCAcolumns.append('PC{0}'.format(PC + 1))

        self.pca_df = pd.DataFrame(pca_data, index=self.PCAlabels, columns=self.PCAcolumns)


    def pcaScatterPlot(self, PCx, PCy=None, PCz=None, SC=False):
        self._loadData()
        self._getPCAdf()

        if PCy == None and PCz == None:
            fig = px.scatter(self.pca_df, x='PC{0}'.format(PCx), color=self.PCAlabels)

        if PCz == None and PCy != None:
            fig = px.scatter(self.pca_df, x='PC{0}'.format(PCx), y='PC{0}'.format(PCy), color=self.PCAlabels)

        if PCy != None and PCz != None:
            fig = px.scatter_3d(self.pca_df, x='PC{0}'.format(PCx), y='PC{0}'.format(PCy), z='PC{0}'.format(PCz), color=self.PCAlabels)

        plot(fig)


    def getSTD(self):
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
            ratio = mean/(STD**2)
            ratios.append(ratio)

            print('WL: {3},mean: {0}, STD: {1}, ratio: {2}'.format(mean, STD, ratio, WL))

        plt.plot(self.spectra[0].wavelenghts, ratios)
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


    def fixSpec(self):
        for spectrum in self.spectra:
            spectrum.fixSpec()


    def _standardizeData(self):
        self.SCData = StandardScaler().fit_transform(self.data)


    def lda(self, n_components=2, SC=False):
        self._loadData()

        assert len(self.data) == len(self.labelList), "'data' and 'label' arrays must be the same lenght"

        data = self.data

        if SC == True:
            self._standardizeData()
            assert len(self.SCData) == len(self.labelList), "'SCdata' and 'label' arrays must be the same lenght"
            data = self.SCData

        self.LDA = LinearDiscriminantAnalysis(n_components=n_components)
        self.LDA.fit(data, self.labelList)

        print('coefs: ', self.LDA.coef_)
        print('intercept: ', self.LDA.intercept_)
        # print('covariance: ', self.LDA.covariance_)
        print('explained variance ratio: ', self.LDA.explained_variance_ratio_)
        print('means: ', self.LDA.means_)
        print('priors: ', self.LDA.priors_)
        print('scalings: ', np.shape(self.LDA.scalings_), self.LDA.scalings_)
        print('xbar: ', self.LDA.xbar_)
        print('classes: ', self.LDA.classes_)
        print('features: ', self.LDA.n_features_in_)
        # print('features names: ', self.LDA.feature_names_in_)


        # plt.plot(self.LDA.scalings_.T[0], label='scaling transposed 1')
        # plt.plot(self.LDA.scalings_.T[1], label='scaling transposed 2')
        # plt.plot(self.LDA.scalings_.T[2], label='scaling transposed 3')
        # plt.plot(self.LDA.scalings_.T[3], label='scaling transposed 4')
        # plt.plot(self.LDA.scalings_.T[4], label='scaling transposed 5')
        # plt.legend()
        # plt.title('scaling transposed')
        # plt.show()


        # plt.plot(self.spectra[0].counts)
        # plt.title('spectrum :' + self.spectra[0].label)
        # plt.show()

        # plt.plot(self.LDA.coef_[0], label='coef 1')
        # plt.plot(self.LDA.coef_[1], label='coef 2')
        # plt.plot(self.LDA.coef_[2], label='coef 3')
        # plt.plot(self.LDA.coef_[3], label='coef 4')
        # plt.plot(self.LDA.coef_[4], label='coef 5')
        # plt.plot(self.LDA.coef_[5], label='coef 6')
        # plt.legend()
        # plt.title('LDA coefficients')
        # plt.show()

        # plt.plot(self.LDA.means_[0], label='mean 1')
        # plt.plot(self.LDA.means_[1], label='mean 2')
        # plt.plot(self.LDA.means_[2], label='mean 3')
        # plt.plot(self.LDA.means_[3], label='mean 4')
        # plt.plot(self.LDA.means_[4], label='mean 5')
        # plt.plot(self.LDA.means_[5], label='mean 6')
        # plt.legend()
        # plt.title('LDA means')
        # plt.show()

        # plt.plot(self.LDA.xbar_, label='xbar')
        # plt.title('xbar')
        # plt.show()


    def ldaScatterPlot(self, LDx, LDy=None, LDz=None, SC=False):
        self._loadData()

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
            fig = px.scatter_3d(self.lda_df, x='LD{0}'.format(LDx), y='LD{0}'.format(LDy), z='LD{0}'.format(LDz), color=labels)

        plot(fig)



