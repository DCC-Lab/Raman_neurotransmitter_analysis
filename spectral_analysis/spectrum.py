import os
import fnmatch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit


class OceanViewSpectralFile:

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



class Acquisition:
    def __init__(self, directory):
        self.directory = directory
        self.spectralFiles = []
        self.directoryName = directory.split('/')[-2]

        filePaths = self._listNameOfFiles()
        for filepath in filePaths:
            spectralFile = OceanViewSpectralFile(directory + filepath)
            self.spectralFiles.append(spectralFile)


    def _listNameOfFiles(self, extension="txt") -> list:
        foundFiles = []

        for file in os.listdir(self.directory):
            if file[0] == '.':
                continue
            if file == 'README.txt':
                #should do stuff like reading it and taking info from it
                RM = open(self.directory + file, "r")
                self.README = list(RM)
                continue
            if fnmatch.fnmatch(file, f'*.{extension}'):
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



class Spectrum:
    def __init__(self, wavelenghts, counts, integrationTime, label):
        self.wavelenghts = wavelenghts
        self.wavenumbers = ((10 ** 7) * ((1 / 785) - (1 / np.array(self.wavelenghts))))
        self.counts = counts
        self.integrationTime = integrationTime
        self.label = label


    def getSNR(self, bgStart=550, bgEnd=800):
        bg_AVG = 0
        for i in range(bgStart, bgEnd):
            bg_AVG += self.counts[i] / (bgEnd - bgStart)
        return (np.amax(self.counts) - bg_AVG) / np.sqrt(bg_AVG), np.amax(self.counts), (np.amax(self.counts) - bg_AVG)


    def display(self, WN=False, xlabel='Wavelenght [nm]', ylabel='Counts [-]'):
        if WN == True:
            xlabel = 'Wavenumber [cm-1]'
            plt.plot(self.wavenumbers, self.counts, label=self.label + ', SNR= '+str(self.getSNR()[0])+', peak = '+str(self.getSNR()[1]))
        if WN == False:
            plt.plot(self.wavelenghts, self.counts, label=self.label + ', SNR= '+str(self.getSNR()[0])+', peak = '+str(self.getSNR()[1]))
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
            integration += i

        self.counts = list(np.array(self.counts) / integration)


    def normalizeCounts(self):
        self.counts = list(np.array(self.counts) / np.amax(self.counts))


class Spectra:
    def __init__(self, spectra):
        self.spectra = spectra


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


