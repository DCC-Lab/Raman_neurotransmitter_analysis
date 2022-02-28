import os
import fnmatch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit


class Spectrum:

    def __init__(self, directory: str):
        self.directory = directory
        self.x = []
        self.y = []
        self.sum_y = [np.zeros(1044)]
        self.fileName = []

        self._load()
        self.sumData()


    def _listNameOfFiles(self, extension="txt") -> list:
        foundFiles = []

        for file in os.listdir(self.directory):
            if fnmatch.fnmatch(file, f'*.{extension}'):
                foundFiles.append(file)
        return foundFiles


    def _load(self):
        donnees_tot_x = []
        donnees_tot_y = []

        for nom in self._listNameOfFiles():
            # Nom du fichier à importer
            fich = open(self.directory + '/' + nom, "r")
            test_str = list(fich)[14:]
            fich.close()
            x_raw = []
            y_raw = []
            self.fileName.append(nom)
            # Nettoyer les informations
            for j in test_str:
                elem_str = j.replace(",", ".").replace("\n", "").replace("\t", ",")
                elem = elem_str.split(",")
                x_raw.append(float(elem[0]))
                y_raw.append(float(elem[1]))

            donnees_tot_x.append(x_raw)
            donnees_tot_y.append(y_raw)

        self.x, self.y = donnees_tot_x, donnees_tot_y


    def directoryName(self):
        return self.directory.split('/')[-2]

    @staticmethod
    def getSNR(val, bgStart=550, bgEnd=800):
        bg_AVG = 0
        for i in range(bgStart, bgEnd):
            bg_AVG += val[i] / (bgEnd - bgStart)
        return (np.amax(val) - bg_AVG) / np.sqrt(bg_AVG), np.amax(val), (np.amax(val) - bg_AVG)


    @staticmethod
    def getWN(x, lambda_0=785):
        return ((10 ** 7) * ((1 / lambda_0) - (1 / np.array(x))))


    def sumData(self):
        y = np.zeros(1044)
        for i in self.y:
            y += np.array(i)
        self.sum_y = [y]

    def plotSpec(self, sum=False, WN=False, xlabel='Wavelenght [nm]', ylabel='Counts [-]'):
        if sum == True and WN == True:
            xlabel = 'Wavenumber [cm-1]'
            plt.plot(self.getWN(self.x[0]), self.sum_y[0], label=str(len(self.fileName))+' summed files'+', SNR= '+str(self.getSNR(self.sum_y[0])[0])[:5]+', peak = '+str(self.getSNR(self.sum_y[0])[2])[:5])
        if sum == True and WN == False:
            plt.plot(self.x[0], self.sum_y[0], label=str(len(self.fileName))+' summed files'+', SNR= '+str(self.getSNR(self.sum_y[0])[0])[:5]+', peak = '+str(self.getSNR(self.sum_y[0])[2])[:5])
        if sum == False and WN == True:
            xlabel = 'Wavenumber [cm-1]'
            for i in range(len(self.fileName)):
                plt.plot(self.getWN(self.x[i]), self.y[i], label=self.fileName[i]+', SNR= '+str(self.getSNR(self.y[i])[0])[:5]+', peak = '+str(self.getSNR(self.y[i])[2])[:5])
        if sum == False and WN == False:
            for i in range(len(self.fileName)):
                plt.plot(self.x[i], self.y[i], label=self.fileName[i]+', SNR= '+str(self.getSNR(self.y[i])[0])[:5]+', peak = '+str(self.getSNR(self.y[i])[2])[:5])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(self.directoryName())
        plt.legend()
        plt.show()


    @staticmethod
    def customPlot(x, *args, WN = False, lambda_0=785, xlabel='Wavelenght [nm]', ylabel='Counts [-]', title=None):
        first = 0
        if WN == True:
            x = (10 ** 7) * ((1 / lambda_0) - (1 / np.array(x)))
            xlabel = 'Wavenumber [cm-1]'
            for i in args:
                second = 0
                first += 1
                for j in i:
                    second += 1
                    plt.plot(x, j, label=str(first)+'.'+str(second))
        if WN == False:
            for i in args:
                second = 0
                first += 1
                for j in i:
                    second += 1
                    plt.plot(x, j, label=str(first)+'.'+str(second))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.show()


    def substract(self, background, bg_acq_time=600, **kwargs): # kwargs expected: time and factor
        time = 60
        for key, value in kwargs.items():
            if key == 'time':
                time = value
            else:
                pass

        for i in range(len(self.y)):
            self.y[i] = (np.array(self.y[i]) - (time * np.array(background) / bg_acq_time))
            self.y[i] = list(self.y[i])

        self.sumData()


    def factor(self, factor):
        for i in range(len(self.y)):
            self.y[i] = (np.array(self.y[i]) * factor)
            self.y[i] = list(self.y[i])

        self.sumData()


# ------------------------------------------------------------------------------------------------------------------------


