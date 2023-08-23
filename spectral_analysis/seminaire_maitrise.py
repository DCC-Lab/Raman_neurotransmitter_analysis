
import spectrum
import os
import numpy as np




def figure_seminaire():
    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/20211001/data/'):
        if dir[0] == '.':
            continue
        data.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/maitrise/DATA/20211001/data/' + dir + '/').spectra())
    data = spectrum.Spectra(data)
    data.displayMeanSTD()

def ashVSobj():
    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/20220921/'):
        if dir[0] == '.':
            continue
        data.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/maitrise/DATA/20220921/' + dir + '/').spectra())
    data = spectrum.Spectra(data)
    data.displayMeanSTD()

def l4():
    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/20211119/toplot/'):
        if dir[0] == '.':
            continue
        data.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/maitrise/DATA/20211119/toplot/' + dir + '/').spectra())
    data = spectrum.Spectra(data)
    data.displayMeanSTD()


def first_monkey():
    bg = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20220802/bg/').spectraSum()


    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/20220802/Monkey_brain/'):
        if dir[0] == '.':
            continue
        data.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/maitrise/DATA/20220802/Monkey_brain/' + dir + '/').spectra())
    data = spectrum.Spectra(data)
    # data = data.combineSpectra(add=60)
    data.removeThermalNoise(bg)
    data.smooth(n=5)
    data.butterworthFilter(cutoff_frequency=8, order=3)
    # data.cut(2700, 3000, WN=True)
    data.cut(450, 1900, WN=True)
    # data.KNNIndividualSpec()
    # data.displayMeanSTD(WN=True)
    # data.pca()
    data.display()
    data.lda(display=True)
    data.ldaScatterPlot(LDx=1)

    # data.pcaScatterPlot(1, 2)
    # data.pcaScatterPlot(3, 4)

    # data.pcaDisplay(1, 2, WN=True)
    # data.pcaDisplay(3, 4, WN=True)

def veal():
    bg = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20220929/morning_verif/darknoise/').spectraSum()
    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/20221028/data_seminaire/'):
        if dir[0] == '.':
            continue
        data.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/maitrise/DATA/20221028/data_seminaire/' + dir + '/').spectra())
    data = spectrum.Spectra(data)
    data.removeThermalNoise(bg)
    data.smooth(n=3)

    # Fingerprint
    data.butterworthFilter(cutoff_frequency=5, order=4)
    data.cut(450, 1900, WN=True)

    # HWN
    # data.butterworthFilter(cutoff_frequency=8, order=4)
    # data.cut(2700, 3000, WN=True)

    data.shortenLabels()
    # data.displayMeanSTD(WN=True)
    data.pca()

    # data.KNNIndividualLabel()

    data.pcaScatterPlot(2, 3)

    # data.pcaScatterPlot(3, 4)

    # data.pcaScatterPlot(5, 6)

    # data.pcaScatterPlot(7, 8)

    # data.pcaDisplay(1, 2, WN=True)
    # data.pcaDisplay(3, 4, WN=True)
    # data.pcaDisplay(5, 6, WN=True)
    # data.pcaDisplay(7, 8, WN=True)

def hanu():
    bg = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20220929/morning_verif/darknoise/').spectraSum()
    data = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/maitrise/DATA/Walter/seminaire/'):
        if dir[0] == '.':
            continue
        data.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/maitrise/DATA/Walter/seminaire/' + dir + '/').spectra())
    data = spectrum.Spectra(data)
    data.removeThermalNoise(bg)
    data.smooth(n=5)

    # HWN
    data.butterworthFilter(cutoff_frequency=8, order=4)
    data.cut(2700, 3000, WN=True)

    # Fingerprint
    # data.butterworthFilter(cutoff_frequency=5, order=3)
    # data.cut(450, 1900, WN=True)

    # data.prob_classifier(plot_mean_std=False)

    # data.shortenLabels()
    # data.displayMeanSTD(WN=True)
    # data.pca()
    # data.umap()
    # data.tsne()

    data.PCA_KNNIndividualLabel(nn=5)

    # data.pcaScatterPlot(1, 2)
    #
    # data.pcaScatterPlot(3, 4)
    #
    # data.pcaScatterPlot(5, 6)
    #
    # data.pcaScatterPlot(7, 8)

    # data.pcaDisplay(1, 2, WN=True)
    # data.pcaDisplay(3, 4, WN=True)
    # data.pcaDisplay(5, 6, WN=True)
    # data.pcaDisplay(7, 8, WN=True)

# figure_seminaire()
# ashVSobj()
# l4()
# first_monkey()
# veal()
hanu()

import scipy.stats as stats

# # Exemple : Distribution normale avec moyenne 0 et écart type 1
# mean = 0
# std_dev = 0.1
#
# # Chiffre dont vous voulez calculer la probabilité
# x = 0.1
#
# # Calcul de la probabilité en utilisant la fonction de densité de probabilité (PDF)
# probability = stats.norm.pdf(x, mean, std_dev)
#
# # Affichage de la probabilité
# print("La probabilité que le chiffre", x, "appartienne à la distribution normale est :", probability)

#
# # Les spectres moyens de chaque échantillon
# average_spectra = [
#     [1.2, 1.3, 1.4, 1.5],  # Exemple : échantillon 1
#     [2.1, 2.2, 2.3, 2.4],  # Exemple : échantillon 2
#     # ... Ajoutez les spectres moyens pour les autres échantillons
# ]
#
# # L'écart type pour chaque longueur d'onde
# std_spectra = [
#     [0.1, 0.1, 0.1, 0.1],  # Exemple : échantillon 1
#     [0.2, 0.2, 0.2, 0.2],  # Exemple : échantillon 2
#     # ... Ajoutez les écarts types pour les autres échantillons
# ]
#
# # Le nouveau spectre à classer
# spectrum = [1.25, 1.35, 1.45, 1.55]
#
# prob_list = []
# for i in range(len(average_spectra)):
#     # Calcul de la prob pour chaque longueur d'onde
#     probs = []
#     for j in range(len(spectrum)):
#         probability = stats.norm.pdf(spectrum[j], average_spectra[i][j], std_spectra[i][j])
#         print(probability)
#         probs.append(probability)
#     avg_prob = np.mean(np.array(probs))
#     prob_list.append(avg_prob)
# print(prob_list)




