import spectrum
import numpy as np
import matplotlib.pyplot as plt
import os


def anal_data():
    bg = spectrum.Acquisition(
        '/Users/antoinerousseau/Desktop/maitrise/DATA/20220929/morning_verif/darknoise/').spectraSum()

    P1 = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/discgolf_Raman/position_exp_P1/'):
        if dir[0] == '.':
            continue
        P1.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/discgolf_Raman/position_exp_P1/' + dir + '/').spectra())
    P1 = spectrum.Spectra(P1)

    data = []
    label_array = []
    for dir in os.listdir('/Users/antoinerousseau/Desktop/discgolf_Raman/DATA/'):
        if dir[0] == '.':
            continue
        label_array.append(dir)
        data.append(
            spectrum.Acquisition(
                '/Users/antoinerousseau/Desktop/discgolf_Raman/DATA/' + dir + '/').spectra())
    data = spectrum.Spectra(data)
    data.removeThermalNoise(bg)
    P1.removeThermalNoise(bg)
    P1.changeLabel('P1_exp')
    # P1 = P1.combineSpectra(5)
    # print(label_array)
    brand = []
    plastic = []
    disc = []
    for label in label_array:
        # Brand
        x = label.split('_')
        for i in range(1):
            if x[0] == 'M':
                brand.append('MVP')
            elif x[0] == 'DM':
                brand.append('Discmania')
            elif x[0] == 'I':
                brand.append('Innova')
            elif x[0] == 'DC':
                brand.append('Discraft')
            elif x[0] == 'P':
                brand.append('Prodigy')
            elif x[0] == 'A':
                brand.append('AceLine')
            else:
                brand.append(x[0])
            # Plastic
            plastic.append(x[1])
            # Disc
            disc.append(x[2])

    # data.changeLabel(plastic)
    # data.addAnnotation(disc)
    # data.removelown(15)

    # data.removeLabel('NI')
    # data.removeLabel('S-line')
    # data.removeLabel('neo')

    # data.removeLabel('neo')
    # data.removeLabel('D-line')
    # data.removeLabel('S-line')
    # data.removeLabel('')

    data = data.combineSpectra(5)
    data.changeLabel(plastic)
    data.addAnnotation(disc)
    data.removelown(4)
    data.removeLabel('NI')
    # data.add(P1)
    # Data processing
    data.ORPL(min_bubble_widths=50)
    data.cut(400, 3025, WN=True)
    # data.cut(2700, 3025, WN=True)
    # data.cut(400, 1800, WN=True)
    data.normalizeIntegration()

    # data.display2Colored('neutron', 'C-line')
    # for spec in data.spectra:
    #     spec.display()
    # print(data.spectra)

    # new_label_list = get_list_for_classifier(data.labelList, 5)
    # data.changeLabel(new_label_list)

    # data.display3Colored('C-line', 'S-line', 'D-line')
    # data.displayMeanSTD()
    # print(data.labelList)
    # data.prob_classifier()
    # data.R2_classifier()

    data.pca()
    # data.umap(n_neighbors=3)

    # data.pcaDisplay(4)
    data.pcaScatterPlot(1, 2)
    data.pcaScatterPlot(3, 4)
    data.pcaScatterPlot(5, 6)
    data.pcaScatterPlot(7, 8)
    #
    # data.pcaScatterPlot(1, 2, show_annotations=True)
    # data.pcaScatterPlot(3, 4, show_annotations=True)
    # data.pcaScatterPlot(5, 6, show_annotations=True)
    # data.pcaScatterPlot(7, 8, show_annotations=True)

def get_list_for_classifier(label_list, n):
    unique_array = list(np.unique(np.array(label_list)))
    count_array = list(np.zeros(len(unique_array)))
    new_label_list = []
    for label in label_list:
        index = unique_array.index(label)
        count_array[index] += 1

        for i in range(n):
            new_label_list.append(label + '{0}'.format(int(count_array[index])))

    return new_label_list




anal_data()

# str = '1'
# print(str.isnumeric())

# liste = [2, 4, 5, 4, 4, 3, 5, 6, 1, 3, 5, 3, 4, 3, 3, 3, 4, 5, 6, 2, 4, 3]
# unique_array = np.unique(np.array(liste))
#
# for label in unique_array:
#     x = [i for i, j in enumerate(liste) if j == label]
#     if len(x) < 4:
#         for i in sorted(x, reverse=True):
#             del liste[i]
#
#
# print(liste)
