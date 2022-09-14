import spectrum
import numpy as np
import pandas as pd

def BarCode(Section, lenght):
    TRUTH = pd.read_csv(Section)[:-3]['Gray_Value'].to_numpy()
    splits = np.array_split(TRUTH, lenght)
    GL_TRUTH = []
    for i in splits:
        GL_TRUTH.append(np.mean(i))
    GL_TRUTH = np.array(GL_TRUTH)
    GL_TRUTH[GL_TRUTH > 245] = 1
    GL_TRUTH[GL_TRUTH > 180] = 2
    GL_TRUTH[GL_TRUTH > 105] = 3
    GL_TRUTH = np.where(GL_TRUTH == 1, 'WHITE', GL_TRUTH)
    GL_TRUTH = np.where(GL_TRUTH == '2.0', 'MIXED', GL_TRUTH)
    GL_TRUTH = np.where(GL_TRUTH == '3.0', 'GREY', GL_TRUTH)

    return GL_TRUTH


def PCADeapoliLiveMonkeyData():
    STNr = spectrum.Acquisition('/Users/antoinerousseau/Desktop/ddpaoli/20161103_InVivoMonkeySurgery/RSTN_labelfixed/', fileType='Depaoli').spectra()
    STNr_label = BarCode('/Users/antoinerousseau/Downloads/2016-DRS-STNright-MonkeyBrain-barcodeGWM.csv', len(STNr.spectra))
    STNr.changeLabel(STNr_label)
    # STNr.removeSpectra(0, 67)
    # STNr.cut(500, 650, WL=True)
    STNr.smooth()
    STNr.cut(350, 800, WL=True)
    STNr.normalizeIntegration()
    STNr.display3Colored(label1='WHITE', label2='GREY', label3="MIXED", WN=False)
    # STNr.polyfit(4)
    STNr.pca()
    STNr.pcaScatterPlot(1, 2)
    STNr.pcaScatterPlot(3, 4)
    STNr.pcaScatterPlot(5, 6)

    STNr.pcaDisplay(1, 2, 3)
    STNr.pcaDisplay(4, 5)



PCADeapoliLiveMonkeyData()
