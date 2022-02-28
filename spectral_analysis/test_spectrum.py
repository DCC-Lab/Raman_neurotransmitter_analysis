import unittest
import os
import spectrum
import numpy as np

class TestSpectrum(unittest.TestCase):

    def test_listNameOfFiles(self):
        test = spectrum.Spectrum(os.getcwd() + '/unit_test_data/')
        actual_list_of_files = ['test_data_1.txt', 'test_data_2.txt', 'test_data_3.txt', 'test_data_4.txt', 'test_data_5.txt', 'test_data_6.txt']

        self.assertCountEqual(test._listNameOfFiles(), actual_list_of_files)
        self.assertEqual(len(test._listNameOfFiles()), len(actual_list_of_files))

    def test_load(self):
        test = spectrum.Spectrum(os.getcwd() + '/unit_test_data/')
        nb_of_file = 6
        nb_of_pixels = 1044

        self.assertEqual(len(test.x), nb_of_file)
        self.assertEqual(len(test.y), nb_of_file)
        self.assertEqual(len(test.fileName), nb_of_file)

        self.assertEqual(len(test.x[0]), nb_of_pixels)
        self.assertEqual(len(test.x[-1]), nb_of_pixels)
        self.assertEqual(len(test.y[0]), nb_of_pixels)
        self.assertEqual(len(test.y[-1]), nb_of_pixels)

    def test_directoryName(self):
        test = spectrum.Spectrum(os.getcwd() + '/unit_test_data/')

        self.assertEqual('unit_test_data', test.directoryName())

    def test_getSNR(self):
        data_list = [0, 1, 3, 6, 7, 9, 10, 10, 11, 9, 6, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 2, 2, 1, 1]

        calc_SNR, calc_max, calc_rel_max  = spectrum.Spectrum.getSNR(data_list, bgStart=14, bgEnd=22)
        real_SNR, real_max, real_rel_max = 10, 11, 10

        self.assertEqual(calc_SNR, real_SNR)
        self.assertEqual(calc_max, real_max)
        self.assertEqual(calc_rel_max, real_rel_max)

    def test_getWN(self):
        WL_list_1 = [785, 850, 1000, 1100]
        WL_list_2 = [500, 600, 650, 750]

        calc_WN_1 = spectrum.Spectrum.getWN(WL_list_1)
        calc_WN_2 = spectrum.Spectrum.getWN(WL_list_2, lambda_0=500)

        real_WN_1 = []
        real_WN_2 = []
        for i in WL_list_1:
            real_WN_1.append(((10 ** 7) * ((1 / 785) - (1 / i))))
        for i in WL_list_2:
            real_WN_2.append(((10 ** 7) * ((1 / 500) - (1 / i))))

        self.assertEqual(list(calc_WN_1), real_WN_1)
        self.assertEqual(list(calc_WN_2), real_WN_2)

    def test_sumData(self):
        test = spectrum.Spectrum(os.getcwd() + '/unit_test_data/')

        calc_sum = np.zeros(1044)
        for i in test.y:
            calc_sum += i

        self.assertEqual(list(test.sum_y[0]), list(calc_sum))
        self.assertEqual(len(test.sum_y), 1)
        self.assertEqual(len(test.sum_y[0]), 1044)

    def test_substract(self):
        bg = spectrum.Spectrum(os.getcwd() + '/unit_test_data/').y[0]
        test = spectrum.Spectrum(os.getcwd() + '/unit_test_data/')

        test.substract(bg)

        self.assertEqual(list(test.y[0]), list(np.zeros(1044)))
        self.assertNotEqual(list(test.y[1]), list(np.zeros(1044)))
