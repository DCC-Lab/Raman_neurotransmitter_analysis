import numpy as np
import spectrum
import matplotlib.pyplot as plt
import os
import pandas as pd

main = pd.DataFrame({'Params':[],
                     'Total Accuracy':[],
                     'Accuracy per label':[],
                     'Max val list':[],
                     'Max str list':[],
                     'Label str':[]})

#get the dark noise one time so it doesnt compute is everytime
bg = spectrum.Acquisition(
    'dn/').spectraSum()

def best_anal(FDRC_array, pre_array, filter_array, DR_array, cluster_array, normalize=False):
    global main
    # FDRC_array[0]:
    # 0 : smooth --> pre_array : 1d array
    # 1 : savgol --> pre_array : 2d array
    # ----------------------
    # FDRC_array[1]:
    # 0 : ALS --> filter_array : 2d array
    # 1 : BW --> filter_array : 2d array
    # 2 : ORPL --> filter_array : 1d array
    # ----------------------
    # FDRC_array[2]:
    # 0 : PCA --> DR_array : 1d
    # 1 : UMAP --> DR_array : 3d
    # 2 : Pic Ratio --> DR_array : TODO
    # 3 : PCA ext comp --> DR_array : TODO
    # 4 : PCA normalized --> DR array : 1d
    # ----------------------
    # FDRC_array[3]:
    # 0 : KNN --> cluster_array : 1d
    # 1 : R2 --> cluster_array : None
    # 2 : Prob --> cluster_array : None

    p1 = [1]
    p2 = [1]
    f1 = [1]
    f2 = [1]
    DR1 = [1]
    DR2 = [1]
    DR3 = [1]
    C1 = [1]

    if FDRC_array[0] == 0:
        pre = 'smooth({0})'
        p1 = pre_array
    if FDRC_array[0] == 1:
        pre = 'savgol({0}, {1})'
        p1 = pre_array[0]
        p2 = pre_array[1]

    if FDRC_array[1] == 0:
        filter = 'ALS({0}, {1})'
        f1 = filter_array[0]
        f2 = filter_array[1]
    if FDRC_array[1] == 1:
        filter = 'BW({0}, {1})'
        f1 = filter_array[0]
        f2 = filter_array[1]
    if FDRC_array[1] == 2:
        filter = 'ORPL({0})'
        f1 = filter_array

    if FDRC_array[2] == 0:
        DR =  'PCA({0})'
        DR1 = DR_array
    if FDRC_array[2] == 1:
        DR = 'UMAAP({0}, {1}, {2})'
        DR1 = DR_array[0]
        DR2 = DR_array[1]
        DR3 = DR_array[2]
    # if FDRC_array[2] == 2
    #     DR = 'PicRatio(TODO)'
    #     TODO someting like a 2d array and lets we fit array[0][0] with array[1][0], array[0][1] with array[1][1], array[0][2] with array[1][2], etc
    # if FDRC_array[2] == 3:
    #     DR = 'PCAEC(TODO)'
    #     TODO
    if FDRC_array[2] == 4:
        DR = 'PCA({0}, normalize_PCs=True)'
        DR1 = DR_array

    if FDRC_array[3] == 0:
        Cluster =  'KNN({0})'
        C1 = cluster_array
    if FDRC_array[3] == 1:
        Cluster = 'R2 classifier()'
        C1 = cluster_array
    if FDRC_array[3] == 2:
        Cluster = 'Prob classifier()'
        C1 = cluster_array

    for p_1 in p1:
        for p_2 in p2:

            for f_1 in f1:
                for f_2 in f2:

                    for DR_1 in DR1:
                        for DR_2 in DR2:
                            for DR_3 in DR3:

                                for C_1 in C1:
                                    # tot_accuracy, accuracy_per_label, max_val_list, max_str_list, label_str
                                    FP_tot_accuracy, FP_accuracy_per_label, FP_max_val_list, FP_max_str_list, FP_label_str, FP_mat = compute_combo(FDRC_array, [p_1, p_2], [f_1, f_2], [DR_1, DR_2, DR_3], [C_1], 1, normalize=normalize)
                                    NR_tot_accuracy, NR_accuracy_per_label, NR_max_val_list, NR_max_str_list, NR_label_str, NR_mat = compute_combo(FDRC_array, [p_1, p_2], [f_1, f_2], [DR_1, DR_2, DR_3], [C_1], 2, normalize=normalize)
                                    HWN_tot_accuracy, HWN_accuracy_per_label, HWN_max_val_list, HWN_max_str_list, HWN_label_str, HWN_mat = compute_combo(FDRC_array, [p_1, p_2], [f_1, f_2], [DR_1, DR_2, DR_3], [C_1], 3, normalize=normalize)

                                    # exemple de param [smooth(5), ALS(1000, 0.1), PCA(10), KNN(8)]
                                    if normalize == True:
                                        norm = 'Normalized'
                                    if normalize == False:
                                        norm = 'Not Normalized'
                                    FP_df = pd.DataFrame(pd.DataFrame({'Params': [[pre.format(p_1, p_2),
                                                                                  filter.format(f_1, f_2),
                                                                                  DR.format(DR_1, DR_2, DR_3),
                                                                                  Cluster.format(C_1),
                                                                                  norm,
                                                                                  'FingerPrint']],
                                                                         'Total_Accuracy':[FP_tot_accuracy],
                                                                         'Accuracy_per_label':[FP_accuracy_per_label],
                                                                         'Max_val_list':[FP_max_val_list],
                                                                         'Max_str_list':[FP_max_str_list],
                                                                         'Label_str':[FP_label_str],
                                                                         'Matrice':[FP_mat]}))

                                    NR_df = pd.DataFrame(pd.DataFrame({'Params': [[pre.format(p_1, p_2),
                                                                                  filter.format(f_1, f_2),
                                                                                  DR.format(DR_1, DR_2, DR_3),
                                                                                  Cluster.format(C_1),
                                                                                  norm,
                                                                                  'No Raman Region']],
                                                                       'Total_Accuracy': [NR_tot_accuracy],
                                                                       'Accuracy_per_label': [NR_accuracy_per_label],
                                                                       'Max_val_list': [NR_max_val_list],
                                                                       'Max_str_list': [NR_max_str_list],
                                                                       'Label_str': [NR_label_str],
                                                                       'Matrice':[NR_mat]}))

                                    HWN_df = pd.DataFrame(pd.DataFrame({'Params': [[pre.format(p_1, p_2),
                                                                                  filter.format(f_1, f_2),
                                                                                  DR.format(DR_1, DR_2, DR_3),
                                                                                  Cluster.format(C_1), norm,
                                                                                  'High Wavenumbers']],
                                                                       'Total_Accuracy': [HWN_tot_accuracy],
                                                                       'Accuracy_per_label': [HWN_accuracy_per_label],
                                                                       'Max_val_list': [HWN_max_val_list],
                                                                       'Max_str_list': [HWN_max_str_list],
                                                                       'Label_str': [HWN_label_str],
                                                                       'Matrice':[HWN_mat]}))

                                    current_df = pd.concat([FP_df, NR_df, HWN_df], ignore_index=True)
                                    main = pd.concat([main, current_df], ignore_index=True)
                                    # print(main)
                                    # print(FP_df)
                                    # print(NR_df)
                                    # print(HWN_df)

                                    # print('Normalized data == {0}'.format(normalize))
                                    # print(pre.format(p_1, p_2))
                                    # print(filter.format(f_1, f_2))
                                    # print(DR.format(DR_1, DR_2, DR_3))
                                    # print(Cluster.format(C_1))
                                    # print('Dead Region Accuracy : {0} , Fingerprint Region Accuracy : {1} , HWN Region Accuracy : {2}'.format(round(NR_tot_accuracy, 4), round(FP_tot_accuracy, 4), round(HWN_tot_accuracy, 4)), '\n')


def compute_combo(FDRC_array, p_param, f_param, DR_param, c_param, k, normalize=False):
    # FDRC_array[0]:
    # 0 : smooth --> pre_array : 1d array
    # 1 : savgol --> pre_array : 2d array
    # ----------------------
    # FDRC_array[1]:
    # 0 : ALS --> filter_array : 2d array
    # 1 : BW --> filter_array : 2d array
    # 2 : ORPL --> filter_array : 1d array
    # ----------------------
    # FDRC_array[2]:
    # 0 : PCA --> DR_array : 1d
    # 1 : UMAP --> DR_array : 3d
    # 2 : Pic Ratio --> DR_array : TODO
    # 3 : PCA ext comp --> DR_array : TODO
    # 4 : PCA normalized --> DR array : 1d
    # ----------------------
    # FDRC_array[3]:
    # 0 : KNN --> cluster_array : 1d
    # 1 : R2 --> cluster_array : None
    # 2 : Prob --> cluster_array : None

    data = []
    for dir in os.listdir('brain_data/'):
        if dir[0] == '.':
            continue
        data.append(
            spectrum.Acquisition(
                'brain_data/' + dir + '/').spectra())
    data = spectrum.Spectra(data)
    data.removeThermalNoise(bg)
    data.CRRemoval()

    # Pre filter
    if FDRC_array[0] == 0:
        data.smooth(n=p_param[0])

    if FDRC_array[0] == 1:
        data.savgolFilter(window_length=p_param[0], order=p_param[1])

    # Filter
    if FDRC_array[1] == 0:
        data.ALS(lam=f_param[0], p=f_param[1])

    if FDRC_array[1] == 1:
        data.butterworthFilter(cutoff_frequency=f_param[0], order=f_param[1])

    if FDRC_array[1] == 2:
        data.ORPL(min_bubble_widths=f_param[0])

    # Get WN region
    if k == 1:
        data.cut(400, 1800, WN=True)
    if k == 2:
        data.cut(2000, 2600, WN=True)
    if k == 3:
        data.cut(2810, 3020, WN=True)

    if normalize == True:
        data.normalizeIntegration()

    # reduce dim
    if FDRC_array[2] == 0 and FDRC_array[3] == 0:
        data.pca(nbOfComp=DR_param[0])

    if FDRC_array[2] == 1 and FDRC_array[3] == 0:
        data.umap(n_components=DR_param[0], n_neighbors=DR_param[1], min_dist=DR_param[2], display=False)

    if FDRC_array[2] == 2 and FDRC_array[3] == 0:
        data.picRatio()

    if FDRC_array[2] == 3 and FDRC_array[3] == 0:
        data.PCAExternalComposit()

    if FDRC_array[2] == 4 and FDRC_array[3] == 0:
        data.pca(nbOfComp=DR_param[0], normalize_PCs=True)

    # Cluster
    if FDRC_array[3] == 0:
        if FDRC_array[2] == 0 or FDRC_array[2] == 4:
            tot_accuracy, accuracy_per_label, max_val_list, max_str_list, label_str, mat = data.PCA_KNNIndividualLabel(nn=c_param[0], return_details=True, display=False)
        if FDRC_array[2] == 1:
            tot_accuracy, accuracy_per_label, max_val_list, max_str_list, label_str, mat = data.UMAP_KNNIndividualLabel(nn=c_param[0], return_details=True, display=False)
        if FDRC_array[2] == 2:
            tot_accuracy, accuracy_per_label, max_val_list, max_str_list, label_str, mat = data.PR_KNNIndividualLabel(nn=c_param[0], return_details=True, display=False)
        if FDRC_array[2] == 3:
            tot_accuracy, accuracy_per_label, max_val_list, max_str_list, label_str, mat = data.PCAEC_KNNIndividualLabel(nn=c_param[0], return_details=True, display=False)

    if FDRC_array[3] == 1:
        tot_accuracy, accuracy_per_label, max_val_list, max_str_list, label_str, mat = data.R2_classifier(return_details=True, display=False)

    if FDRC_array[3] == 2:
        tot_accuracy, accuracy_per_label, max_val_list, max_str_list, label_str, mat = data.prob_classifier(return_details=True, display=False)

    return tot_accuracy, accuracy_per_label, max_val_list, max_str_list, label_str, mat



# best_anal([0, 0, 0, 0], [5, 7], [[1000, 10000], [0.01, 0.1, 1]], [3, 5, 7], [5, 10])

def iterate_trough_permutations():
    pre = [0, 1]
    filter = [0, 1, 2]
    # DR = [0, 1, 2, 3, 4]
    DR = [0, 1, 4]
    cluster = [0, 1, 2]
    # cluster = [0, 1]
    tracking_number = 0

    for p in pre:
        if p == 0:
            p_array = [3, 7, 11]
            # p_array = [5]
        if p == 1:
            p_array = [[5, 10, 20], [2]]
            # p_array = [[9], [3]]

        for f in filter:
            if f == 0:
                f_array = [[0.1, 10, 1000], [0.00001, 0.0001]]
                # f_array = [[10], [0.001]]
            if f == 1:
                f_array = [[5, 10, 20], [3]]
                # f_array = [[10], [3]]
            if f == 2:
                f_array = [20, 50, 100, 200]
                # f_array = [40]

            for d in DR:
                if d == 0:
                    d_array = [1, 2, 3, 5, 10]
                    # d_array = [5]
                if d == 1:
                    d_array = [[2, 5], [5, 10, 30], [0.05, 0.5]]
                    # d_array = [[4], [15], [0.5]]

                # if d == 2:
                    # TODO pic ratio
                # if d == 3:
                    # TODO pcaec
                if d == 4:
                    d_array = [1, 2, 3, 5, 10]
                    # d_array = [5]


                for c in cluster:
                    if c == 0:
                        c_array = [5, 10, 30]
                        # c_array = [8]
                    if c == 1:
                        c_array = [1]
                    if c == 2:
                        c_array = [1]

                    for norm in [0, 1]:
                        print(str(tracking_number * 100/(len(pre) * len(filter) * len(DR) * len(cluster) * 2)) + ' %')
                        tracking_number += 1

                        if norm == 0:
                            n = True
                        if norm == 1:
                            n = False

                        best_anal([p, f, d, c], pre_array=p_array, filter_array=f_array, DR_array=d_array, cluster_array=c_array, normalize=n)


iterate_trough_permutations()
main.to_csv('Memoire_df.csv', index=False)

# print(main)