from read_sysu_data import read_file
from scipy.optimize import curve_fit
from scipy.stats import geom
import collections
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from coordinates import get_coords
import copy
import sys
import warnings
warnings.filterwarnings("ignore")

def approximated(x, y):
    popt, _ = curve_fit(func, x, y, maxfev=1e4, bounds=([0, 0.9999], [1e3, 1-1e-9]))    
    t0, c = popt
    return func(x, t0, c), popt

def func(x, t0, c):
    global N0
    return t0 * np.arctanh(x/(N0-c))

def correction_func(b_1, b_2):
    c = 17613 - b_1 - b_2
    q = np.concatenate((np.linspace(2, 1, b_1), np.ones(c), np.linspace(1, 2, b_2)))
    return q

def modify_hittime(hitTimeLPMT_array):
    modified_hitTimeLPMT_array = []
    for i in range(len(hitTimeLPMT_array)):
        x = range(int(hitTimeLPMT_array[i].min().round()), int(hitTimeLPMT_array[i].max().round()))
        for k in x:
            if len((np.where((hitTimeLPMT_array[i] < k+1) & (hitTimeLPMT_array[i] > k)))[0]) > 15:
                break
        modif_hitTime = hitTimeLPMT_array[i][hitTimeLPMT_array[i] > k] - k
        modified_hitTimeLPMT_array.append(modif_hitTime)
    return np.array(modified_hitTimeLPMT_array)

b1 = int(sys.argv[1])
b2 = int(sys.argv[2])
csv_name = '/mnt/cephfs/ml_data/sysu_data/ProcessedData/train_extra/500/1M/trigger_sim/ProcessedTrain{}_{}'.format(sys.argv[3], sys.argv[4])

paths = ['/mnt/cephfs/ml_data/sysu_data/samples/train/eplus_ekin_0_10MeV/{}/root_data/'.format(sys.argv[3])]
csv_names = [csv_name]

bounds = [500]# 175, 250, 350, 500, 750, 1500]

c_opt = 20

thr_array = [1 + i for i in range(9)] + [5 * (i + 1) for i in range(1, 18)] + [91 + i for i in range(9)]
thr_array = np.array(thr_array)

for bound in bounds:
    for i in range(len(paths)):
        dir_path = paths[i]
        csv_name = csv_names[i]

        files = sorted(os.listdir(dir_path), key=len)[b1:b2]
        data_list = []
        for k in range(len(files)):

            train_df = read_file(dir_path+files[k])
            train_df = train_df.reset_index()
            train_df = train_df[train_df.edepX**2 + train_df.edepY**2 + train_df.edepZ**2 < 17200**2]
            evtIDs = train_df.entry.unique()
            N = len(evtIDs)
            #---------------------------------------------------------------------------------------------------------------

            Edep_array = np.array([train_df[train_df.entry==i].edep.iloc[0] for i in evtIDs])
            initEkin_array = np.array([train_df[train_df.entry==i].initEkin.iloc[0] for i in evtIDs])
            hitTimeLPMT_array = np.array([np.array(sorted(train_df[train_df['entry']==i]['hittime'])) for i in evtIDs])
            hitTime = np.array([np.array(train_df[train_df['entry']==i]['hittime']) for i in evtIDs])
            hitTime = np.array([hitTime[i] - hitTimeLPMT_array[i][0] for i in range(N)])
            hitTimeLPMT_array = np.array([hitTimeLPMT_array[i] - hitTimeLPMT_array[i][0] for i in range(N)])
            #hitTimeLPMT_array = modify_hittime(hitTimeLPMT_array)

            for i in range(hitTimeLPMT_array.shape[0]):
                hitTimeLPMT_array[i] = hitTimeLPMT_array[i][hitTimeLPMT_array[i] < bound]

            lpmt_s_array = np.array([train_df[train_df.entry==i].npe for i in evtIDs])

            for i in range(hitTimeLPMT_array.shape[0]):
                lpmt_s_array[i] = lpmt_s_array[i][hitTime[i] < bound]

            pmtIDs_array = np.array([train_df[train_df.entry==i].pmtID for i in evtIDs])

            for i in range(hitTimeLPMT_array.shape[0]):
                pmtIDs_array[i] = pmtIDs_array[i][hitTime[i] < bound]

            pmts_max_npe = np.array([
                train_df[(train_df['entry'] == i)][hitTime[np.argwhere(evtIDs == i)[0][0]] < bound].sort_values('npe', ascending=False).pmtID.iloc[0]
                for i in evtIDs])

            for i in range(hitTime.shape[0]):
                hitTime[i] = hitTime[i][hitTime[i] < bound]

            npe_max = np.array([lpmt_s_array[i].max() for i in range(N)])
            npe_mean = np.array([lpmt_s_array[i].mean() for i in range(N)])

            ht_mean = [hitTimeLPMT_array[i].mean() for i in range(N)]
            ht_std = [hitTimeLPMT_array[i].std() for i in range(N)]

            #---------------------------------------------------------------------------------------------------------------

            edepX_array = np.array([train_df[train_df.entry==i].edepX.iloc[0]/1000. for i in evtIDs])
            edepY_array = np.array([train_df[train_df.entry==i].edepY.iloc[0]/1000. for i in evtIDs])
            edepZ_array = np.array([train_df[train_df.entry==i].edepZ.iloc[0]/1000. for i in evtIDs])
            edepR_array = (edepX_array ** 2 + edepY_array ** 2 + edepZ_array ** 2) ** 0.5
            edepTheta = np.arctan2((edepX_array ** 2 + edepY_array ** 2) ** 0.5, edepZ_array)
            edepPhi = np.arctan2(edepY_array, edepX_array)

            #---------------------------------------------------------------------------------------------------------------

            x_c_max = []
            y_c_max = []
            z_c_max = []

            x_ht_array = []
            y_ht_array = []
            z_ht_array = []

            x_w_array = []
            y_w_array = []
            z_w_array = []

            allHits = []
            #reweightHist = []
            allpmt = []

            for i in range(len(pmtIDs_array)):
                lpmt_x, lpmt_y, lpmt_z = get_coords.get_lpmt_coords(np.array(pmtIDs_array[i]))

                index = pd.Series(hitTime[i]).index
                x_ht = sum(lpmt_x[index] / (hitTime[i] + c_opt)) / \
                       sum(1 / np.array(hitTime[i] + c_opt))
                y_ht = sum(lpmt_y[index] / np.array(hitTime[i] + c_opt)) / \
                       sum(1 / np.array(hitTime[i] + c_opt))
                z_ht = sum(lpmt_z[index] / np.array(hitTime[i] + c_opt)) / \
                       sum(1 / np.array(hitTime[i] + c_opt))

                x_ht_array.append(x_ht)
                y_ht_array.append(y_ht)
                z_ht_array.append(z_ht)

                lpmt_x, lpmt_y, lpmt_z = get_coords.get_lpmt_coords(np.array(pmts_max_npe[i]))
                x_c_max.append(lpmt_x)
                y_c_max.append(lpmt_y)
                z_c_max.append(lpmt_z)

                lpmt_x, lpmt_y, lpmt_z = get_coords.get_lpmt_coords(np.array(pmtIDs_array[i]))

                npe_rew = copy.deepcopy(lpmt_s_array)
                npe_rew[i] *= correction_func(b1_opt, b2_opt)[np.array(pmtIDs_array[i])]

                x_w = sum(lpmt_x[index] * np.array(npe_rew[i])) / \
                      sum(np.array(npe_rew[i]))

                y_w = sum(lpmt_y[index] * np.array(npe_rew[i])) / \
                      sum(np.array(npe_rew[i]))

                z_w = sum(lpmt_z[index] * np.array(npe_rew[i])) / \
                      sum(np.array(npe_rew[i]))

                allHits.append(sum(lpmt_s_array[i]))
                allpmt.append(len(lpmt_s_array[i]))

                x_w_array.append(x_w)
                y_w_array.append(y_w)
                z_w_array.append(z_w)

            x_w_array = np.array(x_w_array)
            y_w_array = np.array(y_w_array)
            z_w_array = np.array(z_w_array)

            x_ht_array = np.array(x_ht_array)
            y_ht_array = np.array(y_ht_array)
            z_ht_array = np.array(z_ht_array)

            allHits = np.array(allHits)
            allpmt = np.array(allpmt)

            x_c_max = np.array(x_c_max)
            y_c_max = np.array(y_c_max)
            z_c_max = np.array(z_c_max)

            #---------------------------------------------------------------------------------------------------------------

            theta_max = np.arctan2((x_c_max**2 + y_c_max**2)**0.5, z_c_max)
            phi_max = np.arctan2(y_c_max, x_c_max)
            zenith_max = z_c_max / (x_c_max ** 2 + y_c_max ** 2) ** 0.5

            #---------------------------------------------------------------------------------------------------------------

            R_ht = (x_ht_array**2 + y_ht_array**2 + z_ht_array**2)**0.5
            theta_ht = np.arctan2((x_ht_array ** 2 + y_ht_array ** 2) ** 0.5, z_ht_array)
            phi_ht = np.arctan2(y_ht_array, x_ht_array)
            zenith_ht = z_ht_array / (x_ht_array**2 + y_ht_array**2)**0.5
            yenith_ht = y_ht_array / (z_ht_array**2 + x_ht_array**2)**0.5
            xenith_ht = x_ht_array / (z_ht_array**2 + y_ht_array**2)**0.5

            sin_theta_ht = np.sin(theta_ht)
            cos_theta_ht = np.cos(theta_ht)

            sin_phi_ht = np.sin(phi_ht)
            cos_phi_ht = np.cos(phi_ht)

            jacob_ht = R_ht**2 * np.sin(theta_ht)

            #---------------------------------------------------------------------------------------------------------------

            R_w = (x_w_array**2 + y_w_array**2 + z_w_array**2)**0.5
            theta_w = np.arctan2((x_w_array ** 2 + y_w_array ** 2) ** 0.5, z_w_array)
            phi_w = np.arctan2(y_w_array, x_w_array)
            zenith_w = z_w_array/(x_w_array**2 + y_w_array**2)**0.5
            yenith_w = y_w_array/(z_w_array**2 + x_w_array**2)**0.5
            xenith_w = x_w_array/(z_w_array**2 + y_w_array**2)**0.5

            sin_theta_w = np.sin(theta_w)
            cos_theta_w = np.cos(theta_w)

            sin_phi_w = np.sin(phi_w)
            cos_phi_w = np.cos(phi_w)

            jacob_w = R_w**2 * np.sin(theta_w)

            #---------------------------------------------------------------------------------------------------------------

            t0_array = []
            for i in range(len(evtIDs)):
                ht = hitTimeLPMT_array[i]
                N0 = len(hitTimeLPMT_array[i])
                X = np.arange(N0)
                t0 = approximated(X, ht)[1][0]
                t0_array.append(t0)

            #---------------------------------------------------------------------------------------------------------------

            entries1 = []
            entries2 = []
            bins_med = []

            for i in range(len(lpmt_s_array)):
                bins = np.array(collections.Counter(lpmt_s_array[i]).most_common())[:, 0]
                entries = np.array(collections.Counter(lpmt_s_array[i]).most_common())[:, 1] / len(lpmt_s_array[i])

                entries1.append(entries[0])
                entries2.append(entries[1])
                bins_med.append(np.median(bins))

            #---------------------------------------------------------------------------------------------------------------

            ht_ps = []
            for thr in thr_array:
                ht_ps.append([np.percentile(hitTimeLPMT_array[i], thr) for i in range(N)])

            #---------------------------------------------------------------------------------------------------------------

            data = pd.DataFrame({
                                 'x_cht': x_ht_array,
                                 'y_cht': y_ht_array,
                                 'z_cht': z_ht_array,
                                 'gamma_z_cht': zenith_ht,
                                 'gamma_y_cht': yenith_ht,
                                 'gamma_x_cht': xenith_ht,
                                 'R_cht': R_ht,
                                 'theta_cht': theta_ht,
                                 'phi_cht': phi_ht,
                                 'sin_theta_cht': sin_theta_ht,
                                 'cos_theta_cht': cos_theta_ht,
                                 'sin_phi_cht': sin_phi_ht,
                                 'cos_phi_cht': cos_phi_ht,
                                 'jacob_cht': jacob_ht,

                                 'x_cc': x_w_array,
                                 'y_cc': y_w_array,
                                 'z_cc': z_w_array,
                                 'gamma_z_cc': zenith_w,
                                 'gamma_y_cc': yenith_w,
                                 'gamma_x_cc': xenith_w,
                                 'R_cc': R_w,
                                 'theta_cc': theta_w,
                                 'phi_cc': phi_w,
                                 'sin_theta_cc': sin_theta_w,
                                 'cos_theta_cc': cos_theta_w,
                                 'sin_phi_cc': sin_phi_w,
                                 'cos_phi_cc': cos_phi_w,
                                 'jacob_cc': jacob_w,

                                 'x_max': x_c_max,
                                 'y_max': y_c_max,
                                 'z_max': z_c_max,
                                 'theta_max': theta_max,
                                 'phi_max': phi_max,
                                 'zenith_max': zenith_max,

                                 't0': np.array(t0_array),
                                 'ht_mean': ht_mean,
                                 'ht_std' : ht_std,

                                 'ht_1p': ht_ps[0],
                                 'ht_2p': ht_ps[1],
                                 'ht_3p': ht_ps[2],
                                 'ht_4p': ht_ps[3],
                                 'ht_5p': ht_ps[4],
                                 'ht_6p': ht_ps[5],
                                 'ht_7p': ht_ps[6],
                                 'ht_8p': ht_ps[7],
                                 'ht_9p': ht_ps[8],
                                 'ht_10p': ht_ps[9],
                                 'ht_15p': ht_ps[10],
                                 'ht_20p': ht_ps[11],
                                 'ht_25p': ht_ps[12],
                                 'ht_30p': ht_ps[13],
                                 'ht_35p': ht_ps[14],
                                 'ht_40p': ht_ps[15],
                                 'ht_45p': ht_ps[16],
                                 'ht_50p': ht_ps[17],
                                 'ht_55p': ht_ps[18],
                                 'ht_60p': ht_ps[19],
                                 'ht_65p': ht_ps[20],
                                 'ht_70p': ht_ps[21],
                                 'ht_75p': ht_ps[22],
                                 'ht_80p': ht_ps[23],
                                 'ht_85p': ht_ps[24],
                                 'ht_90p': ht_ps[25],
                                 'ht_91p': ht_ps[26],
                                 'ht_92p': ht_ps[27],
                                 'ht_93p': ht_ps[28],
                                 'ht_94p': ht_ps[29],
                                 'ht_95p': ht_ps[30],
                                 'ht_96p': ht_ps[31],
                                 'ht_97p': ht_ps[32],
                                 'ht_98p': ht_ps[33],
                                 'ht_99p': ht_ps[34],

                                 'npe_max': npe_max,
                                 'npe_mean': npe_mean,
                                 'entries1': entries1,
                                 'entries2': entries2,
                                 'bins_median': bins_med,
                                 'nPMTs': allpmt,
                                 'nHits': allHits,

                                 'initEkin': initEkin_array,
                                 'Edep': Edep_array,
                                 'edepX': edepX_array,
                                 'edepY': edepY_array,
                                 'edepZ': edepZ_array,
                                 'edepR': edepR_array,
                                 'edepTheta': edepTheta,
                                 'edepPhi': edepPhi
                                }
            )

            data_list.append(data)

            #allData = pd.concat([data_list[i] for i in range(len(data_list))], ignore_index=True)
            #allData.to_csv(csv_name + '_' + str(bound) + '.csv.gz', index=False, compression='gzip')

        allData = pd.concat([data_list[i] for i in range(len(data_list))], ignore_index=True)
        allData.to_csv(csv_name + '_' + str(bound) + '.csv.gz', index=False, compression='gzip')
