# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 08:50:52 2021

@author: Yichao

Analysis matrix correlation Balance vs Loco 

"""

import matplotlib.pyplot as plt
import numpy as np
import os
import ntpath
import sys
import pandas as pd 
import seaborn as sns
from sklearn import linear_model
import scipy

# Get the files' dataframe

path_result_posture = 'C:/Users/Utilisateur/Desktop/DocumentsSync/JNJ/AnalysePosture/JnJ_Result_SOT.csv'
path_result_ratio_posture = 'C:/Users/Utilisateur/Desktop/DocumentsSync/JNJ/AnalysePosture/JnJ_Result_Ratio_Posture.csv'
path_result_mean_contrast_vision = 'C:/Users/Utilisateur/Desktop/DocumentsSync/JNJ/AnalysePosture/JnJ_Contrast_Vision.csv'
path_result_contrast_vision_posture = 'C:/Users/Utilisateur/Desktop/DocumentsSync/JNJ/AnalysePosture/JnJ_Contrast_Vision_Posture.csv'
path_folder_figure = 'C:/Users/Utilisateur/Desktop/DocumentsSync/JNJ/AnalysePosture/'

df_result_posture = pd.read_csv(path_result_posture, sep=';')
# Exclude the S06 and S12 and S20
df_result_posture = df_result_posture.drop(df_result_posture[df_result_posture['Subject Public ID'] == 'S06'].index)
df_result_posture = df_result_posture.drop(df_result_posture[df_result_posture['Subject Public ID'] == 'S12'].index)
df_result_posture = df_result_posture.drop(df_result_posture[df_result_posture['Subject Public ID'] == 'S20'].index)

df_result_ratio_posture = pd.DataFrame()

df_result_mean_contrast_vision = pd.read_csv(path_result_mean_contrast_vision, sep=';')
# Exclude the P_006_Session1 and P_025_Session1 and P_025_Session1
df_result_mean_contrast_vision = df_result_mean_contrast_vision.drop(df_result_mean_contrast_vision[df_result_mean_contrast_vision['SubjectID'] == 'P_006_Session1'].index)
df_result_mean_contrast_vision = df_result_mean_contrast_vision.drop(df_result_mean_contrast_vision[df_result_mean_contrast_vision['SubjectID'] == 'P_012_Session1'].index)
df_result_mean_contrast_vision = df_result_mean_contrast_vision.drop(df_result_mean_contrast_vision[df_result_mean_contrast_vision['SubjectID'] == 'P_020_Session1'].index)
df_result_mean_contrast_vision = df_result_mean_contrast_vision.drop(df_result_mean_contrast_vision[df_result_mean_contrast_vision['SubjectID'] == 'P_025_Session1'].index)

df_result_contrast_vision_posture = pd.DataFrame()

# Get the list of 'Postural Sway - Acc - RMS Sway (m/s^2)' value in all 6 conditions
postural_value_FA_EO_Firm_list = df_result_posture.loc[lambda df_result_posture: df_result_posture['Condition'] == 'Feet Apart, Eyes Open, Firm Surface']['Postural Sway - Acc - RMS Sway (m/s^2)'].tolist()
postural_value_FA_EC_Firm_list = df_result_posture.loc[lambda df_result_posture: df_result_posture['Condition'] == 'Feet Apart, Eyes Closed, Firm Surface']['Postural Sway - Acc - RMS Sway (m/s^2)'].tolist()
postural_value_FA_EO_Foam_list = df_result_posture.loc[lambda df_result_posture: df_result_posture['Condition'] == 'Feet Apart, Eyes Open, Foam Surface']['Postural Sway - Acc - RMS Sway (m/s^2)'].tolist()
postural_value_FA_EC_Foam_list = df_result_posture.loc[lambda df_result_posture: df_result_posture['Condition'] == 'Feet Apart, Eyes Closed, Foam Surface']['Postural Sway - Acc - RMS Sway (m/s^2)'].tolist()
postural_value_OF_EO_Firm_list = df_result_posture.loc[lambda df_result_posture: df_result_posture['Condition'] == 'Optic Flow, Eyes Open, Firm Surface']['Postural Sway - Acc - RMS Sway (m/s^2)'].tolist()
postural_value_OF_EO_Foam_list = df_result_posture.loc[lambda df_result_posture: df_result_posture['Condition'] == 'Optic Flow, Eyes Open, Foam Surface']['Postural Sway - Acc - RMS Sway (m/s^2)'].tolist()

# print(df_result_posture.loc[lambda df_result_posture: df_result_posture['Condition'] == 'Feet Apart, Eyes Open, Foam Surface']['Postural Sway - Acc - RMS Sway (m/s^2)'])

# Calculate the ratio with the list of 'Postural Sway - Acc - RMS Sway (m/s^2)' value

list_romberg_ratio_firm = []
list_romberg_ratio_foam = []
list_somatosensory_ratio = []
list_visual_ratio = []
list_vestibular_ratio = []
list_optic_flow_ratio_firm = []
list_optic_flow_ratio_foam = []

for i in range(26):
    postural_value_FA_EO_Firm = float(postural_value_FA_EO_Firm_list[i].replace(',', '.'))
    postural_value_FA_EC_Firm = float(postural_value_FA_EC_Firm_list[i].replace(',', '.'))
    postural_value_FA_EO_Foam = float(postural_value_FA_EO_Foam_list[i].replace(',', '.'))
    postural_value_FA_EC_Foam = float(postural_value_FA_EC_Foam_list[i].replace(',', '.'))
    postural_value_OF_EO_Firm = float(postural_value_OF_EO_Firm_list[i].replace(',', '.'))
    postural_value_OF_EO_Foam = float(postural_value_OF_EO_Foam_list[i].replace(',', '.'))

    # Calculate Romberg Ratio (Firm & Foam)
    romberg_ratio_firm = (postural_value_FA_EC_Firm - postural_value_FA_EO_Firm) / (postural_value_FA_EC_Firm + postural_value_FA_EO_Firm) * 100
    list_romberg_ratio_firm.append(romberg_ratio_firm)
    romberg_ratio_foam = (postural_value_FA_EC_Foam - postural_value_FA_EO_Foam) / (postural_value_FA_EC_Foam + postural_value_FA_EO_Foam) * 100
    list_romberg_ratio_foam.append(romberg_ratio_foam)

    # Calculate Somatosensory Ratio
    somatosensory_ratio = postural_value_FA_EC_Firm / postural_value_FA_EO_Firm
    list_somatosensory_ratio.append(somatosensory_ratio)

    # Calculate visual ratio
    visual_ratio = postural_value_FA_EO_Foam / postural_value_FA_EO_Firm
    list_visual_ratio.append(visual_ratio)

    # Calculate vestibular ratio
    vestibular_ratio = postural_value_FA_EC_Foam / postural_value_FA_EO_Firm
    list_vestibular_ratio.append(vestibular_ratio)

    # Calculate optic flow ratio (Firm & Foam)
    optic_flow_ratio_firm = postural_value_FA_EO_Firm / postural_value_OF_EO_Firm
    list_optic_flow_ratio_firm.append(optic_flow_ratio_firm)
    optic_flow_ratio_foam = postural_value_FA_EO_Foam / postural_value_OF_EO_Foam
    list_optic_flow_ratio_foam.append(optic_flow_ratio_foam)

# Get the list of 30 subjects
list_subjectID = []
for i in range(1, 6):
    index_subjectID = f'{i:02d}'
    list_subjectID.append('P_0' + index_subjectID + '_Session1')

for i in range(7, 12):
    index_subjectID = f'{i:02d}'
    list_subjectID.append('P_0' + index_subjectID + '_Session1')

for i in range(13, 20):
    index_subjectID = f'{i:02d}'
    list_subjectID.append('P_0' + index_subjectID + '_Session1')

for i in range(21, 25):
    index_subjectID = f'{i:02d}'
    list_subjectID.append('P_0' + index_subjectID + '_Session1')

# for i in range(7, 25):
#     index_subjectID = f'{i:02d}'
#     list_subjectID.append('P_0' + index_subjectID + '_Session1')

for i in range(26, 31):
    index_subjectID = f'{i:02d}'
    list_subjectID.append('P_0' + index_subjectID + '_Session1')

# print(list_subjectID)

# Create the dataframe and save as csv
data_result_ratio_posture = {
    'SubjectID': list_subjectID,
    'romberg_ratio_firm': list_romberg_ratio_firm,
    'romberg_ratio_foam': list_romberg_ratio_foam,
    'somatosensory_ratio': list_somatosensory_ratio,
    'visual_ratio': list_visual_ratio,
    'vestibular_ratio': list_vestibular_ratio,
    'optic_flow_ratio_firm': list_optic_flow_ratio_firm,
    'optic_flow_ratio_foam': list_optic_flow_ratio_foam,
}

df_result_ratio_posture = pd.DataFrame(data_result_ratio_posture)

df_result_ratio_posture.to_csv(path_result_ratio_posture, sep=';')

# Combine the df_result_ratio_posture and df_result_mean_contrast_vision to be df_result_contrast_vision_posture
df_result_contrast_vision_posture = df_result_mean_contrast_vision.merge(df_result_ratio_posture, how='inner', on='SubjectID')
df_result_contrast_vision_posture = df_result_contrast_vision_posture.drop(columns=['Unnamed: 0'])

df_result_contrast_vision_posture.to_csv(path_result_contrast_vision_posture, sep=';')

# Define the analyse methode
def fit_sklearn(x, y):
    linear_regressor = linear_model.LinearRegression()
    x = np.array(x)
    y = np.array(y)
    x = x.reshape(-1, 1)
    linear_regressor.fit(x, y)
    y_pred = linear_regressor.predict(np.array([x.min(), x.max()]).reshape(-1, 1))
    linregress = scipy.stats.linregress(np.squeeze(x), np.squeeze(y))
    return linear_regressor, y_pred, linregress

# Get all the independent variable in the list (walking_speed & mean_percentage_pink_pixel_per_step)
independent_variables_list = []
independent_variables_list.append(np.array(df_result_contrast_vision_posture['walking_speed']))
independent_variables_list.append(np.array(df_result_contrast_vision_posture['mean_percentage_pink_pixel_per_step']))

# Get all the dependent variable in the list (all ratio)
dependent_variables_list = []
dependent_variables_list.append(np.array(df_result_contrast_vision_posture['romberg_ratio_firm']))
dependent_variables_list.append(np.array(df_result_contrast_vision_posture['romberg_ratio_foam']))
dependent_variables_list.append(np.array(df_result_contrast_vision_posture['somatosensory_ratio']))
dependent_variables_list.append(np.array(df_result_contrast_vision_posture['visual_ratio']))
dependent_variables_list.append(np.array(df_result_contrast_vision_posture['vestibular_ratio']))
dependent_variables_list.append(np.array(df_result_contrast_vision_posture['optic_flow_ratio_firm']))
dependent_variables_list.append(np.array(df_result_contrast_vision_posture['optic_flow_ratio_foam']))


for j in range(2):
    for k in range(7):
        print('j: ' + str(j))
        print('k:' + str(k))
        independent_variable = independent_variables_list[j]
        dependent_variable = dependent_variables_list[k]

        # figure = plt.gcf()
        figure = plt.figure(figsize=(20, 12), dpi=80)

        if j == 0:
            independent_variable_name = 'walking_speed'
        elif j == 1:
            independent_variable_name = 'mean_percentage_pink_pixel_per_step'

        if k == 0:
            dependent_variable_name = 'romberg_ratio_firm'
        elif k == 1:
            dependent_variable_name = 'romberg_ratio_foam'
        elif k == 2:
            dependent_variable_name = 'somatosensory_ratio'
        elif k == 3:
            dependent_variable_name = 'visual_ratio'
        elif k == 4:
            dependent_variable_name = 'vestibular_ratio'
        elif k == 5:
            dependent_variable_name = 'optic_flow_ratio_firm'
        elif k == 6:
            dependent_variable_name = 'optic_flow_ratio_foam'

        # Add a global title
        figure.suptitle('X:' + independent_variable_name + '    Y:' + dependent_variable_name, fontsize=14)

        # figure, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2)
        figure.tight_layout(pad=10)

        ########## First plot, vision_type 0, contrast 4.1
        ax1 = plt.subplot(321)

        x_vt_0_contrast_4 = []
        y_vt_0_contrast_4 = []

        # Get the x list and y list under the condition
        for i in range(((len(dependent_variable)) // 6)):
            x_vt_0_contrast_4.append(independent_variable[i * 6])
            y_vt_0_contrast_4.append(dependent_variable[i * 6])

        # print(x_vt_0_contrast_4)
        # print(y_vt_0_contrast_4)

        linear_regressor, y_pred, linregress = fit_sklearn(x_vt_0_contrast_4, y_vt_0_contrast_4)
        # Draw the scatter image
        plt.scatter(x_vt_0_contrast_4, y_vt_0_contrast_4, s=30, facecolors='none', edgecolors='k')

        # Draw the predict y value line
        if linregress.pvalue < 0.05:
            plt.plot([min(x_vt_0_contrast_4), max(x_vt_0_contrast_4)], y_pred, color=(0.9, 0.0, 0.0), linewidth=2)
        else:
            plt.plot([min(x_vt_0_contrast_4), max(x_vt_0_contrast_4)], y_pred, color=(0.3, 0.3, 0.3), linewidth=2)

        # Set the plot title
        plt.title('VT 0, C 4.1 / ' + 'R=' + str(linregress.rvalue)[1:4] + ' p=' + str(linregress.pvalue)[1:5])

        ########## Second plot, vision_type 1, contrast 4.1
        ax2 = plt.subplot(322)

        x_vt_1_contrast_4 = []
        y_vt_1_contrast_4 = []

        # Get the x list and y list under the condition
        for i in range(((len(dependent_variable)) // 6)):
            x_vt_1_contrast_4.append(independent_variable[i * 6 + 3])
            y_vt_1_contrast_4.append(dependent_variable[i * 6 + 3])

        # print(x_vt_1_contrast_4)
        # print(y_vt_1_contrast_4)

        linear_regressor, y_pred, linregress = fit_sklearn(x_vt_1_contrast_4, y_vt_1_contrast_4)
        # Draw the scatter image
        plt.scatter(x_vt_1_contrast_4, y_vt_1_contrast_4, s=30, facecolors='none', edgecolors='k')

        # Draw the predict y value line
        if linregress.pvalue < 0.05:
            plt.plot([min(x_vt_1_contrast_4), max(x_vt_1_contrast_4)], y_pred, color=(0.9, 0.0, 0.0), linewidth=2)
        else:
            plt.plot([min(x_vt_1_contrast_4), max(x_vt_1_contrast_4)], y_pred, color=(0.3, 0.3, 0.3), linewidth=2)

        # Set the plot title
        plt.title('VT 1, C 4.1 / ' + 'R=' + str(linregress.rvalue)[1:4] + ' p=' + str(linregress.pvalue)[1:5])

        ########## Third plot, vision_type 0, contrast 6.19
        ax3 = plt.subplot(323)

        x_vt_0_contrast_6 = []
        y_vt_0_contrast_6 = []

        # Get the x list and y list under the condition
        for i in range(((len(dependent_variable)) // 6)):
            x_vt_0_contrast_6.append(independent_variable[i * 6 + 1])
            y_vt_0_contrast_6.append(dependent_variable[i * 6 + 1])

        # print(x_vt_0_contrast_6)
        # print(y_vt_0_contrast_6)

        linear_regressor, y_pred, linregress = fit_sklearn(x_vt_0_contrast_6, y_vt_0_contrast_6)
        # Draw the scatter image
        plt.scatter(x_vt_0_contrast_6, y_vt_0_contrast_6, s=30, facecolors='none', edgecolors='k')

        # Draw the predict y value line
        if linregress.pvalue < 0.05:
            plt.plot([min(x_vt_0_contrast_6), max(x_vt_0_contrast_6)], y_pred, color=(0.9, 0.0, 0.0), linewidth=2)
        else:
            plt.plot([min(x_vt_0_contrast_6), max(x_vt_0_contrast_6)], y_pred, color=(0.3, 0.3, 0.3), linewidth=2)

        # Set the plot title
        plt.title('VT 0, C 6.19 / ' + 'R=' + str(linregress.rvalue)[1:4] + ' p=' + str(linregress.pvalue)[1:5])

        ########## Fourth plot, vision_type 1, contrast 6.19
        ax4 = plt.subplot(324)

        x_vt_1_contrast_6 = []
        y_vt_1_contrast_6 = []

        # Get the x list and y list under the condition
        for i in range(((len(dependent_variable)) // 6)):
            x_vt_1_contrast_6.append(independent_variable[i * 6 + 4])
            y_vt_1_contrast_6.append(dependent_variable[i * 6 + 4])

        # print(x_vt_1_contrast_6)
        # print(y_vt_1_contrast_6)

        linear_regressor, y_pred, linregress = fit_sklearn(x_vt_1_contrast_6, y_vt_1_contrast_6)
        # Draw the scatter image
        plt.scatter(x_vt_1_contrast_6, y_vt_1_contrast_6, s=30, facecolors='none', edgecolors='k')

        # Draw the predict y value line
        if linregress.pvalue < 0.05:
            plt.plot([min(x_vt_1_contrast_6), max(x_vt_1_contrast_6)], y_pred, color=(0.9, 0.0, 0.0), linewidth=2)
        else:
            plt.plot([min(x_vt_1_contrast_6), max(x_vt_1_contrast_6)], y_pred, color=(0.3, 0.3, 0.3), linewidth=2)

        # Set the plot title
        plt.title('VT 1, C 6.19 / ' + 'R=' + str(linregress.rvalue)[1:4] + ' p=' + str(linregress.pvalue)[1:5])

        ########## Fifth plot, vision_type 0, contrast 9.36
        ax5 = plt.subplot(325)

        x_vt_0_contrast_9 = []
        y_vt_0_contrast_9 = []

        # Get the x list and y list under the condition
        for i in range(((len(dependent_variable)) // 6)):
            x_vt_0_contrast_9.append(independent_variable[i * 6 + 2])
            y_vt_0_contrast_9.append(dependent_variable[i * 6 + 2])

        # print(x_vt_0_contrast_9)
        # print(y_vt_0_contrast_9)

        linear_regressor, y_pred, linregress = fit_sklearn(x_vt_0_contrast_9, y_vt_0_contrast_9)
        # Draw the scatter image
        plt.scatter(x_vt_0_contrast_9, y_vt_0_contrast_9, s=30, facecolors='none', edgecolors='k')

        # Draw the predict y value line
        if linregress.pvalue < 0.05:
            plt.plot([min(x_vt_0_contrast_9), max(x_vt_0_contrast_9)], y_pred, color=(0.9, 0.0, 0.0), linewidth=2)
        else:
            plt.plot([min(x_vt_0_contrast_9), max(x_vt_0_contrast_9)], y_pred, color=(0.3, 0.3, 0.3), linewidth=2)

        # Set the plot title
        plt.title('VT 0, C 9.36 / ' + 'R=' + str(linregress.rvalue)[1:4] + ' p=' + str(linregress.pvalue)[1:5])

        ########## Sixth plot, vision_type 1, contrast 9.36
        ax6 = plt.subplot(326)

        x_vt_1_contrast_9 = []
        y_vt_1_contrast_9 = []

        # Get the x list and y list under the condition
        for i in range(((len(dependent_variable)) // 6)):
            x_vt_1_contrast_9.append(independent_variable[i * 6 + 5])
            y_vt_1_contrast_9.append(dependent_variable[i * 6 + 5])

        # print(x_vt_1_contrast_9)
        # print(y_vt_1_contrast_9)

        linear_regressor, y_pred, linregress = fit_sklearn(x_vt_1_contrast_9, y_vt_1_contrast_9)
        # Draw the scatter image
        plt.scatter(x_vt_1_contrast_9, y_vt_1_contrast_9, s=30, facecolors='none', edgecolors='k')

        # Draw the predict y value line
        if linregress.pvalue < 0.05:
            plt.plot([min(x_vt_1_contrast_9), max(x_vt_1_contrast_9)], y_pred, color=(0.9, 0.0, 0.0), linewidth=2)
        else:
            plt.plot([min(x_vt_1_contrast_9), max(x_vt_1_contrast_9)], y_pred, color=(0.3, 0.3, 0.3), linewidth=2)

        # Set the plot title
        plt.title('VT 1, C 9.36 / ' + 'R=' + str(linregress.rvalue)[1:4] + ' p=' + str(linregress.pvalue)[1:5])

        # plt.show()

        figure.savefig(path_folder_figure + str(j) + '_' + str(k) + '_' + independent_variable_name + '_' + dependent_variable_name + '.png', dpi=200)


# def f(t):
#     return np.exp(-t) * np.cos(2*np.pi*t)
#
# t1 = np.arange(0.0, 3.0, 0.01)
#
# ax1 = plt.subplot(212)
# ax1.margins(0.05)           # Default margin is 0.05, value 0 means fit
# ax1.plot(t1, f(t1))
#
# ax2 = plt.subplot(221)
# ax2.margins(2, 2)           # Values >0.0 zoom out
# ax2.plot(t1, f(t1))
# ax2.set_title('Zoomed out')
#
# ax3 = plt.subplot(222)
# ax3.margins(x=0, y=-0.25)   # Values in (-0.5, 0.0) zooms in to center
# ax3.plot(t1, f(t1))
# ax3.set_title('Zoomed in')
#
# plt.show()

# df_CT
# session_name, trial_name, index, category, light_condition_lux, vision_type, contrast, windows_visibility, 
# task_type, smi, index_in_phase, first_step_left, index_maze, unity_color, trial_duration, walking_duration, 
# walking_distance, walking_speed, sd_walking_speed, nb_step, nb_pink_zone, nb_pink_zone_per_step, nb_pink_pixel, 
# nb_pink_pixel_per_step, nb_pink_and_red_pixel, mean_percentage_pink_pixel_per_step

# df_SOT
# List of the visual conditions in the experiment
# list_SOT_conds = ["Feet Apart. Eyes Open. Firm Surface",        #SOT1,    "Eyes Open, Fixed Vision, Fixed Floor"
# "Feet Apart. Eyes Closed. Firm Surface",                        #SOT2,    "Eyes Closed, Fixed Vision, Fixed Floor"
# "Feet Apart. Eyes Open. Foam Surface",                          #SOT3,    "Eyes Open, Fixed Vision, Sway Floor"
# "Feet Apart. Eyes Closed. Foam Surface",                        #SOT4,    "Eyes Closed, Fixed Vision, Sway Floor"
# "Optic Flow. Eyes Open. Firm Surface",                          #SOT1FO,    "Eyes Open, Fixed Vision, Fixed, FloorFlux Optic"
# "Optic Flow. Eyes Open. Foam Surface"]                          #SOT3FO,    "Eyes Open, Fixed Vision, Sway Floor, Flux Optic"

# SubjectID, Condition, 
### Postural_Sway_-_Acc_-_Sway_Area_(m^2/s^4), 
### Postural_Sway_-_Acc_-_Jerk_(m^2/s^5), 
### Postural_Sway_-_Acc_-_Mean_Velocity_(m/s), 
### Postural_Sway_-_Acc_-_Path_Length_(m/s^2), 
### Postural_Sway_-_Acc_-_RMS_Sway_(m/s^2), 

################## Loading the dataset  ######################################
# df_SOT = pd.read_csv(fnameSOT, sep=";",header=0)
# df_CT = pd.read_csv(fnameCT, sep=";",header=0)


################## df_CT means res ##################
# onevalpercond=[]
# for s in np.unique(df_CT.session_name):
#     for vt in np.unique(df_CT.vision_type):
#         for cst in np.unique(df_CT.contrast):
#             dfh=df_CT[np.logical_and.reduce((df_CT.session_name==s, df_CT.vision_type==vt, df_CT.contrast==cst))]
#             onevalpercond.append([s,vt,cst,
#             dfh.trial_duration.mean(),
#             dfh.walking_duration.mean(),
#             dfh.walking_distance.mean(),
#             dfh.walking_speed.mean(),
#             dfh.sd_walking_speed.mean(),
#             dfh.nb_step.mean(),
#             dfh.nb_pink_zone.mean(),
#             dfh.nb_pink_zone_per_step.mean(),
#             dfh.nb_pink_pixel.mean(),
#             dfh.nb_pink_pixel_per_step.mean(),
#             dfh.nb_pink_and_red_pixel.mean(),
#             dfh.mean_percentage_pink_pixel_per_step.mean()
#             ])
#
# df_onevalpercond = pd.DataFrame(onevalpercond, columns=[
# 'SubjectID', 'vision_type', 'contrast',
# 'trial_duration', 'walking_duration', 'walking_distance',
# 'walking_speed', 'sd_walking_speed', 'nb_step', 'nb_pink_zone',
# 'nb_pink_zone_per_step', 'nb_pink_pixel', 'nb_pink_pixel_per_step',
# 'nb_pink_and_red_pixel', 'mean_percentage_pink_pixel_per_step'])
#
#
# df_onevalpercond.to_csv(ffn_meanCT, sep=";")
# df_CT_means = pd.read_csv(ffn_meanCT, sep=";")
# CT_VDs = ['walking_speed','mean_percentage_pink_pixel_per_step']

################## df_SOT means res ##################
# onevalpercond=[]
# for s in np.unique(df_SOT.Subject Public ID):
#     for cond in np.unique(df_SOT.Condition):
#         for cst in np.unique(df_SOT.contrast):
#             dfh=df_SOT[np.logical_and.reduce((df_SOT.session_name==s, df_SOT.vision_type==vt, df_SOT.contrast==cst))]
#             onevalpercond.append([s,vt,cst,
#             dfh.trial_duration.mean(),
#             dfh.walking_duration.mean(),
#             dfh.walking_distance.mean(),
#             dfh.walking_speed.mean(),
#             dfh.sd_walking_speed.mean(),
#             dfh.nb_step.mean(),
#             dfh.nb_pink_zone.mean(),
#             dfh.nb_pink_zone_per_step.mean(),
#             dfh.nb_pink_pixel.mean(),
#             dfh.nb_pink_pixel_per_step.mean(),
#             dfh.nb_pink_and_red_pixel.mean(),
#             dfh.mean_percentage_pink_pixel_per_step.mean()
#             ])
                    
# df_onevalpercond = pd.DataFrame(onevalpercond, columns=[
# 'SubjectID', 'vision_type', 'contrast',
# 'trial_duration', 'walking_duration', 'walking_distance',
# 'walking_speed', 'sd_walking_speed', 'nb_step', 'nb_pink_zone',
# 'nb_pink_zone_per_step', 'nb_pink_pixel', 'nb_pink_pixel_per_step',
# 'nb_pink_and_red_pixel', 'mean_percentage_pink_pixel_per_step'])
#
# df_onevalpercond.to_csv(ffn_meanCT, sep=";")
# df_CT_means = pd.read_csv(ffn_meanCT, sep=";")
# CT_VDs = ['walking_speed','mean_percentage_pink_pixel_per_step']

# Subject Public ID, Condition, 
# Postural Sway - Acc - 95% Ellipse Axis 1 Radius (m/s^2), 
# Postural Sway - Acc - 95% Ellipse Axis 2 Radius (m/s^2), 
# Postural Sway - Acc - 95% Ellipse Rotation (radians), 
### Postural Sway - Acc - Sway Area (m^2/s^4), 
# Postural Sway - Acc - Centroidal Frequency (Hz), 
# Postural Sway - Acc - Centroidal Frequency (Coronal) (Hz), 
# Postural Sway - Acc - Centroidal Frequency (Sagittal) (Hz), 
# Postural Sway - Acc - Frequency Dispersion (AD), 
# Postural Sway - Acc - Frequency Dispersion (Coronal) (AD), 
# Postural Sway - Acc - Frequency Dispersion (Sagittal) (AD), 
### Postural Sway - Acc - Jerk (m^2/s^5), 
# Postural Sway - Acc - Jerk (Coronal) (m^2/s^5), 
# Postural Sway - Acc - Jerk (Sagittal) (m^2/s^5), 
### Postural Sway - Acc - Mean Velocity (m/s), 
# Postural Sway - Acc - Mean Velocity (Coronal) (m/s), 
# Postural Sway - Acc - Mean Velocity (Sagittal) (m/s), 
### Postural Sway - Acc - Path Length (m/s^2), 
# Postural Sway - Acc - Path Length (Coronal) (m/s^2), 
# Postural Sway - Acc - Path Length (Sagittal) (m/s^2), 
### Postural Sway - Acc - RMS Sway (m/s^2), 
# Postural Sway - Acc - RMS Sway (Coronal) (m/s^2), 
# Postural Sway - Acc - RMS Sway (Sagittal) (m/s^2), 
# Postural Sway - Acc - Range (m/s^2), 
# Postural Sway - Acc - Range (Coronal) (m/s^2), 
# Postural Sway - Acc - Range (Sagittal) (m/s^2), 
# Postural Sway - Angles - 95% Ellipse Axis 1 Radius (degrees), 
# Postural Sway - Angles - 95% Ellipse Axis 2 Radius (degrees), 
# Postural Sway - Angles - 95% Ellipse Rotation (radians), 
# Postural Sway - Angles - Sway Area (degrees^2), 
# Postural Sway - Angles - Duration (s), 
# Postural Sway - Angles - RMS Sway (degrees), 
# Postural Sway - Angles - RMS Sway (Coronal) (degrees), 
# Postural Sway - Angles - RMS Sway (Sagittal) (degrees)


# SOT_VDs = ['Postural Sway - Acc - Sway Area (m^2/s^4)', 'Postural Sway - Acc - Jerk (m^2/s^5)', 'Postural Sway - Acc - Mean Velocity (m/s)',
# 'Postural Sway - Acc - Path Length (m/s^2)', 'Postural Sway - Acc - RMS Sway (m/s^2)']

# Begin the loop
# for dependent_variable in dependent_variables_list:
#     for independent_variable in independent_variables_list:
#         figure = plt.figure(figsize=(12, 8), dpi=300)
#         ax1=plt.subplot(3, 2, 1)
#         dfh = df2[np.logical_and(df2.Surface=='Firm', df2.Visibility=='Eyes Closed')]
#         x,y=dfh[VI].values.reshape(-1, 1),dfh[VD].values.reshape(-1, 1)
#         y, x = y[~np.isnan(x)].reshape(-1, 1), x[~np.isnan(x)].reshape(-1, 1)# remove NaNs
#         linear_regressor, y_pred, lr = fit_sklearn(x,y)
#         plt.scatter(x, y,  s=30, facecolors='none', edgecolors='k')
#         if lr.pvalue<0.05:
#             plt.plot([x.min(), x.max()], y_pred, color=(0.9,0.0,0.0), linewidth=2)
#         else:
#             plt.plot([x.min(), x.max()], y_pred, color=(0.3,0.3,0.3), linewidth=2)
#         plt.title('Firm / Eye Closed / '+'R='+str(lr.rvalue)[1:4]+' p='+str(lr.pvalue)[1:5])










