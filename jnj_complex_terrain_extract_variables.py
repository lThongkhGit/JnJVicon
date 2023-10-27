import os
from typing import List, Any, Union

import pandas
import numpy
import math
import cv2
import shutil
import csv
import statistics

from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw, ImageFilter
from ppretty import ppretty

import jnj_methods_analysis
import jnj_methods_analysis as analysis
import jnj_methods_data_preparation as preparation
import numpy as np
import csv
import copy

# The path of the all sessions data
root_data_path = 'C:/Users/lthongkh/Documents/JNJ/clinical_study_Complex/'
# root_data_path = 'D:/JNJ/clinical_study_Complex/'

# The path of all the maze image
maze_image_folder_path = 'C:/Users/lthongkh/Documents/JNJ/Maze/MazeImages/'
# maze_image_folder_path = 'H:/JNJ/Maze/MazeImages/'

# The step length of the subject in meter
# subject_step_length = 0.7

# The size of the subject's foot in inches
# subject_foot_size = 43

# List of foot size
foot_size_list = [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

# List of foot length in meter
foot_length_list = [0.242, 0.248, 0.255, 0.26, 0.263, 0.27, 0.275, 0.283, 0.288, 0.296, 0.3, 0.31, 0.316]

# margin for foot placement (margin for each side of the foot in meter)
margin = 0.01

# The pixel/meter according to the projected image
pixel_per_meter = 587.5

# The min Vicon y axis of the projected image (in mm)
min_vicon_y_mm = -2980.59

# The min Vicon x axis of the projected image (in mm)
min_vicon_x_mm = 1388.3

# The number of pixels in x axis for maze image
number_pixel_x = 2390

# The number of pixels in y axis for maze image
number_pixel_y = 1080


class ComplexTerrainSessionData(preparation.SessionData):
    def __init__(self, root_data_path_, participant_id_, session_id_):
        # Call the parent method
        super().__init__(root_data_path_, participant_id_, session_id_)

        # Configure the path of PWSCalibration folder
        self.complex_calibration_folder_path = self.session_folder_path + 'PWSCalibration/'

        # Create trial data list
        self.list_trials = []

        # Create archived trial data list
        self.list_archived_trials = []
        session_archived_folder_path = self.session_folder_path + 'Archived'
        for path in os.listdir(session_archived_folder_path):
            # (self, session_, index_, category_, light_condition_lux_, task_type_, step_1_height_, step_1_position_, step_2_height_, step_2_position_, index_in_phase_):
            complex_trial_data = ComplexTerrainTrialData(False, self, '0', 'Archived', 14.34, 6.19, '2Steps', 'True-0', 0, 'True', 1, 0.4972)
            complex_trial_data.trial_folder_path = session_archived_folder_path + '/' + path + '/'
            self.list_archived_trials.append(complex_trial_data)

        print('self.list_archived_trials: ', len(self.list_archived_trials))

        # Add PWS calibration trials
        # list_calibration_folders = [f.path for f in os.scandir(self.complex_calibration_folder_path) if f.is_dir()]
        # for folder in list_calibration_folders:
        #     self.list_trials.append(ComplexTerrainTrialData(True, self, int(folder.split('_')[-1]), None, None, None, None, None, None, None, None, None,))

        # Add complex terrain trials
        for index, row in self.df_config.iterrows():
            self.list_trials.append(ComplexTerrainTrialData(False, self, row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9]))

        print('self.list_trials: ', len(self.list_trials))


# The trial data in Complex Terrain task
class ComplexTerrainTrialData(preparation.TrialData):
    def __init__(self, is_pws, session_, index_, category_, light_condition_lux_, contrast_, windows_visibility_,
                 task_type_, index_in_phase_, first_step_left_, index_maze_, unity_color_):

        # Call the parent method
        super().__init__(session_)

        if is_pws:
            # The TrialConfig instance of this trial
            self.config = PWSCalibrationTrialConfig(self, index_)

            # The path of the trial folder
            self.trial_folder_path = self.session.session_folder_path + 'PWSCalibration/' + self.config.name + '/'
        else:
            # The TrialConfig instance of this trial
            self.config = ComplexTerrainTrialConfig(self, index_, category_, light_condition_lux_, contrast_, windows_visibility_, task_type_, index_in_phase_, first_step_left_, index_maze_,
                                                    unity_color_)

            # The path of the trial folder
            self.trial_folder_path = self.session.session_folder_path + 'Data/Trial_' + str(
                self.config.index) + '_' + self.config.category + '_' + self.config.task_type + '/'

        # The results of the trial
        self.result = ComplexTerrainTrialResult(self)

        # The list of heel strike frames for left foot
        self.list_frame_HS_left = None

        # The list of heel strike frames for right foot
        self.list_frame_HS_right = None

        # The list of toe off frames for left foot
        self.list_frame_TO_left = None

        # The list of toe off frames for right foot
        self.list_frame_TO_right = None

        # The list of precise stable step frames for left foot
        self.list_frame_stable_precise_left = None

        # The list of precise stable step frames for right foot
        self.list_frame_stable_precise_right = None

    # data preparation:
    # read vicon data, restructure and filter data
    # can be used for both complex terrain trials and for PWS calibration trials
    def data_preparation(self):
        # Call the parent method
        super().data_preparation()

    # The analysis routine of global variables and spatial temporal variables for complex terrain task and PWS calibration
    def complex_terrain_analysis_routine(self, verbose=True, plot_it=False, save_plot=False):

        if len(self.df_vicon_raw.index) < 1:
            print(self.trial_folder_path + ' data error, please verify')
            return

        # Calculate the global variables : trial duration, walking duration, walking distance, walking speed
        self.calculate_global_variables()

        # Get the basic information of step analyse
        self.get_basic_info_step_analyse()

        # Define the save plot path for the step analyse
        self.save_plot_path = self.session.session_plot_folder_path + self.config.name + '_complex_' + str(self.config.index) + '_'

        # Get the list of frames of all heel strike in the trial for left and right foot
        self.list_frame_HS_left, self.list_frame_HS_right, \
        self.list_frame_TO_left, self.list_frame_TO_right, \
            = analysis.complex_calculation(
            self.df_vicon_filtered.loc[:, preparation.name_right_heel_marker].to_numpy(),
            self.df_vicon_filtered.loc[:, preparation.name_left_heel_marker].to_numpy(),
            self.df_vicon_filtered.loc[:, preparation.name_right_toe_marker].to_numpy(),
            self.df_vicon_filtered.loc[:, preparation.name_left_toe_marker].to_numpy(),
            verbose=verbose, plot_it=plot_it, save_plot=save_plot,
            save_plot_path=self.save_plot_path)

        # For PWS: keep only heel strikes frames that the participant is walking in the correct direction and in the PWS zone
        # For other trials : keep only heel strikes frames that the entire foot is in the range of the maze image
        self.filter_frame_heel_strike()

        # Calculate Margin of Stability variables for all trials (including pws)
        self.calculate_margin_stability_variables()

        # Calculate the step length and step width for each step for all trials (including pws)
        self.calculate_step_length_width()

        # Analyse all variables if it is not PWS
        if self.config.category != 'pws':
            # 2. Get the list of left and right foot stable position
            self.calculate_foot_stable_positions()

            # 3. Draw the coordinate of left and right foot stable position in the maze image
            self.draw_two_feet_stable_positions()

            # Analyse image for each step

        # Export trial result if it is not PWS
        if self.config.category != 'pws':
            # 7. Export trial result to csv
            self.export_trial_result()

    # Calculate the global variables : trial duration, walking duration, walking distance, walking speed
    def calculate_global_variables(self):
        if not self.load_vicon_filtered():
            return

        # Calculate the trial duration (s)
        self.result.trial_duration = \
            self.df_vicon_filtered.loc[len(self.df_vicon_filtered.index) - 1, 'TimeElapsed'].values[0]

        # Important!
        # The original "Head" rigidbody should be used for calculating accumulated distance since it is recorded by Vicon,
        # so if there is a period that the Head isn't tracked at the beginning, data will be missing and then completed by Spline during post processing.
        # If we use HeadCalibrated, the accumulated distance will be much longer when the Head isn't tracked at the beginning,
        # since HeadCalibrated is generated in Unity, Unity will suppose that the Head is at (0,0,0) instead of leaving the data missing,
        # then the HeadCalibrated will suddenly jump to the right position when the Head is tracked,
        # and during post processing, Spline won't be applied to fix that error because it's only applied to missing data.
        # The "HeadCalibrated" should be used to calculate head orientation

        # df_head: The head rigidbody position dataframe
        df_head = self.df_vicon_filtered.loc[:, ('Head', preparation.position_iterables)] / 1000.

        # Get all frames when y is between [-3, 0.5]
        df_head = df_head[np.logical_and(df_head[('Head', 'PosY')] < 0.5, df_head[('Head', 'PosY')].values > -3.)]

        # calculate scalar speed (x and y axis) for each frame when y is between [-3, 0.5]
        diff_head = df_head.diff().fillna(0.)
        diff_head['DistanceXY'] = np.sqrt(diff_head.loc[:, ('Head', 'PosX')] ** 2 + diff_head.loc[:, ('Head', 'PosY')] ** 2)
        diff_head['SpeedXY'] = diff_head['DistanceXY'] / (1 / 120.)

        # Calculate the walking distance (m) The walking distance accumulated by the head in x and y axis when y is between [-3, 0.5]
        self.result.walking_distance = np.sum(diff_head['DistanceXY'])

        # Calculate the walking duration (s) (the duration of all frames when y is between [-3, 0.5])
        self.result.walking_duration = len(diff_head.index) / 120.

        # Calculate the mean walking speed (mm/s) (x and y axis)
        self.result.walking_speed = np.mean(diff_head['SpeedXY'].to_numpy())

        # Calculate the standard deviation of walking speed of all frames when y is between [-3, 0.5]
        self.result.sd_walking_speed = np.std(diff_head['SpeedXY'].to_numpy())

        # Calculate the pitch hand orientation for the mean and standard deviation
        df_head_quaternion = self.df_vicon_filtered.loc[:, ('HeadCalibrated', preparation.quaternion_iterables)].fillna(0)
        list_quaternion = df_head_quaternion.values.tolist()
        print(list_quaternion)

        # If the quaternion is (0, 0, 0, 0), change it to identity quaternion (0, 0, 0, 1)
        for i in range(len(list_quaternion)):
            if list_quaternion[i] == [0, 0, 0, 0]:
                list_quaternion[i] = [0, 0, 0, 1]

        list_rotation = Rotation.from_quat(list_quaternion)
        list_euler = list_rotation.as_euler('yxz', degrees=True)

        # The outcome is when subject look under euler angle is negative
        list_euler_x = []

        for euler in list_euler:
            list_euler_x.append(euler[1])

        # print(list_euler_x)
        self.result.pitch_head_mean = np.mean(list_euler_x)
        self.result.pitch_head_sd = np.std(list_euler_x)

        # Save diff_head dataframe to csv file
        # diff_head.to_csv(self.trial_folder_path + 'Vicon/' + self.config.name + '_diff_head.csv', sep=';')

    # Get the basic information of step analyse
    def get_basic_info_step_analyse(self):
        complex_calibration_file_path = session.session_folder_path + "/PWSCalibration/Calibration.csv"
        complex_calibration_df = pandas.read_csv(complex_calibration_file_path, sep=';')

        # print(complex_calibration_df.columns[2])

        subject_actual_step_length = float(complex_calibration_df.columns[2])
        if subject_actual_step_length <= 0.525:
            self.subject_step_length = 0.5
        elif 0.525 < subject_actual_step_length <= 0.575:
            self.subject_step_length = 0.55
        elif 0.575 < subject_actual_step_length <= 0.625:
            self.subject_step_length = 0.6
        elif 0.625 < subject_actual_step_length <= 0.675:
            self.subject_step_length = 0.65
        elif 0.675 < subject_actual_step_length <= 0.725:
            self.subject_step_length = 0.7
        elif 0.725 < subject_actual_step_length <= 0.775:
            self.subject_step_length = 0.75
        elif 0.775 < subject_actual_step_length:
            self.subject_step_length = 0.8

        self.subject_foot_size = float(complex_calibration_df.columns[3])

        print('Subject step length: ' + self.subject_step_length.__str__() + ' Subject foot size: ' + self.subject_foot_size.__str__())

    # For PWS: keep only heel strikes frames that the participant is walking in the correct direction and in the PWS zone
    # For other trials : keep only heel strikes frames that the entire foot is in the range of the maze image
    def filter_frame_heel_strike(self):
        # For PWS:
        # Keep only one direction (the walking direction of complex trials) for PWS (this direction is repeated five times)
        if self.config.category == 'pws':
            # df_head: The head rigidbody position dataframe
            df_head = self.df_vicon_filtered.loc[:, ('Head', preparation.position_iterables)] / 1000.
            # frames_start_direction: the list of frames that the head across the 2 meter in y axis in the walking direction of complex trials
            frames_start_direction = []
            # frames_end_direction: the list of frames that the head across the -2 meter in y axis in the walking direction of complex trials
            frames_end_direction = []
            # Get all frames when y is between [-2, 2]
            in_zone = False  # True if the y axis of head is in between [-2, 2]
            correct_direction = True  # True if the participant is walking in the right direction (from the door to inside of the Rue Artificielle)
            for frame in range(df_head.index.size):
                if not in_zone and correct_direction:  # If not in the PWS zone and in the correct direction, look for the start frame
                    if df_head.loc[frame, ('Head', 'PosY')] < 2.:
                        frames_start_direction.append(frame)
                        in_zone = True
                elif in_zone and correct_direction:  # If in the PWS zone and in the correct direction, look for the end frame
                    if df_head.loc[frame, ('Head', 'PosY')] < -2.:
                        frames_end_direction.append(frame)
                        in_zone = False
                        correct_direction = False
                elif not in_zone and not correct_direction:
                    if df_head.loc[frame, ('Head', 'PosY')] > -2.:
                        in_zone = True
                elif in_zone and not correct_direction:
                    if df_head.loc[frame, ('Head', 'PosY')] > 2.:
                        in_zone = False
                        correct_direction = True
            # print('------------frames_start_direction:')
            # print(frames_start_direction)
            # print('------------frames_end_direction:')
            # print(frames_end_direction)

            # Get the list of indices of all frames that the participant is walking in the correct direction and in the PWS zone
            keep_indices = []
            for start_frame, end_frame in zip(frames_start_direction, frames_end_direction):
                keep_indices.extend(range(start_frame, end_frame))

            # Remove the heel strike frame if the frame is not in the range
            list_frame_HS_left_copy = copy.deepcopy(self.list_frame_HS_left)
            for frame in list_frame_HS_left_copy:
                if frame not in keep_indices:
                    self.list_frame_HS_left.remove(frame)
                    # print('remove: ' + str(frame))
            list_frame_HS_right_copy = copy.deepcopy(self.list_frame_HS_right)
            for frame in list_frame_HS_right_copy:
                if frame not in keep_indices:
                    self.list_frame_HS_right.remove(frame)
                    # print('remove: ' + str(frame))

        # For other trials
        # Keep only heel strikes frames that the entire foot is in the range of the maze image
        else:
            # Calculate the red rectangle length for each size of foot
            red_rect_L_pixel = foot_length_list[foot_size_list.index(int(self.subject_foot_size))] * pixel_per_meter + 2 * margin * pixel_per_meter

            # print('red_rect_L_pixel: ' + str(red_rect_L_pixel))

            # The left side threshold of maze image for Vicon y coordinate
            left_threshold_vicon_y = int(red_rect_L_pixel * 1000 / pixel_per_meter + min_vicon_y_mm)

            # The right side threshold of maze image for Vicon y coordinate
            right_threshold_vicon_y = int(number_pixel_x * 1000 / pixel_per_meter + min_vicon_y_mm)

            # print('range: ' + str(left_threshold_vicon_y) + ' to ' + str(right_threshold_vicon_y))

            # Remove the heel strike frame if the position of heel is not in the range
            list_frame_HS_left_copy = copy.deepcopy(self.list_frame_HS_left)
            for frame in list_frame_HS_left_copy:
                # print(str(frame) + ':      ' + str(self.df_vicon_filtered.loc[frame, (preparation.name_left_heel_marker, 'PosY')]))
                if not left_threshold_vicon_y < self.df_vicon_filtered.loc[frame, (preparation.name_left_heel_marker, 'PosY')] < right_threshold_vicon_y:
                    self.list_frame_HS_left.remove(frame)
                    # print('remove: ' + str(frame))
            list_frame_HS_right_copy = copy.deepcopy(self.list_frame_HS_right)
            for frame in list_frame_HS_right_copy:
                # print(str(frame) + ':      ' + str(self.df_vicon_filtered.loc[frame, (preparation.name_right_heel_marker, 'PosY')]))
                if not left_threshold_vicon_y < self.df_vicon_filtered.loc[frame, (preparation.name_right_heel_marker, 'PosY')] < right_threshold_vicon_y:
                    self.list_frame_HS_right.remove(frame)
                    # print('remove: ' + str(frame))

    # Calculate Margin of Stability variables
    # Can be used for both complex terrain trials or pws calibration trials
    def calculate_margin_stability_variables(self):
        for axis in preparation.position_iterables:
            # Calculate CoM - Centre of Mass - definition - barycenter of the 4 markers (in meter)
            '''self.df_vicon_filtered['COM_' + axis] = (self.df_vicon_filtered.loc[:, (preparation.name_LASI_marker, axis)] +
                                                     self.df_vicon_filtered.loc[:, (preparation.name_LPSI_marker, axis)] +
                                                     self.df_vicon_filtered.loc[:, (preparation.name_RASI_marker, axis)] +
                                                     self.df_vicon_filtered.loc[:, (preparation.name_RPSI_marker, axis)]) / 4000.'''
            # Updated the calculation of CoM, now simply uses the marker created to that end
            self.df_vicon_filtered['COM_' + axis] = (self.df_vicon_filtered.loc[:,(preparation.name_pelvis_marker, axis)]) / 1000.

            # Calculate distanceCoM - the distance of the CoM in each frame (in meter)
            self.df_vicon_filtered['DistanceCOM_' + axis] = self.df_vicon_filtered['COM_' + axis].diff().fillna(0.)

            # Calculate velocityCoM - the velocity of the CoM (in m/s)
            self.df_vicon_filtered['VelocityCOM_' + axis] = self.df_vicon_filtered['DistanceCOM_' + axis] / (1 / 120.)

            # Calculate AccelerationCoM - the acceleration of the CoM (in m/s^2)
            self.df_vicon_filtered['AccelerationCOM_' + axis] = self.df_vicon_filtered['VelocityCOM_' + axis].diff().fillna(0.) / (1 / 120.)

            # Calculate the Distance between the CoM and Heel (in meter)
            self.df_vicon_filtered['DistanceCOMLeftHeel_' + axis] = self.df_vicon_filtered['COM_' + axis] - self.df_vicon_filtered.loc[:, (preparation.name_left_heel_marker, axis)] / 1000.
            self.df_vicon_filtered['DistanceCOMRightHeel_' + axis] = self.df_vicon_filtered['COM_' + axis] - self.df_vicon_filtered.loc[:, (preparation.name_right_heel_marker, axis)] / 1000.

        # Calculate the Distance between the CoM and left foot Heel (in meter)
        self.df_vicon_filtered['DistanceCOMLeftHeel'] = np.sqrt(self.df_vicon_filtered['DistanceCOMLeftHeel_PosX'] ** 2
                                                                + self.df_vicon_filtered['DistanceCOMLeftHeel_PosY'] ** 2
                                                                + self.df_vicon_filtered['DistanceCOMLeftHeel_PosZ'] ** 2)

        # Calculate the Distance between the CoM and right foot Heel (in meter)
        self.df_vicon_filtered['DistanceCOMRightHeel'] = np.sqrt(self.df_vicon_filtered['DistanceCOMRightHeel_PosX'] ** 2
                                                                 + self.df_vicon_filtered['DistanceCOMRightHeel_PosY'] ** 2
                                                                 + self.df_vicon_filtered['DistanceCOMRightHeel_PosZ'] ** 2)

        # Calculate XCoM - Extrapolated center of mass
        for axis in preparation.position_iterables:
            # Left foot
            self.df_vicon_filtered['XCOM_Left_' + axis] = self.df_vicon_filtered['COM_' + axis]
            + self.df_vicon_filtered['VelocityCOM_' + axis] / np.sqrt(9.81 / self.df_vicon_filtered['DistanceCOMLeftHeel'])

            # Right foot
            self.df_vicon_filtered['XCOM_Right_' + axis] = self.df_vicon_filtered['COM_' + axis]
            + self.df_vicon_filtered['VelocityCOM_' + axis] / np.sqrt(9.81 / self.df_vicon_filtered['DistanceCOMRightHeel'])

        # Calculate the MOS_AP
        # reverse the two sides of calculation so that all the MOS_AP of right foot to positive
        # Left foot
        self.df_vicon_filtered['MOS_AP_Left_PosX'] = self.df_vicon_filtered['XCOM_Left_PosX'] - self.df_vicon_filtered[(preparation.name_left_toe_marker, 'PosX')] / 1000.
        self.df_vicon_filtered['MOS_AP_Left_PosY'] = self.df_vicon_filtered['XCOM_Left_PosY'] - self.df_vicon_filtered[(preparation.name_left_toe_marker, 'PosY')] / 1000.
        self.df_vicon_filtered['MOS_AP_Left_PosZ'] = self.df_vicon_filtered['XCOM_Left_PosZ'] - self.df_vicon_filtered[(preparation.name_left_toe_marker, 'PosZ')] / 1000.
        # self.df_vicon_filtered['MOS_AP_Left_norm'] = np.sqrt(self.df_vicon_filtered['MOS_AP_Left_PosX'] ** 2 + self.df_vicon_filtered['MOS_AP_Left_PosY'] ** 2 + self.df_vicon_filtered['MOS_AP_Left_PosZ'] ** 2)

        # Right foot
        self.df_vicon_filtered['MOS_AP_Right_PosX'] = self.df_vicon_filtered['XCOM_Right_PosX'] - self.df_vicon_filtered[(preparation.name_right_toe_marker, 'PosX')] / 1000.
        self.df_vicon_filtered['MOS_AP_Right_PosY'] = self.df_vicon_filtered['XCOM_Right_PosY'] - self.df_vicon_filtered[(preparation.name_right_toe_marker, 'PosY')] / 1000.
        self.df_vicon_filtered['MOS_AP_Right_PosZ'] = self.df_vicon_filtered['XCOM_Right_PosZ'] - self.df_vicon_filtered[(preparation.name_right_toe_marker, 'PosZ')] / 1000.
        # self.df_vicon_filtered['MOS_AP_Right_norm'] = np.sqrt(self.df_vicon_filtered['MOS_AP_Right_PosX'] ** 2 + self.df_vicon_filtered['MOS_AP_Right_PosY'] ** 2 + self.df_vicon_filtered['MOS_AP_Right_PosZ'] ** 2)

        # Find the variables at all the frames of heel strike
        list_foot = []
        list_frame_heel_strike = []
        list_com_x = []
        list_com_y = []
        list_com_z = []
        list_xcom_x = []
        list_xcom_y = []
        list_xcom_z = []
        list_l = []
        heel_x = []
        heel_y = []
        heel_z = []
        BOSAP_toe_x = []
        BOSAP_toe_y = []
        BOSAP_toe_z = []
        BOSML_MTP_joint_x = []
        BOSML_MTP_joint_y = []
        BOSML_MTP_joint_z = []
        MOS_AP_x = []
        MOS_AP_y = []
        MOS_AP_z = []
        MOS_ML_x = []
        MOS_ML_y = []
        MOS_ML_z = []
        Quality_MOS_AP_Y = []
        Quality_MOS_ML_X = []

        for frame in self.list_frame_HS_left:
            list_foot.append('Left')
            list_frame_heel_strike.append(frame)
            list_com_x.append(self.df_vicon_filtered.loc[frame, 'COM_PosX'])
            list_com_y.append(self.df_vicon_filtered.loc[frame, 'COM_PosY'])
            list_com_z.append(self.df_vicon_filtered.loc[frame, 'COM_PosZ'])
            list_xcom_x.append(self.df_vicon_filtered.loc[frame, 'XCOM_Left_PosX'])
            list_xcom_y.append(self.df_vicon_filtered.loc[frame, 'XCOM_Left_PosY'])
            list_xcom_z.append(self.df_vicon_filtered.loc[frame, 'XCOM_Left_PosZ'])
            list_l.append(self.df_vicon_filtered.loc[frame, 'DistanceCOMLeftHeel'])
            heel_x.append(self.df_vicon_filtered.loc[frame, (preparation.name_left_heel_marker, 'PosX')] / 1000.)
            heel_y.append(self.df_vicon_filtered.loc[frame, (preparation.name_left_heel_marker, 'PosY')] / 1000.)
            heel_z.append(self.df_vicon_filtered.loc[frame, (preparation.name_left_heel_marker, 'PosZ')] / 1000.)
            BOSAP_toe_x.append(self.df_vicon_filtered.loc[frame, (preparation.name_left_toe_marker, 'PosX')] / 1000.)
            BOSAP_toe_y.append(self.df_vicon_filtered.loc[frame, (preparation.name_left_toe_marker, 'PosY')] / 1000.)
            BOSAP_toe_z.append(self.df_vicon_filtered.loc[frame, (preparation.name_left_toe_marker, 'PosZ')] / 1000.)
            MOS_AP_x.append(self.df_vicon_filtered.loc[frame, 'MOS_AP_Left_PosX'])
            MOS_AP_y.append(self.df_vicon_filtered.loc[frame, 'MOS_AP_Left_PosY'])
            MOS_AP_z.append(self.df_vicon_filtered.loc[frame, 'MOS_AP_Left_PosZ'])

            # Calculate BOSML_MTP_Joint - Extrapolated metatarsophalangeal joint  with Toe and Heel position - maximum lateral excursion
            vector_left_foot_toe = numpy.array(
                [self.df_vicon_filtered.loc[frame, (preparation.name_left_toe_marker, 'PosX')] / 1000.,
                 self.df_vicon_filtered.loc[frame, (preparation.name_left_toe_marker, 'PosY')] / 1000.,
                 self.df_vicon_filtered.loc[frame, (preparation.name_left_toe_marker, 'PosZ')] / 1000., ])

            vector_left_foot_heel = numpy.array(
                [self.df_vicon_filtered.loc[frame, (preparation.name_left_heel_marker, 'PosX')] / 1000.,
                 self.df_vicon_filtered.loc[frame, (preparation.name_left_heel_marker, 'PosY')] / 1000.,
                 self.df_vicon_filtered.loc[frame, (preparation.name_left_heel_marker, 'PosZ')] / 1000., ])

            norme_toe_to_heel = numpy.linalg.norm(vector_left_foot_toe - vector_left_foot_heel)

            vector_left_foot_BOSML = vector_left_foot_heel + ((vector_left_foot_toe - vector_left_foot_heel) * (13 / 19)) + numpy.dot(
                ((vector_left_foot_toe - vector_left_foot_heel) / norme_toe_to_heel),
                (numpy.array([0, 0, 1]))) * 30 * norme_toe_to_heel / 190

            BOSML_MTP_joint_x.append(vector_left_foot_BOSML[0])
            BOSML_MTP_joint_y.append(vector_left_foot_BOSML[1])
            BOSML_MTP_joint_z.append(vector_left_foot_BOSML[2])

            # Calculate the MOS_ML
            MOS_ML_x.append(vector_left_foot_BOSML[0] - self.df_vicon_filtered.loc[frame, 'XCOM_Left_PosX'])
            MOS_ML_y.append(vector_left_foot_BOSML[1] - self.df_vicon_filtered.loc[frame, 'XCOM_Left_PosY'])
            MOS_ML_z.append(vector_left_foot_BOSML[2] - self.df_vicon_filtered.loc[frame, 'XCOM_Left_PosZ'])

            # Quality of MOS
            if MOS_AP_y[-1].values[0] < 0:
                Quality_MOS_AP_Y.append('negative')
            elif MOS_AP_y[-1].values[0] > 0.7:
                Quality_MOS_AP_Y.append('outside')
            else:
                Quality_MOS_AP_Y.append('normal')

            if MOS_ML_x[-1].values[0] < 0:
                Quality_MOS_ML_X.append('negative')
            elif MOS_ML_x[-1].values[0] > 0.35:
                Quality_MOS_ML_X.append('outside')
            else:
                Quality_MOS_ML_X.append('normal')

        for frame in self.list_frame_HS_right:
            list_foot.append('Right')
            list_frame_heel_strike.append(frame)
            list_com_x.append(self.df_vicon_filtered.loc[frame, 'COM_PosX'])
            list_com_y.append(self.df_vicon_filtered.loc[frame, 'COM_PosY'])
            list_com_z.append(self.df_vicon_filtered.loc[frame, 'COM_PosZ'])
            list_xcom_x.append(self.df_vicon_filtered.loc[frame, 'XCOM_Right_PosX'])
            list_xcom_y.append(self.df_vicon_filtered.loc[frame, 'XCOM_Right_PosY'])
            list_xcom_z.append(self.df_vicon_filtered.loc[frame, 'XCOM_Right_PosZ'])
            list_l.append(self.df_vicon_filtered.loc[frame, 'DistanceCOMRightHeel'])
            heel_x.append(self.df_vicon_filtered.loc[frame, (preparation.name_right_heel_marker, 'PosX')] / 1000.)
            heel_y.append(self.df_vicon_filtered.loc[frame, (preparation.name_right_heel_marker, 'PosY')] / 1000.)
            heel_z.append(self.df_vicon_filtered.loc[frame, (preparation.name_right_heel_marker, 'PosZ')] / 1000.)
            BOSAP_toe_x.append(self.df_vicon_filtered.loc[frame, (preparation.name_right_toe_marker, 'PosX')] / 1000.)
            BOSAP_toe_y.append(self.df_vicon_filtered.loc[frame, (preparation.name_right_toe_marker, 'PosY')] / 1000.)
            BOSAP_toe_z.append(self.df_vicon_filtered.loc[frame, (preparation.name_right_toe_marker, 'PosZ')] / 1000.)
            MOS_AP_x.append(self.df_vicon_filtered.loc[frame, 'MOS_AP_Right_PosX'])
            MOS_AP_y.append(self.df_vicon_filtered.loc[frame, 'MOS_AP_Right_PosY'])
            MOS_AP_z.append(self.df_vicon_filtered.loc[frame, 'MOS_AP_Right_PosZ'])

            # Calculate BOSML_MTP_Joint - Extrapolated metatarsophalangeal joint  with Toe and Heel position - maximum lateral excursion
            vector_right_foot_toe = numpy.array(
                [self.df_vicon_filtered.loc[frame, (preparation.name_right_toe_marker, 'PosX')] / 1000.,
                 self.df_vicon_filtered.loc[frame, (preparation.name_right_toe_marker, 'PosY')] / 1000.,
                 self.df_vicon_filtered.loc[frame, (preparation.name_right_toe_marker, 'PosZ')] / 1000., ])

            vector_right_foot_heel = numpy.array(
                [self.df_vicon_filtered.loc[frame, (preparation.name_right_heel_marker, 'PosX')] / 1000.,
                 self.df_vicon_filtered.loc[frame, (preparation.name_right_heel_marker, 'PosY')] / 1000.,
                 self.df_vicon_filtered.loc[frame, (preparation.name_right_heel_marker, 'PosZ')] / 1000., ])

            norme_toe_to_heel = numpy.linalg.norm(vector_right_foot_toe - vector_right_foot_heel)

            vector_right_foot_BOSML = vector_right_foot_heel + ((vector_right_foot_toe - vector_right_foot_heel) * (13 / 19)) + numpy.dot(
                ((vector_right_foot_toe - vector_right_foot_heel) / norme_toe_to_heel),
                (numpy.array([0, 0, 1]))) * 30 * norme_toe_to_heel / 190

            BOSML_MTP_joint_x.append(vector_right_foot_BOSML[0])
            BOSML_MTP_joint_y.append(vector_right_foot_BOSML[1])
            BOSML_MTP_joint_z.append(vector_right_foot_BOSML[2])

            # Calculate the MOS_ML
            MOS_ML_x.append(vector_right_foot_BOSML[0] - self.df_vicon_filtered.loc[frame, 'XCOM_Right_PosX'])
            MOS_ML_y.append(vector_right_foot_BOSML[1] - self.df_vicon_filtered.loc[frame, 'XCOM_Right_PosY'])
            MOS_ML_z.append(vector_right_foot_BOSML[2] - self.df_vicon_filtered.loc[frame, 'XCOM_Right_PosZ'])

            # Quality of MOS
            if MOS_AP_y[-1].values[0] < 0:
                Quality_MOS_AP_Y.append('negative')
            elif MOS_AP_y[-1].values[0] > 0.7:
                Quality_MOS_AP_Y.append('outside')
            else:
                Quality_MOS_AP_Y.append('normal')

            if MOS_ML_x[-1].values[0] > 0:
                Quality_MOS_ML_X.append('negative')
            elif MOS_ML_x[-1].values[0] < -0.35:
                Quality_MOS_ML_X.append('outside')
            else:
                Quality_MOS_ML_X.append('normal')

        # Add these variables in the result file margin_stability_variables.csv
        dic_margin_stability_variables = {
            'Foot': list_foot,
            'Frame_Heel_Strike': list_frame_heel_strike,
            'COM_X': list_com_x,
            'COM_Y': list_com_y,
            'COM_Z': list_com_z,
            'XCOM_X': list_xcom_x,
            'XCOM_Y': list_xcom_y,
            'XCOM_Z': list_xcom_z,
            'L': list_l,
            'Heel_X': heel_x,
            'Heel_Y': heel_y,
            'Heel_Z': heel_z,
            'BOSAP_Toe_X': BOSAP_toe_x,
            'BOSAP_Toe_Y': BOSAP_toe_y,
            'BOSAP_Toe_Z': BOSAP_toe_z,
            'BOSML_MTP_joint_X': BOSML_MTP_joint_x,
            'BOSML_MTP_joint_Y': BOSML_MTP_joint_y,
            'BOSML_MTP_joint_Z': BOSML_MTP_joint_z,
            'MOS_AP_X': MOS_AP_x,
            'MOS_AP_Y': MOS_AP_y,
            'MOS_AP_Z': MOS_AP_z,
            'MOS_ML_X': MOS_ML_x,
            'MOS_ML_Y': MOS_ML_y,
            'MOS_ML_Z': MOS_ML_z,
            'Quality_MOS_AP_Y': Quality_MOS_AP_Y,
            'Quality_MOS_ML_X': Quality_MOS_ML_X
        }

        # The margin_stability_variables dataframe
        df_margin_stability_variables = pandas.DataFrame()
        for key, item in dic_margin_stability_variables.items():
            df_margin_stability_variables[key] = np.array(item)

        # Save margin_stability_variables dataframe to csv file
        df_margin_stability_variables.to_csv(self.trial_folder_path + 'Result/' + self.config.name + '_margin_stability_variables.csv', sep=';')

        # Calculate the general variables
        # For MOS_AP
        mean_MOS_AP_y = []
        std_MOS_AP_y = []
        max_MOS_AP_y = []
        min_MOS_AP_y = []
        per_forward_MOS_AP = []

        mean_MOS_AP_y.append(np.mean(MOS_AP_y))
        std_MOS_AP_y.append(np.std(MOS_AP_y))
        max_MOS_AP_y.append(np.max(MOS_AP_y))
        min_MOS_AP_y.append(np.min(MOS_AP_y))

        forward_MOS_AP_count = 0
        for y in MOS_AP_y:
            if float(y) > 0:
                forward_MOS_AP_count += 1

        per_forward_MOS_AP.append(forward_MOS_AP_count / len(MOS_AP_y))

        mean_MOS_ML_x = []
        std_MOS_ML_x = []
        max_MOS_ML_x = []
        min_MOS_ML_x = []
        per_inside_MOS_ML = []

        MOS_ML_x_converted = MOS_ML_x
        for i in range(len(MOS_ML_x_converted)):
            if list_foot[i] == 'Right':
                MOS_ML_x_converted[i] = MOS_ML_x_converted[i] * (-1)

        mean_MOS_ML_x.append(np.mean(MOS_ML_x_converted))
        std_MOS_ML_x.append(np.std(MOS_ML_x_converted))
        max_MOS_ML_x.append(np.max(MOS_ML_x_converted))
        min_MOS_ML_x.append(np.min(MOS_ML_x_converted))

        inside_MOS_ML_count = 0
        for x in MOS_ML_x_converted:
            if float(x) > 0:
                inside_MOS_ML_count += 1

        per_inside_MOS_ML.append(inside_MOS_ML_count / len(MOS_ML_x_converted))

        # Data quality
        number_of_outliers_MOS_AP_Y = Quality_MOS_AP_Y.count('negative') + Quality_MOS_AP_Y.count('outside')
        number_of_outliers_MOS_ML_X = Quality_MOS_ML_X.count('negative') + Quality_MOS_ML_X.count('outside')

        # Root mean square of head's acceleration and plevis' acceleration
        # Head : barycentre du Rigidbody
        # Changed used markers from the Rigidbody "Head" to the "HeadCalibrated"
        #df_head = self.df_vicon_filtered.loc[:, ('Head', preparation.position_iterables)] / 1000.
        #head_positions_X = df_head.loc[:, ('Head', 'PosX')]
        df_head = self.df_vicon_filtered.loc[:, ('HeadCalibrated', preparation.position_iterables)] / 1000.
        head_positions_X = df_head.loc[:, ('HeadCalibrated', 'PosX')]
        head_acceleration_X = (head_positions_X.diff().fillna(0.) / (1 / 120.)).diff().fillna(0.) / (1 / 120.)

        acc_RMS_head_X = np.sqrt((head_acceleration_X ** 2).mean(axis=0))

        # Same change as line 677
        #head_positions_Y = df_head.loc[:, ('Head', 'PosY')]
        head_positions_Y = df_head.loc[:, ('HeadCalibrated', 'PosY')]
        head_acceleration_Y = (head_positions_Y.diff().fillna(0.) / (1 / 120.)).diff().fillna(0.) / (1 / 120.)
        acc_RMS_head_Y = np.sqrt((head_acceleration_Y ** 2).mean(axis=0))

        acc_RMS_pelvis_X = np.sqrt((self.df_vicon_filtered['AccelerationCOM_PosX'] ** 2).mean(axis=0))
        acc_RMS_pelvis_Y = np.sqrt((self.df_vicon_filtered['AccelerationCOM_PosY'] ** 2).mean(axis=0))

        # Conclude on the dictionary of margin stability general variables
        dic_margin_stability_general_variables = {
            'MEAN_MOS_AP_Y': mean_MOS_AP_y,
            'STD_MOS_AP_Y': std_MOS_AP_y,
            'MAX_MOS_AP_Y': max_MOS_AP_y,
            'MIN_MOS_AP_Y': min_MOS_AP_y,
            'PER_Forward_MOS_AP': per_forward_MOS_AP,
            'MEAN_MOS_ML_X': mean_MOS_ML_x,
            'STD_MOS_ML_X': std_MOS_ML_x,
            'MAX_MOS_ML_X': max_MOS_ML_x,
            'MIN_MOS_ML_X': min_MOS_ML_x,
            'PER_Inside_MOS_ML': per_inside_MOS_ML,
            'Number_of_outliers_MOS_AP_Y': number_of_outliers_MOS_AP_Y,
            'Number_of_outliers_MOS_ML_X': number_of_outliers_MOS_ML_X,
            'Acc_RMS_Head_X': acc_RMS_head_X,
            'Acc_RMS_Head_Y': acc_RMS_head_Y,
            'Acc_RMS_Pelvis_X': acc_RMS_pelvis_X,
            'Acc_RMS_Pelvis_Y': acc_RMS_pelvis_Y,
        }

        # The dic_margin_stability_general_variables dataframe
        df_margin_stability_general_variables = pandas.DataFrame()
        for key, item in dic_margin_stability_general_variables.items():
            df_margin_stability_general_variables[key] = item

        # print('dataframe:::::::')
        # print(df_margin_stability_general_variables)

        # Save margin_stability_variables dataframe to csv file
        df_margin_stability_general_variables.to_csv(self.trial_folder_path + 'Result/' + self.config.name + '_margin_stability_general_variables.csv', sep=';')

    # Calculate the step length and step width for each step
    def calculate_step_length_width(self):
        # List of frames combining the frames of heel strike of both feet
        list_frame_HS = sorted(self.list_frame_HS_right + self.list_frame_HS_left)
        # print('list_frame_HS_right:')
        # print(self.list_frame_HS_right)
        # print('list_frame_HS_left:')
        # print(self.list_frame_HS_left)
        # print('list_frame_HS')
        # print(list_frame_HS)
        list_step_length = []
        list_step_width = []

        # Starting from the min frame of heel strike, find the next frame of heel strike
        # If the interval between 2 heel strike is smaller than 400 frames, calculate the step length and step width
        # Otherwise, it means that the 2 heel strikes happen during the PWS while the participant is turning over, so ignore it
        for i in range(len(list_frame_HS) - 1):
            if list_frame_HS[i + 1] - list_frame_HS[i] > 400:
                # print('skip i: ' + str(list_frame_HS[i]) + ', i+1: ' + str(list_frame_HS[i+1]))
                continue
            # print('i: ' + str(list_frame_HS[i]) + ', i+1: ' + str(list_frame_HS[i+1]))

            # If the frame of heel strike i is in the list_frame_HS_left, then the frame of heel strike i+1 is in the list_frame_HS_right
            if list_frame_HS[i] in self.list_frame_HS_left:
                list_step_length.append(abs(self.df_vicon_filtered.loc[list_frame_HS[i], (preparation.name_left_heel_marker, 'PosY')] - self.df_vicon_filtered.loc[
                    list_frame_HS[i + 1], (preparation.name_right_heel_marker, 'PosY')]) / 1000)
                list_step_width.append(abs(self.df_vicon_filtered.loc[list_frame_HS[i], (preparation.name_left_heel_marker, 'PosX')] - self.df_vicon_filtered.loc[
                    list_frame_HS[i + 1], (preparation.name_right_heel_marker, 'PosX')]) / 1000)
            else:
                list_step_length.append(abs(self.df_vicon_filtered.loc[list_frame_HS[i], (preparation.name_right_heel_marker, 'PosY')] - self.df_vicon_filtered.loc[
                    list_frame_HS[i + 1], (preparation.name_left_heel_marker, 'PosY')]) / 1000)
                list_step_width.append(abs(self.df_vicon_filtered.loc[list_frame_HS[i], (preparation.name_right_heel_marker, 'PosX')] - self.df_vicon_filtered.loc[
                    list_frame_HS[i + 1], (preparation.name_left_heel_marker, 'PosX')]) / 1000)

        # Write to file
        csv_path = self.trial_folder_path + 'Result/' + self.config.name + '_step_length_step_width.csv'
        with open(csv_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=';')
            writer.writerow(['index', 'step_length', 'step_width'])
            for i in range(len(list_step_length)):
                writer.writerow([str(i + 1), str(list_step_length[i]), str(list_step_width[i])])
            writer.writerow(['mean', str(np.mean(list_step_length)), str(np.mean(list_step_width))])
            writer.writerow(['standard_deviation', (np.std(list_step_length)), str(np.std(list_step_width))])

    # Get the list of both foot's stable positions
    def calculate_foot_stable_positions(self):
        # Left foot
        # Get the list of index of frame where the left foot is stable on the ground
        _, _, _, list_stable_left = analysis.one_side_steps_detection_low_precision(
            self.df_vicon_filtered.loc[:, preparation.name_left_toe_marker].to_numpy())

        # Save the list of each steps toe and heel coordinate in the maze image

        # The Python Imaging Library uses a Cartesian pixel coordinate system, with (0,0) in the upper left corner
        # Main list of dataframe to save the coordinate
        maze_coordinate_dataframes = pandas.DataFrame({
            'foot_side': [],
            'toe_position_maze_x': [],
            'toe_position_maze_y': [],
            'heel_position_maze_x': [],
            'heel_position_maze_y': [],
            'center_position_maze_x': [],
            'center_position_maze_y': [],
        })

        # For each stable step of left feet
        last_toe_position_vector2 = numpy.array((0, 0))

        for i in range(0, len(list_stable_left)):
            toe_position_vicon_x = self.df_vicon_filtered.loc[
                list_stable_left[i], (preparation.name_left_toe_marker, 'PosX')]
            toe_position_vicon_y = self.df_vicon_filtered.loc[
                list_stable_left[i], (preparation.name_left_toe_marker, 'PosY')]
            heel_position_vicon_x = self.df_vicon_filtered.loc[
                list_stable_left[i], (preparation.name_left_heel_marker, 'PosX')]
            heel_position_vicon_y = self.df_vicon_filtered.loc[
                list_stable_left[i], (preparation.name_left_heel_marker, 'PosY')]

            toe_position_vector2 = numpy.array((toe_position_vicon_x, toe_position_vicon_y))

            # As sometime there are two steps of the same foot too close to each other we add this condition
            if numpy.linalg.norm(toe_position_vector2 - last_toe_position_vector2) > (
                    self.subject_step_length * 500) or i == 0:
                last_toe_position_vector2 = toe_position_vector2

                # According to the excel: https://drive.google.com/file/d/1kcyFfKPf2C_4XzIlRgOApPGjxqwSc1ga/view?usp=sharing
                toe_position_maze_x = (toe_position_vicon_y - min_vicon_y_mm) * pixel_per_meter / 1000
                toe_position_maze_y = (toe_position_vicon_x - min_vicon_x_mm) * pixel_per_meter / 1000
                heel_position_maze_x = (heel_position_vicon_y - min_vicon_y_mm) * pixel_per_meter / 1000
                heel_position_maze_y = (heel_position_vicon_x - min_vicon_x_mm) * pixel_per_meter / 1000
                center_position_maze_x = (toe_position_maze_x + heel_position_maze_x) / 2
                center_position_maze_y = (toe_position_maze_y + heel_position_maze_y) / 2

                # Add condition for the step in the maze image
                # If toe or heel is outside the image, ignore it #
                if 15 <= toe_position_maze_x <= number_pixel_x and 0 <= toe_position_maze_y <= number_pixel_y and 15 <= heel_position_maze_x <= number_pixel_x and 0 <= heel_position_maze_y <= number_pixel_y:
                    left_step_coordinate_dataframes = pandas.DataFrame({
                        'foot_side': ['left'],
                        'toe_position_maze_x': [toe_position_maze_x],
                        'toe_position_maze_y': [toe_position_maze_y],
                        'heel_position_maze_x': [heel_position_maze_x],
                        'heel_position_maze_y': [heel_position_maze_y],
                        'center_position_maze_x': [center_position_maze_x],
                        'center_position_maze_y': [center_position_maze_y],
                    })

                    maze_coordinate_dataframes = pandas.concat([maze_coordinate_dataframes, left_step_coordinate_dataframes], ignore_index=True)

        # Delete the original csv file in the folder Data
        left_foot_step_positions_maze_path = self.trial_folder_path + 'Vicon/' + self.config.name + '_left_foot_step_maze.csv'
        if os.path.exists(left_foot_step_positions_maze_path):
            os.remove(left_foot_step_positions_maze_path)

        # Right foot
        # Get the list of index of frame where the right foot is stable  on the ground
        _, _, _, list_stable_right = analysis.one_side_steps_detection_low_precision(
            self.df_vicon_filtered.loc[:, preparation.name_right_toe_marker].to_numpy())

        # For each stable step of right feet
        last_toe_position_vector2 = numpy.array((0, 0))

        for i in range(0, len(list_stable_right)):
            toe_position_vicon_x = self.df_vicon_filtered.loc[
                list_stable_right[i], (preparation.name_right_toe_marker, 'PosX')]
            toe_position_vicon_y = self.df_vicon_filtered.loc[
                list_stable_right[i], (preparation.name_right_toe_marker, 'PosY')]
            heel_position_vicon_x = self.df_vicon_filtered.loc[
                list_stable_right[i], (preparation.name_right_heel_marker, 'PosX')]
            heel_position_vicon_y = self.df_vicon_filtered.loc[
                list_stable_right[i], (preparation.name_right_heel_marker, 'PosY')]

            toe_position_vector2 = numpy.array((toe_position_vicon_x, toe_position_vicon_y))

            # As sometime there are two steps of the same foot too close to each other we add this condition
            if numpy.linalg.norm(toe_position_vector2 - last_toe_position_vector2) > (
                    self.subject_step_length * 500) or i == 0:
                last_toe_position_vector2 = toe_position_vector2

                # According to the excel: https://drive.google.com/file/d/1kcyFfKPf2C_4XzIlRgOApPGjxqwSc1ga/view?usp=sharing
                toe_position_maze_x = (toe_position_vicon_y - min_vicon_y_mm) * pixel_per_meter / 1000
                toe_position_maze_y = (toe_position_vicon_x - min_vicon_x_mm) * pixel_per_meter / 1000
                heel_position_maze_x = (heel_position_vicon_y - min_vicon_y_mm) * pixel_per_meter / 1000
                heel_position_maze_y = (heel_position_vicon_x - min_vicon_x_mm) * pixel_per_meter / 1000
                center_position_maze_x = (toe_position_maze_x + heel_position_maze_x) / 2
                center_position_maze_y = (toe_position_maze_y + heel_position_maze_y) / 2

                # if 0 <= toe_position_maze_x <= 2390 and 0 <= toe_position_maze_y <= 1080 and 0 <= heel_position_maze_x <= 2390 and 0 <= heel_position_maze_y <= 1080:
                if 15 <= toe_position_maze_x <= number_pixel_x and 0 <= toe_position_maze_y <= number_pixel_y and 15 <= heel_position_maze_x <= number_pixel_x and 0 <= heel_position_maze_y <= number_pixel_y:
                    right_step_coordinate_dataframes = pandas.DataFrame({
                        'foot_side': ['right'],
                        'toe_position_maze_x': [toe_position_maze_x],
                        'toe_position_maze_y': [toe_position_maze_y],
                        'heel_position_maze_x': [heel_position_maze_x],
                        'heel_position_maze_y': [heel_position_maze_y],
                        'center_position_maze_x': [center_position_maze_x],
                        'center_position_maze_y': [center_position_maze_y],
                    })

                    maze_coordinate_dataframes = pandas.concat([maze_coordinate_dataframes, right_step_coordinate_dataframes], ignore_index=True)

        # Delete the original csv file in the folder Data
        right_foot_step_positions_maze_path = self.trial_folder_path + 'Vicon/' + self.config.name + '_right_foot_step_maze.csv'
        if os.path.exists(right_foot_step_positions_maze_path):
            os.remove(right_foot_step_positions_maze_path)

        # Export to a csv file
        foot_step_positions_maze_path = self.trial_folder_path + 'Vicon/' + self.config.name + '_foot_step_maze.csv'
        maze_coordinate_dataframes.to_csv(foot_step_positions_maze_path, sep=';', encoding='utf-8')

    # Draw the coordinate of left and right foot stable position in the maze image
    def draw_two_feet_stable_positions(self):
        # Read the left and right foot maze coordinate csv
        foot_step_positions_maze_path = self.trial_folder_path + 'Vicon/' + self.config.name + '_foot_step_maze.csv'
        df_foot_step_positions_maze = pandas.read_csv(foot_step_positions_maze_path, sep=';', index_col=[0])
        left_foot_maze_coordinate_dataframe = df_foot_step_positions_maze.loc[df_foot_step_positions_maze['foot_side'] == 'left']
        right_foot_maze_coordinate_dataframe = df_foot_step_positions_maze.loc[df_foot_step_positions_maze['foot_side'] == 'right']

        # Get the image of the current trial using the configuration file and copy it in the folder named 'OriginalMazeImage.png'
        config_complex_dataframe = pandas.read_csv(session.config_file_path, sep=';')
        # print(config_complex_dataframe)

        step_length_folder_name = 'median_' + self.subject_step_length.__str__()

        foot_size_folder_name = 'foot_' + int(self.subject_foot_size).__str__()

        # print(config_complex_dataframe.loc[self.config.index - 1])

        if config_complex_dataframe.at[self.config.index - 1, 'FirstStepLeft'] == True:
            first_step_folder_name = 'first_step_left'
        else:
            first_step_folder_name = 'first_step_right'

        # print('index: ' + (self.config.index - 1).__str__() + ' index maze: ' + config_complex_dataframe.at[self.config.index, 'IndexMaze'].__str__())
        maze_index = config_complex_dataframe.at[self.config.index - 1, 'IndexMaze']

        # The path of the maze image in the folder
        maze_name = step_length_folder_name + '_' + foot_size_folder_name + '_' + first_step_folder_name + '_' + maze_index.__str__()
        base_maze_image_path = maze_image_folder_path + '/' + step_length_folder_name + '/' + foot_size_folder_name + '/' + first_step_folder_name + '/' + maze_name + '_extra.png'
        # print('maze_image_path: ' + base_maze_image_path)
        target_maze_image_path = self.trial_folder_path + 'Vicon/OriginalMazeImage_' + maze_name + '.png'

        # Copy the step length and step width in maze to the result folder
        step_length_width_path = base_maze_image_path.replace('_extra.png', '.csv')
        if os.path.exists(step_length_width_path):
            shutil.copy(step_length_width_path, self.trial_folder_path + 'Result/' + step_length_width_path.split('/')[-1])

        # Make a copy in the Vicon folder
        shutil.copy(base_maze_image_path, target_maze_image_path)

        # Get the feet image according to the feet size

        # left_foot_image_path = os.getcwd() + '/ComplexResource/LeftFootImage47.png'
        left_foot_image_path = os.getcwd() + '/ComplexResource/LeftFootImage' + str(int(self.subject_foot_size)) + '.png'
        # right_foot_image_path = os.getcwd() + '/ComplexResource/RightFootImage47.png'
        right_foot_image_path = os.getcwd() + '/ComplexResource/RightFootImage' + str(int(self.subject_foot_size)) + '.png'

        original_image = Image.open(target_maze_image_path).convert("RGBA")
        left_foot_image = Image.open(left_foot_image_path).convert("RGBA")
        right_foot_image = Image.open(right_foot_image_path).convert("RGBA")

        # test_image = left_foot_image.rotate(45, center=(0, 75), expand=True)
        # test_image_path = 'C:/Users/Utilisateur/Desktop/Pretest_Complex_20112020/P_001_Session1/Data/Trial_5_ComplexT_STV/Vicon/TestImage.png'
        # test_image.save(test_image_path)

        # Paste the foot image onto the original image
        target_image = original_image.copy()

        # For the left footstep
        for i in left_foot_maze_coordinate_dataframe.index:
            left_toe_maze_x = left_foot_maze_coordinate_dataframe.at[i, 'toe_position_maze_x']
            left_toe_maze_y = left_foot_maze_coordinate_dataframe.at[i, 'toe_position_maze_y']
            left_heel_maze_x = left_foot_maze_coordinate_dataframe.at[i, 'heel_position_maze_x']
            left_heel_maze_y = left_foot_maze_coordinate_dataframe.at[i, 'heel_position_maze_y']

            # Calculate the angle and rotate
            rotation_angle = math.degrees(numpy.arctan((left_heel_maze_y - left_toe_maze_y) / (left_heel_maze_x - left_toe_maze_x)))
            left_foot_image_rotated = left_foot_image.rotate(-rotation_angle)

            # Calculate the feet size and resize
            # ratio_resize = self.subject_foot_size / 47
            # width, height = left_foot_image_rotated.size
            # left_foot_image_resized = left_foot_image_rotated.resize((int(width * ratio_resize), int(height * ratio_resize)))

            # Calculate the paste position of the foot image (center position - 150)
            left_foot_center_tuple = (
                int(left_foot_maze_coordinate_dataframe.at[i, 'center_position_maze_x'] - 150),
                int(left_foot_maze_coordinate_dataframe.at[i, 'center_position_maze_y'] - 150))

            target_image.paste(left_foot_image_rotated, left_foot_center_tuple, mask=left_foot_image_rotated)

        # For the right footstep
        for i in right_foot_maze_coordinate_dataframe.index:
            right_toe_maze_x = right_foot_maze_coordinate_dataframe.at[i, 'toe_position_maze_x']
            right_toe_maze_y = right_foot_maze_coordinate_dataframe.at[i, 'toe_position_maze_y']
            right_heel_maze_x = right_foot_maze_coordinate_dataframe.at[i, 'heel_position_maze_x']
            right_heel_maze_y = right_foot_maze_coordinate_dataframe.at[i, 'heel_position_maze_y']

            # Calculate the angle
            rotation_angle = math.degrees(numpy.arctan((right_heel_maze_y - right_toe_maze_y) / (right_heel_maze_x - right_toe_maze_x)))
            right_foot_image_rotated = right_foot_image.rotate(-rotation_angle)

            # Calculate the feet size and resize
            # ratio_resize = self.subject_foot_size / 47
            # width, height = right_foot_image_rotated.size
            # right_foot_image_resized = right_foot_image_rotated.resize((int(width * ratio_resize), int(height * ratio_resize)))

            # Calculate the center position of the foot image
            right_foot_center_tuple = (
                int(right_foot_maze_coordinate_dataframe.at[i, 'center_position_maze_x'] - 150),
                int(right_foot_maze_coordinate_dataframe.at[i, 'center_position_maze_y'] - 150))

            target_image.paste(right_foot_image_rotated, right_foot_center_tuple, mask=right_foot_image_rotated)

        final_image = Image.blend(target_image, original_image, alpha=0.5)

        analyse_maze_image_path = self.trial_folder_path + 'Vicon/AnalyseMazeImage.png'
        final_image.save(analyse_maze_image_path, "PNG")


# The TrialConfig class for PWS calibration
class PWSCalibrationTrialConfig(preparation.TrialConfig):
    def __init__(self, trial_, index_):
        # Call the parent method
        super().__init__()

        # The TrialData instance
        self.trial = trial_
        self.index = index_
        self.category = 'pws'

        # The name of the trial in the PWSCalibration folder, for example, Calibration_1
        self.name = 'Calibration_' + str(self.index)

        self.list_config_elements = {}


class ComplexTerrainTrialConfig(preparation.TrialConfig):
    def __init__(self, trial_, index_, category_, light_condition_lux_, contrast_, windows_visibility_,
                 task_type_, index_in_phase_, first_step_left_, index_maze_, unity_color_):
        # Call the parent method
        super().__init__()

        # The TrialData instance
        self.trial = trial_
        self.index = index_
        self.category = category_
        self.light_condition_lux = light_condition_lux_
        self.contrast = contrast_
        self.windows_visibility = windows_visibility_
        self.task_type = task_type_
        self.index_in_phase = index_in_phase_
        self.first_step_left = first_step_left_
        self.index_maze = index_maze_
        self.unity_color = unity_color_

        # The name of the trial in the Data folder, for example, Trial_6_CrossingET_STV
        self.name = 'Trial_' + str(self.index) + '_' + self.category + '_' + self.task_type

        # The list of config elements, used for exporting result
        self.list_config_elements = {'session_name': self.trial.session.name,
                                     'trial_name': self.name,
                                     'index': self.index,
                                     'category': self.category,
                                     'light_condition_lux': self.light_condition_lux,
                                     'contrast': self.contrast,
                                     'windows_visibility': self.windows_visibility,
                                     'task_type': self.task_type,
                                     'index_in_phase': self.index_in_phase,
                                     'first_step_left': self.first_step_left,
                                     'index_maze': self.index_maze,
                                     'unity_color': self.unity_color
                                     }


# The instance that contains all results for a trial
class ComplexTerrainTrialResult(preparation.TrialResult):
    def __init__(self, trial_):
        # Call the parent method
        super().__init__(trial_)

        self.dic_complex_variables = None

    # Generate trial result data for global variables and spatial temporal variables for each crossing
    def generate_trial_result(self):
        # Call the parent method
        super().generate_trial_result()

        ##### Image analyse

        # Get the number of placement of maze
        path_folder_result = self.trial.trial_folder_path + 'Result/'
        list_file = os.listdir(path_folder_result)
        for path_file in list_file:
            if 'median' in path_file:
                path_maze_info = self.trial.trial_folder_path + 'Result/' + path_file

        df_maze_info = pandas.read_csv(path_maze_info, sep=';', index_col=[0])
        nb_placement_maze = len(df_maze_info.index) - 1

        # Get the number of placement of feet within the 2390 * 1080
        foot_step_positions_maze_path = self.trial.trial_folder_path + 'Vicon/' + self.trial.config.name + '_foot_step_maze.csv'
        df_foot_step_positions_maze = pandas.read_csv(foot_step_positions_maze_path, sep=';', index_col=[0])
        nb_placement_trial = len(df_foot_step_positions_maze.index)
        analyse_maze_image_path = self.trial.trial_folder_path + 'Vicon/AnalyseMazeImage.png'

        ##### Get the number of light green pixel and its contour image
        image = cv2.imread(analyse_maze_image_path)

        original = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # According to the analyse (hMin = 0 , sMin = 127, vMin = 0), (hMax = 179 , sMax = 196, vMax = 255)
        lower = numpy.array([0, 127, 0], dtype="uint8")
        upper = numpy.array([179, 196, 255], dtype="uint8")
        mask = cv2.inRange(image, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contour_area_list = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_area_list = contour_area_list[0] if len(contour_area_list) == 2 else contour_area_list[1]

        number_of_pixel = 0
        for c in contour_area_list:
            number_of_pixel += cv2.contourArea(c)
            cv2.drawContours(original, [c], 0, (0, 0, 0), 2)

        cv2.imwrite(self.trial.trial_folder_path + 'Vicon/ContourMazeImage1.png', original)

        nb_light_green_pixel = number_of_pixel

        # Calculate number of light green pixel per step
        nb_light_green_pixel_per_step = nb_light_green_pixel / nb_placement_trial
        #####

        ##### Get the number of brown pixel and its contour image
        image = cv2.imread(analyse_maze_image_path)

        original = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # According to the analyse (hMin = 0 , sMin = 127, vMin = 0), (hMax = 179 , sMax = 196, vMax = 255)
        lower = numpy.array([15, 9, 0], dtype="uint8")
        upper = numpy.array([179, 235, 225], dtype="uint8")
        mask = cv2.inRange(image, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contour_area_list = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_area_list = contour_area_list[0] if len(contour_area_list) == 2 else contour_area_list[1]

        number_of_pixel = 0
        for c in contour_area_list:
            number_of_pixel += cv2.contourArea(c)
            cv2.drawContours(original, [c], 0, (0, 0, 0), 2)

        cv2.imwrite(self.trial.trial_folder_path + 'Vicon/ContourMazeImage2.png', original)

        nb_brown_pixel = number_of_pixel

        # Calculate number of light green pixel per step
        nb_brown_pixel_per_step = nb_brown_pixel / nb_placement_trial
        #####

        ##### Get the number of feet form pixel and its contour image
        image = cv2.imread(analyse_maze_image_path)

        original = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # According to the analyse (hMin = 0 , sMin = 46, vMin = 0), (hMax = 179 , sMax = 255, vMax = 255)
        lower = numpy.array([15, 9, 0], dtype="uint8")
        upper = numpy.array([179, 255, 255], dtype="uint8")
        mask = cv2.inRange(image, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contour_area_list = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_area_list = contour_area_list[0] if len(contour_area_list) == 2 else contour_area_list[1]

        number_of_pixel = 0
        for c in contour_area_list:
            number_of_pixel += cv2.contourArea(c)
            cv2.drawContours(original, [c], 0, (0, 0, 0), 2)

        cv2.imwrite(self.trial.trial_folder_path + 'Vicon/ContourMazeImage3.png', original)

        nb_feet_form_pixel = number_of_pixel

        # Calculate number of feet_form_pixel per step
        nb_feet_form_pixel_per_step = nb_feet_form_pixel / nb_placement_trial
        #####

        # Calculate the mean percentage of light green pixel in feet form pixel
        mean_percentage_light_green_pixel = nb_light_green_pixel_per_step / nb_feet_form_pixel_per_step

        # Calculate the mean percentage of brown pixel in feet form pixel
        mean_percentage_brown_pixel = nb_brown_pixel_per_step / nb_feet_form_pixel_per_step

        # Calculate for each step
        image = cv2.imread(analyse_maze_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        for i in df_foot_step_positions_maze.index:
            # Crop the image according to the foot position ( 117 pixel +/- for x axis and 60 pixel +/- for y axis)
            center_position_maze_x = df_foot_step_positions_maze.at[i, 'center_position_maze_x']
            center_position_maze_y = df_foot_step_positions_maze.at[i, 'center_position_maze_y']
            y_min = max(int(center_position_maze_y - 60), 0)
            y_max = min(int(center_position_maze_y + 60), number_pixel_y)
            x_min = max(int(center_position_maze_x - 117), 0)
            x_max = min(int(center_position_maze_x + 117), number_pixel_x)
            image_crop = image[y_min: y_max, x_min: x_max]

            # Analyse the step_brown_pixel
            lower = numpy.array([15, 9, 0], dtype="uint8")
            upper = numpy.array([179, 235, 225], dtype="uint8")
            mask = cv2.inRange(image_crop, lower, upper)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            contour_area_list = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_area_list = contour_area_list[0] if len(contour_area_list) == 2 else contour_area_list[1]

            step_brown_pixel = 0
            for c in contour_area_list:
                step_brown_pixel += cv2.contourArea(c)

            # Analyse the step_feet_form_pixel
            lower = numpy.array([15, 9, 0], dtype="uint8")
            upper = numpy.array([179, 255, 255], dtype="uint8")
            mask = cv2.inRange(image_crop, lower, upper)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            contour_area_list = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_area_list = contour_area_list[0] if len(contour_area_list) == 2 else contour_area_list[1]

            step_feet_form_pixel = 0
            for c in contour_area_list:
                step_feet_form_pixel += cv2.contourArea(c)

            # Analyse the step_percentage_brown_pixel
            step_percentage_brown_pixel = step_brown_pixel / step_feet_form_pixel

            df_foot_step_positions_maze.at[i, 'step_brown_pixel'] = step_brown_pixel
            df_foot_step_positions_maze.at[i, 'step_feet_form_pixel'] = step_feet_form_pixel
            df_foot_step_positions_maze.at[i, 'step_percentage_brown_pixel'] = step_percentage_brown_pixel

        # Write the csv file
        df_foot_step_positions_maze.to_csv(foot_step_positions_maze_path, sep=';')

        df_foot_step_positions_maze = df_foot_step_positions_maze.sort_values(by='center_position_maze_x', ascending=False)
        df_foot_step_positions_maze = df_foot_step_positions_maze.reset_index()

        list_percentage_brown_pixel = df_foot_step_positions_maze['step_percentage_brown_pixel'].to_numpy()
        # print('list_percentage_brown_pixel', list_percentage_brown_pixel)
        #####

        # Calculate the percentage of brown pixels for the 2 first steps
        if len(list_percentage_brown_pixel) >= 2:
            mean_percentage_brown_2_first_steps = statistics.mean([list_percentage_brown_pixel[0], list_percentage_brown_pixel[1]])
        else:
            mean_percentage_brown_2_first_steps = np.nan

        # Calculate the percentage of brown pixels for the 2 last steps
        if len(list_percentage_brown_pixel) >= 2:
            mean_percentage_brown_2_last_steps = statistics.mean([list_percentage_brown_pixel[-2], list_percentage_brown_pixel[-1]])
        else:
            mean_percentage_brown_2_last_steps = np.nan

        # Calculate the percentage of brown pixels for the steps 3, 4 and 5
        if len(list_percentage_brown_pixel) >= 5:
            mean_percentage_brown_345_steps = statistics.mean([list_percentage_brown_pixel[2], list_percentage_brown_pixel[3], list_percentage_brown_pixel[4]])
        else:
            mean_percentage_brown_345_steps = np.nan

        ##### Get the analysed variables from margin_stability_general_variables
        margin_stability_file_path = self.trial.trial_folder_path + 'Result/' + self.trial.config.name + '_margin_stability_general_variables.csv'
        df_margin_stability_general_variables = pandas.read_csv(margin_stability_file_path, sep=';')
        mean_MOS_AP_y = df_margin_stability_general_variables.at[0, 'MEAN_MOS_AP_Y']
        std_MOS_AP_y = df_margin_stability_general_variables.at[0, 'STD_MOS_AP_Y']
        max_MOS_AP_y = df_margin_stability_general_variables.at[0, 'MAX_MOS_AP_Y']
        min_MOS_AP_y = df_margin_stability_general_variables.at[0, 'MIN_MOS_AP_Y']
        per_forward_MOS_AP = df_margin_stability_general_variables.at[0, 'PER_Forward_MOS_AP']
        mean_MOS_ML_x = df_margin_stability_general_variables.at[0, 'MEAN_MOS_ML_X']
        std_MOS_ML_x = df_margin_stability_general_variables.at[0, 'STD_MOS_ML_X']
        max_MOS_ML_x = df_margin_stability_general_variables.at[0, 'MAX_MOS_ML_X']
        min_MOS_ML_x = df_margin_stability_general_variables.at[0, 'MIN_MOS_ML_X']
        per_inside_MOS_ML = df_margin_stability_general_variables.at[0, 'PER_Inside_MOS_ML']
        number_of_outliers_MOS_AP_Y = df_margin_stability_general_variables.at[0, 'Number_of_outliers_MOS_AP_Y']
        number_of_outliers_MOS_ML_X = df_margin_stability_general_variables.at[0, 'Number_of_outliers_MOS_ML_X']
        acc_RMS_head_X = df_margin_stability_general_variables.at[0, 'Acc_RMS_Head_X']
        acc_RMS_head_Y = df_margin_stability_general_variables.at[0, 'Acc_RMS_Head_Y']
        acc_RMS_pelvis_X = df_margin_stability_general_variables.at[0, 'Acc_RMS_Pelvis_X']
        acc_RMS_pelvis_Y = df_margin_stability_general_variables.at[0, 'Acc_RMS_Pelvis_Y']

        ##### For one trial, check if there are some folders with the same in the folder Archived
        session_folder_path = self.trial.session.session_folder_path
        session_archived_folder_path = session_folder_path + '/Archived'
        trial_name = 'Trial_' + str(self.trial.config.index) + '_'

        number_trial_archived_found = 0
        for path in os.listdir(session_archived_folder_path):
            if trial_name in path:
                number_trial_archived_found += 1

        number_archived_trial = number_trial_archived_found

        ##### Using the Variables_complex_seuils.xlsx creat a bool to detemine if the variable is in the range
        if 4 <= self.trial_duration <= 10:
            trial_duration_in_range = True
        else:
            trial_duration_in_range = False
            self.num_abnormal_value += 1

        if 2 <= self.walking_duration <= 6:
            walking_duration_in_range = True
        else:
            walking_duration_in_range = False
            self.num_abnormal_value += 1

        if 3.4 <= self.walking_distance <= 4:
            walking_distance_in_range = True
        else:
            walking_distance_in_range = False
            self.num_abnormal_value += 1

        if 0.7 <= self.walking_speed <= 1.5:
            walking_speed_in_range = True
        else:
            walking_speed_in_range = False
            self.num_abnormal_value += 1

        if 0 <= self.sd_walking_speed <= 0.35:
            sd_walking_speed_in_range = True
        else:
            sd_walking_speed_in_range = False
            self.num_abnormal_value += 1

        if -60 <= self.pitch_head_mean <= -10:
            pitch_head_mean_in_range = True
        else:
            pitch_head_mean_in_range = False
            self.num_abnormal_value += 1

        if 0 <= self.pitch_head_sd <= 10:
            pitch_head_sd_in_range = True
        else:
            pitch_head_sd_in_range = False
            self.num_abnormal_value += 1

        if 0 <= mean_percentage_brown_pixel <= 1:
            mean_percentage_brown_pixel_in_range = True
        else:
            mean_percentage_brown_pixel_in_range = False
            self.num_abnormal_value += 1

        if 0.2 <= mean_MOS_AP_y <= 0.6:
            MEAN_MOS_AP_Y_in_range = True
        else:
            MEAN_MOS_AP_Y_in_range = False
            self.num_abnormal_value += 1

        if 0 <= std_MOS_AP_y <= 0.1:
            STD_MOS_AP_Y_in_range = True
        else:
            STD_MOS_AP_Y_in_range = False
            self.num_abnormal_value += 1

        if 0 <= max_MOS_AP_y <= 1:
            MAX_MOS_AP_Y_in_range = True
        else:
            MAX_MOS_AP_Y_in_range = False
            self.num_abnormal_value += 1

        if 0 <= min_MOS_AP_y <= 0.6:
            MIN_MOS_AP_Y_in_range = True
        else:
            MIN_MOS_AP_Y_in_range = False
            self.num_abnormal_value += 1

        if 0 <= mean_MOS_ML_x <= 0.2:
            MEAN_MOS_ML_X_in_range = True
        else:
            MEAN_MOS_ML_X_in_range = False
            self.num_abnormal_value += 1

        if 0 <= std_MOS_ML_x <= 0.1:
            STD_MOS_ML_X_in_range = True
        else:
            STD_MOS_ML_X_in_range = False
            self.num_abnormal_value += 1

        if 0 <= max_MOS_ML_x <= 1:
            MAX_MOS_ML_X_in_range = True
        else:
            MAX_MOS_ML_X_in_range = False
            self.num_abnormal_value += 1

        if -0.2 <= min_MOS_ML_x <= 0.2:
            MIN_MOS_ML_X_in_range = True
        else:
            MIN_MOS_ML_X_in_range = False
            self.num_abnormal_value += 1

        if 0 <= acc_RMS_head_X <= 3.5:
            acc_RMS_Head_X_in_range = True
        else:
            acc_RMS_Head_X_in_range = False
            self.num_abnormal_value += 1

        if 0 <= acc_RMS_head_Y <= 3.5:
            acc_RMS_Head_Y_in_range = True
        else:
            acc_RMS_Head_Y_in_range = False
            self.num_abnormal_value += 1

        if 0 <= acc_RMS_pelvis_X <= 3.5:
            acc_RMS_Pelvis_X_in_range = True
        else:
            acc_RMS_Pelvis_X_in_range = False
            self.num_abnormal_value += 1

        if 0 <= acc_RMS_pelvis_Y <= 3.5:
            acc_RMS_Pelvis_Y_in_range = True
        else:
            acc_RMS_Pelvis_Y_in_range = False
            self.num_abnormal_value += 1

        # Add these variables in the final result
        self.dic_complex_variables = {
            'trial_duration_in_range': trial_duration_in_range,
            'walking_duration_in_range': walking_duration_in_range,
            'walking_distance_in_range': walking_distance_in_range,
            'walking_speed_in_range': walking_speed_in_range,
            'sd_walking_speed_in_range': sd_walking_speed_in_range,
            'pitch_head_mean_in_range': pitch_head_mean_in_range,
            'pitch_head_sd_in_range': pitch_head_sd_in_range,
            'nb_placement_maze': nb_placement_maze,
            'nb_placement_trial': nb_placement_trial,
            'nb_light_green_pixel': nb_light_green_pixel,
            'nb_light_green_pixel_per_step': nb_light_green_pixel_per_step,
            'nb_brown_pixel': nb_brown_pixel,
            'nb_brown_pixel_per_step': nb_brown_pixel_per_step,
            'nb_feet_form_pixel': nb_feet_form_pixel,
            'nb_feet_form_pixel_per_step': nb_feet_form_pixel_per_step,
            'mean_percentage_light_green_pixel': mean_percentage_light_green_pixel,
            'mean_percentage_brown_pixel': mean_percentage_brown_pixel,
            'mean_percentage_brown_pixel_in_range': mean_percentage_brown_pixel_in_range,
            'mean_percentage_brown_2_first_steps': mean_percentage_brown_2_first_steps,
            'mean_percentage_brown_2_last_steps': mean_percentage_brown_2_last_steps,
            'mean_percentage_brown_345_steps': mean_percentage_brown_345_steps,
            'MEAN_MOS_AP_Y': mean_MOS_AP_y,
            'MEAN_MOS_AP_Y_in_range': MEAN_MOS_AP_Y_in_range,
            'STD_MOS_AP_Y': std_MOS_AP_y,
            'STD_MOS_AP_Y_in_range': STD_MOS_AP_Y_in_range,
            'MAX_MOS_AP_Y': max_MOS_AP_y,
            'MAX_MOS_AP_Y_in_range': MAX_MOS_AP_Y_in_range,
            'MIN_MOS_AP_Y': min_MOS_AP_y,
            'MIN_MOS_AP_Y_in_range': MIN_MOS_AP_Y_in_range,
            'PER_Forward_MOS_AP': per_forward_MOS_AP,
            'MEAN_MOS_ML_X': mean_MOS_ML_x,
            'MEAN_MOS_ML_X_in_range': MEAN_MOS_ML_X_in_range,
            'STD_MOS_ML_X': std_MOS_ML_x,
            'STD_MOS_ML_X_in_range': STD_MOS_ML_X_in_range,
            'MAX_MOS_ML_X': max_MOS_ML_x,
            'MAX_MOS_ML_X_in_range': MAX_MOS_ML_X_in_range,
            'MIN_MOS_ML_X': min_MOS_ML_x,
            'MIN_MOS_ML_X_in_range': MIN_MOS_ML_X_in_range,
            'PER_Inside_MOS_ML': per_inside_MOS_ML,
            'Number_of_outliers_MOS_AP_Y': number_of_outliers_MOS_AP_Y,
            'Number_of_outliers_MOS_ML_X': number_of_outliers_MOS_ML_X,
            'Acc_RMS_Head_X': acc_RMS_head_X,
            'Acc_RMS_Head_X_in_range': acc_RMS_Head_X_in_range,
            'Acc_RMS_Head_Y': acc_RMS_head_Y,
            'Acc_RMS_Head_Y_in_range': acc_RMS_Head_Y_in_range,
            'Acc_RMS_Pelvis_X': acc_RMS_pelvis_X,
            'Acc_RMS_Pelvis_X_in_range': acc_RMS_Pelvis_X_in_range,
            'Acc_RMS_Pelvis_Y': acc_RMS_pelvis_Y,
            'Acc_RMS_Pelvis_Y_in_range': acc_RMS_Pelvis_Y_in_range,
            'Number_Archived_Trial': number_archived_trial,
        }

        for key, item in self.dic_complex_variables.items():
            self.df_trial_result[key] = item

        # Save the number of total abnormal values
        print('self.num_abnormal_value', self.num_abnormal_value)
        self.df_trial_result['num_abnormal_value'] = self.num_abnormal_value

        # Save the number of total number of peak detected for some main rigidbody
        print('self.num_peak_deleted', self.num_peak_deleted)
        self.df_trial_result['num_peak_deleted'] = self.num_peak_deleted

        # Save trial result dataframe to csv file
        self.df_trial_result.to_csv(self.trial.trial_folder_path + 'Result/' + self.trial.config.name + '_result.csv', sep=';')


# In case of analysing all the session in one folder
for path in os.listdir(root_data_path):
    if "Session" not in path:
        continue

    full_path = os.path.join(root_data_path, path)
    participant_name = path.rsplit('_', 1)[0]
    session_name = path.rsplit('_', 1)[1]

    session = ComplexTerrainSessionData(root_data_path, participant_name, session_name)
    # print(len(session.list_trials))

    # Basic analyse each trial in Archived folder
    for trial in session.list_archived_trials:
        # Create the raw data(all data assembled in a csv), stuctured data(2 layer of index for a fast indexing), filtered data(with interpolation and filer)
        trial.data_preparation()
        trial.export_vicon_to_c3d()
        print('Archived Trial ' + str(trial.trial_folder_path) + ' analyse completed')

    # Analyse each trial in Data folder, and each PWS calibration in PWSCalibration folder
    for trial in session.list_trials:
        # Create the raw data(all data assembled in a csv), stuctured data(2 layer of index for a fast indexing), filtered data(with interpolation and filer)
        trial.data_preparation()

        if trial.config.category not in ['Adaptation']:
            if trial.load_vicon_filtered():
                # Convert the filtered data to a c3d file in order to visualize in the Mokka application
                trial.export_vicon_to_c3d()
                trial.complex_terrain_analysis_routine(verbose=True, plot_it=False, save_plot=True)
                print('Trial ' + str(trial.config.index) + ' analyse completed')

    session.export_session_result()
