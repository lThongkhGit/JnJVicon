import os
import os.path
import pandas as pd
import numpy as np
import scipy as sp
#import pyqtgraph as pg
#import pyqtgraph.exporters
import datetime
import statistics
from ppretty import ppretty
import jnj_methods_analysis as analysis
import jnj_methods_data_preparation as preparation
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation

# The path of the all sessions data
#root_data_path = 'D:/JNJ/clinical_study_Crossing/'
root_data_path = 'C:/Users/lthongkh/Documents/JNJ/clinical_study_Crossing/'

# The list containing all obstacle rigidbody names
list_obstacle_names = ['Step051', 'Step052', 'Step151', 'Step152']


class CrossingSessionData(preparation.SessionData):
    def __init__(self, root_data_path_, participant_id_, session_id_):
        # Call the parent method
        super().__init__(root_data_path_, participant_id_, session_id_)

        # Create trial data list
        self.list_trials = []

        for index, row in self.df_config.iterrows():
            self.list_trials.append(CrossingTrialData(self, row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]))

        print('self.list_trials: ', len(self.list_trials))

        # Create archived trial data list
        self.list_archived_trials = []

        session_archived_folder_path = self.session_folder_path + 'Archived'
        for path in os.listdir(session_archived_folder_path):
            # (self, session_, index_, category_, light_condition_lux_, task_type_, step_1_height_, step_1_position_, step_2_height_, step_2_position_, index_in_phase_):
            crossing_trial_data = CrossingTrialData(self, session_id_, '0', 'Archived', 14.34, '0_0-0_0', 0, 0, 0, 0)
            crossing_trial_data.trial_folder_path = session_archived_folder_path + '/' + path + '/'
            self.list_archived_trials.append(crossing_trial_data)

        print('self.list_archived_trials: ', len(self.list_archived_trials))


class CrossingTrialData(preparation.TrialData):
    def __init__(self, session_, index_, category_, light_condition_lux_, task_type_, step_1_height_, step_1_position_, step_2_height_, step_2_position_, index_in_phase_):

        # Call the parent method
        super().__init__(session_)

        # The TrialConfig instance of this trial
        self.config = CrossingTrialConfig(self, index_, category_, light_condition_lux_, task_type_, step_1_height_, step_1_position_, step_2_height_, step_2_position_, index_in_phase_)

        # The path of the trial folder
        self.trial_folder_path = self.session.session_folder_path + 'Data/Trial_' + str(self.config.index) + '_' + str(self.config.category) + '_' + str(self.config.task_type) + '/'

        # The results of the trial
        self.result = CrossingTrialResult(self)

    # data preparation:
    # read vicon data, restructure and filter data
    def data_preparation(self):
        # Call the parent method
        super().data_preparation()

    # The analyse routine of global variables and spatial temporal variables for crossing task
    def crossing_analysis_routine(self, plot_it=False, save_plot=False, verbose=False):
        # 1. Calculate the global variables : trial duration, walking duration, walking distance, walking speed
        self.calculate_global_variables()

        # 2. Prepare parameters for crossing:
        # Detect the two obstacles in the middle of the room using the position of top center
        # If there is no obstacle for any crossing, do not analyse crossing variables
        if self.crossing_detect_obstacles():
            # Detect the index of frame of applomb for left and right toe for two crossings (the moment that the foot is closest to the obstacle in y axis)
            self.crossing_detect_applomb()

            # Calculate the temporal window of the two crossings to avoid intersection between two crossings
            self.crossing_calculate_temporal_window()

            # 3. For each crossing, calculate the spatial temporal variables
            self.crossing_calculate_spatial_temporal_variables(verbose=verbose, plot_it=plot_it, save_plot=save_plot)

            # 4. For each crossing, calculate the head orientation variables (pitch in degree. When positive, look up; when negative, look down)
            self.crossing_calculate_stability_variable(plot_it=plot_it, save_plot=save_plot)

            # 5. For each crossing, calculate the head orientation variables (pitch in degree. When positive, look up; when negative, look down)
            self.crossing_calculate_head_orientation(plot_it=plot_it, save_plot=save_plot)

            # 6. For some variables, calculate if it is in range or no
            self.crossing_calculate_variables_in_range(plot_it=plot_it, save_plot=save_plot)

            # 7. Verify if there are folders in the Archive have the same name of trail
            self.crossing_calculate_archived_data(plot_it=plot_it, save_plot=save_plot)

        # 8. Export trial result to csv
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
        #df_head = self.df_vicon_filtered.loc[:, ('Head', preparation.position_iterables)]
        df_head = self.df_vicon_filtered.loc[:, ('HeadCalibrated', preparation.position_iterables)]

        # Get all frames when y is between [-2500, 3000]
        #df_head = df_head[np.logical_and(df_head[('Head', 'PosY')].values < 3000, df_head[('Head', 'PosY')].values > -2500)]
        df_head = df_head[
            np.logical_and(df_head[('HeadCalibrated', 'PosY')].values < 3000, df_head[('HeadCalibrated', 'PosY')].values > -2500)]

        # calculate scalar speed (y axis) for each frame (when y is between [-2500, 3000])
        diff_head = df_head.diff().fillna(0.)
        #diff_head['DistanceY'] = np.abs(diff_head.loc[:, ('Head', 'PosY')])
        diff_head['DistanceY'] = np.abs(diff_head.loc[:, ('HeadCalibrated', 'PosY')])
        diff_head['SpeedY'] = diff_head['DistanceY'] / (1 / 120.)

        # Calculate the walking distance (mm) The walking distance accumulated by the head in y axis (when y is between [-2500, 3000])
        self.result.walking_distance = np.sum(diff_head['DistanceY'])

        # Calculate the walking duration (s) (the duration of all frames when y is between [-2500, 3000])
        self.result.walking_duration = len(diff_head.index) / 120.

        # Calculate the mean walking speed (mm/s) (the mean speed of all frames when y is between [-2500, 3000])
        self.result.walking_speed = np.mean(diff_head['SpeedY'].to_numpy())

        # Calculate the standard deviation of walking speed of all frames when y is between [-2500, 3000]
        self.result.sd_walking_speed = np.std(diff_head['SpeedY'].to_numpy())

        # Calculate the pitch hand orientation for the mean and standard deviation
        df_head_quaternion = self.df_vicon_filtered.loc[:, ('HeadCalibrated', preparation.quaternion_iterables)].fillna(0)
        list_quaternion = df_head_quaternion.values.tolist()

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

    # Detect the two obstacles in the middle of the room using the position of top center
    # Verify if the obstacle is correct according to the config file and if there is position or rotation error
    def crossing_detect_obstacles(self):
        # If there is no obstacle for the crossing according to config, add 'No_Crossing' to Crossing.error_code
        if self.config.step_1_height == 0:
            self.result.list_crossing[0].error_code += 'No_Crossing_'
        if self.config.step_2_height == 0:
            self.result.list_crossing[1].error_code += 'No_Crossing_'

        # If there is no obstacle for any crossing, do not analyse crossing variables
        if self.config.step_1_height == 0 and self.config.step_2_height == 0:
            return False

        # Detect the two obstacles in the middle of the room using the position of top center:
        # 1. Detect the two obstacles with the smaller x value;
        # 2. Detect the first obstacle with the larger y value;
        # 3. Detect the second obstacle with the smaller y value.

        # Preparation:
        list_obstacle_name_pos = []
        for obstacle in list_obstacle_names:
            # Check if the obstacle name is in the list_rigidbody_in_trial
            if obstacle in self.list_rigidbody_in_trial:
                # Calculate the mean position for top left and top right for the first 3 seconds, since the participant starts walking after 3 seconds,
                # and we want to avoid the inversion of markers between the foot and the obstacle when the participant approaches the obstacle

                # # The mean of top left position of the obstacle
                # pos_top_left = self.df_vicon_filtered.loc[0:int(3 * preparation.fps),
                #                (obstacle + '_Marker_' + obstacle + 'TopL')].mean()
                # # The mean of top right position of the obstacle
                # pos_top_right = self.df_vicon_filtered.loc[0:int(3 * preparation.fps),
                #                 (obstacle + '_Marker_' + obstacle + 'TopR')].mean()
                # # The mean of top center position of the obstacle
                # pos_top_center = (pos_top_left + pos_top_right) / 2.
                # pos_top_center = [pos_top_center['PosX'], pos_top_center['PosY'], pos_top_center['PosZ']]

                pos_top_center = self.df_vicon_filtered.loc[0:int(3 * preparation.fps), obstacle].mean()
                pos_top_center = [pos_top_center['PosX'], pos_top_center['PosY'], pos_top_center['PosZ']]

                # # Calculate the mean position of all other markers except top left and top right
                # # There are always 5 markers for all obstacles, for example Step051_Marker_Step0511 is the marker 1 for Step051
                # pos_mean_other_markers = 0
                # for i in range(5):
                #     pos_mean_other_markers += self.df_vicon_filtered.loc[0:int(3 * preparation.fps), (obstacle + '_Marker_' + obstacle + str(i + 1))].mean()
                # pos_mean_other_markers /= 5.

                # list_obstacle_name_pos structure:
                # [[name_of_obstacle1, pos_top_center_of_obstacle1, pos_top_left_of_obstacle1, pos_top_right_of_obstacle1, pos_mean_other_markers_of_obstacle1], [...], [...], [...], [...]]
                # list_obstacle_name_pos.append([obstacle, pos_top_center, pos_top_left, pos_top_right, pos_mean_other_markers])
                # list_obstacle_name_pos.append([obstacle, pos_top_center, pos_mean_other_markers])
                list_obstacle_name_pos.append([obstacle, pos_top_center])

        # If there are less than 2 obstacles detected, return false
        if len(list_obstacle_name_pos) < 2:
            print('There are not enough obstacles detected for Trial_' + str(
                self.config.index) + ' in session ' + self.session.participant_id + '_' + self.session.session_id + ' !')

        # 1. Detect the two obstacles with the larger x value:
        list_obstacle_name_pos.sort(key=lambda x: x[1][0], reverse=True)

        # Keep only the first two obstacles (with the larger x value)
        list_obstacle_name_pos = list_obstacle_name_pos[:2]

        # 2. Detect the first obstacle with the larger y value;
        # 3. Detect the second obstacle with the smaller y value.
        list_obstacle_name_pos.sort(key=lambda x: x[1][1], reverse=True)
        for i in range(2):
            print('obstacle ', i, ' name: ', list_obstacle_name_pos[i][0])
            self.result.list_crossing[i].obstacle_name = list_obstacle_name_pos[i][0]
            self.result.list_crossing[i].pos_top_center_obstacle = list_obstacle_name_pos[i][1]

        # Verify whether the position and rotation of obstacles is correct:
        # 1. Verify the obstacles used by comparing the obstacle_name detected during the crossing and the name in config file
        # If verification 1 fail, add 'WrongObstacle' to Crossing.error_code
        # 2. Verify the rotation by comparing the relative position of pos_top_left and pos_top_right in each axis (x, y, z):
        # 2.1. Verify whether the pos_top_left.x > pos_top_right.x
        # If verification 2.1 fail, add 'LeftRightFlip' to Crossing.error_code
        # 2.2. Verify whether the pos_top_center.z > pos_mean_other_markers_of_obstacle.z (except top left marker and top right marker)
        # If verification 2.2 fail, add 'UpsideDown' to Crossing.error_code
        # 2.3. Verify whether abs(pos_top_left.x - pos_top_right.x) > abs(pos_top_left.y - pos_top_right.y)
        # If verification 2.3 fail, add 'NotParallel' to Crossing.error_code

        # 1. Verify the obstacles used by comparing the obstacle_name detected during the crossing and the name in config file
        # If verification 1 fail, add 'WrongObstacle' to Crossing.error_code
        # The name of obstacles used in crossing is saved in list_obstacle_name_pos: Step051/Step052/Step151/Step152
        # The name of obstacles in config file is saved in self.config.step_1_height for self.config.step_2_height: 5/15
        if ('Step05' in list_obstacle_name_pos[0][0] and self.config.step_1_height != 5) or ('Step15' in list_obstacle_name_pos[0][0] and self.config.step_1_height != 15):
            self.result.list_crossing[0].error_code += 'WrongObstacle_'
            print(list_obstacle_name_pos[0][0])
            print(self.config.step_1_height)
        if ('Step05' in list_obstacle_name_pos[1][0] and self.config.step_2_height != 5) or ('Step15' in list_obstacle_name_pos[1][0] and self.config.step_2_height != 15):
            self.result.list_crossing[1].error_code += 'WrongObstacle_'
            print(list_obstacle_name_pos[1][0])
            print(self.config.step_2_height)

        # # 2.1. Verify whether the pos_top_left.x > pos_top_right.x
        # # If verification 2.1 fail, add 'LeftRightFlip' to Crossing.error_code
        # for i in range(2):
        #     if list_obstacle_name_pos[i][2][0] < list_obstacle_name_pos[i][3][0]:
        #         self.result.list_crossing[i].error_code += 'LeftRightFlip_'
        #
        # # 2.2. Verify whether the pos_top_center.z > pos_mean_other_markers_of_obstacle.z
        # # If verification 2.2 fail, add 'UpsideDown' to Crossing.error_code
        # for i in range(2):
        #     if list_obstacle_name_pos[i][1][2] < list_obstacle_name_pos[i][4][2]:
        #         self.result.list_crossing[i].error_code += 'UpsideDown_'
        #
        # # 2.3. Verify whether abs(pos_top_left.x - pos_top_right.x) > abs(pos_top_left.y - pos_top_right.y)
        # # If verification 2.3 fail, add 'NotParallel' to Crossing.error_code
        # for i in range(2):
        #     if abs(list_obstacle_name_pos[i][2][0] - list_obstacle_name_pos[i][3][0]) < abs(list_obstacle_name_pos[i][2][1] - list_obstacle_name_pos[i][3][1]):
        #         print(list_obstacle_name_pos[i][2])
        #         print(list_obstacle_name_pos[i][3])
        #         self.result.list_crossing[i].error_code += 'NotParallel_'

        return True

    # Detect the index of frame of applomb for left and right toe for two crossings
    # (the moment that the foot is closest to the obstacle in y axis)
    def crossing_detect_applomb(self):
        for i in range(2):
            self.result.list_crossing[i].detect_applomb()

    # Calculate the temporal window of the two crossings to avoid intersection between two crossings
    def crossing_calculate_temporal_window(self):
        # Calculate the temporal window of the crossing. (version 2 used)

        # version 1:
        # For the first crossing:
        # from the trial start frame to the frame when the toe marker of the lead foot for the second crossing passes the obstacle
        # For the second crossing:
        # from the frame when the toe marker of the lead foot for the first crossing passes the obstacle to the trial end frame

        # version 2:
        # For the first crossing & the second crossing:
        # from the trial start frame to the trial end frame

        # version 1:
        # self.result.list_crossing[0].set_temporal_window(self.df_vicon_filtered.index[0], self.result.list_crossing[1].idx_applomb_first)
        # self.result.list_crossing[1].set_temporal_window(self.result.list_crossing[0].idx_applomb_first, self.df_vicon_filtered.index[-1])

        # version 2:
        self.result.list_crossing[0].set_temporal_window(self.df_vicon_filtered.index[0],
                                                         self.df_vicon_filtered.index[-1])
        self.result.list_crossing[1].set_temporal_window(self.df_vicon_filtered.index[0],
                                                         self.df_vicon_filtered.index[-1])

    # Calculate the spatial temporal variables for crossings
    def crossing_calculate_spatial_temporal_variables(self, verbose=False, plot_it=False, save_plot=False):
        # For each crossing, calculate the spatial temporal variables
        for i in range(2):
            self.result.list_crossing[i].calculate_spatial_temporal_variables(verbose=verbose, plot_it=plot_it,
                                                                              save_plot=save_plot)

    # Calculate the stability variables
    def crossing_calculate_stability_variable(self, plot_it=False, save_plot=False):
        # For each crossing, calculate the stability variables
        for i in range(2):
            if self.result.list_crossing[i].can_calculate_result:
                self.result.list_crossing[i].calculate_stability_variables(plot_it=plot_it, save_plot=save_plot)

    # Calculate the head orientation variables (pitch in degree. When positive, look up; when negative, look down)
    def crossing_calculate_head_orientation(self, plot_it=False, save_plot=False):
        # For each crossing, calculate the head orientation variables
        for i in range(2):
            if self.result.list_crossing[i].can_calculate_result:
                self.result.list_crossing[i].calculate_head_orientation(plot_it=plot_it, save_plot=save_plot)

    # For one trial, check if there are some folders with the same in the folder Archived
    def crossing_calculate_archived_data(self, plot_it=False, save_plot=False):
        for i in range(2):
            self.result.list_crossing[i].calculate_archived_data(plot_it=plot_it, save_plot=save_plot)

    # Verify if the crossing variables are in range provided from document
    def crossing_calculate_variables_in_range(self, plot_it=False, save_plot=False):
        # Calculate the range of some variables in the result
        for i in range(2):
            if self.result.list_crossing[i].can_calculate_result:
                if 100 <= self.result.list_crossing[i].penultimate_foot_placement <= 1200:
                    self.result.list_crossing[i].penultimate_foot_placement_in_range = True
                else:
                    self.result.list_crossing[i].penultimate_foot_placement_in_range = False
                    self.result.num_abnormal_value += 1

                if 0 <= self.result.list_crossing[i].final_foot_placement <= 400:
                    self.result.list_crossing[i].final_foot_placement_in_range = True
                else:
                    self.result.list_crossing[i].final_foot_placement_in_range = False
                    self.result.num_abnormal_value += 1

                if 0 <= self.result.list_crossing[i].lead_vertical_toe_clearance <= 400:
                    self.result.list_crossing[i].lead_vertical_toe_clearance_in_range = True
                else:
                    self.result.list_crossing[i].lead_vertical_toe_clearance_in_range = False
                    self.result.num_abnormal_value += 1

                if 0 <= self.result.list_crossing[i].trail_vertical_toe_clearance <= 400:
                    self.result.list_crossing[i].trail_vertical_toe_clearance_in_range = True
                else:
                    self.result.list_crossing[i].trail_vertical_toe_clearance_in_range = False
                    self.result.num_abnormal_value += 1

                if -800 <= self.result.list_crossing[i].lead_foot_placement_toe <= -200:
                    self.result.list_crossing[i].lead_foot_placement_toe_in_range = True
                else:
                    self.result.list_crossing[i].lead_foot_placement_toe_in_range = False
                    self.result.num_abnormal_value += 1

                if -800 <= self.result.list_crossing[i].lead_foot_placement_heel <= -100:
                    self.result.list_crossing[i].lead_foot_placement_heel_in_range = True
                else:
                    self.result.list_crossing[i].lead_foot_placement_heel_in_range = False
                    self.result.num_abnormal_value += 1

                if -1700 <= self.result.list_crossing[i].trail_foot_placement_toe <= -700:
                    self.result.list_crossing[i].trail_foot_placement_toe_in_range = True
                else:
                    self.result.list_crossing[i].trail_foot_placement_toe_in_range = False
                    self.result.num_abnormal_value += 1

                if -1700 <= self.result.list_crossing[i].trail_foot_placement_heel <= -700:
                    self.result.list_crossing[i].trail_foot_placement_heel_in_range = True
                else:
                    self.result.list_crossing[i].trail_foot_placement_heel_in_range = False
                    self.result.num_abnormal_value += 1

                if 500 <= self.result.list_crossing[i].step_length_crossing <= 1200:
                    self.result.list_crossing[i].step_length_crossing_in_range = True
                else:
                    self.result.list_crossing[i].step_length_crossing_in_range = False
                    self.result.num_abnormal_value += 1

                if 0 <= self.result.list_crossing[i].step_width_crossing <= 250:
                    self.result.list_crossing[i].step_width_crossing_in_range = True
                else:
                    self.result.list_crossing[i].step_width_crossing_in_range = False
                    self.result.num_abnormal_value += 1

                if 0 <= self.result.list_crossing[i].double_support_before_crossing <= 0.25:
                    self.result.list_crossing[i].double_support_before_crossing_in_range = True
                else:
                    self.result.list_crossing[i].double_support_before_crossing_in_range = False
                    self.result.num_abnormal_value += 1

                if 0 <= self.result.list_crossing[i].single_support_trail <= 0.8:
                    self.result.list_crossing[i].single_support_trail_in_range = True
                else:
                    self.result.list_crossing[i].single_support_trail_in_range = False
                    self.result.num_abnormal_value += 1

                if 0 <= self.result.list_crossing[i].double_support_crossing <= 0.3:
                    self.result.list_crossing[i].double_support_crossing_in_range = True
                else:
                    self.result.list_crossing[i].double_support_crossing_in_range = False
                    self.result.num_abnormal_value += 1

                if 0 <= self.result.list_crossing[i].single_support_lead <= 0.6:
                    self.result.list_crossing[i].single_support_lead_in_range = True
                else:
                    self.result.list_crossing[i].single_support_lead_in_range = False
                    self.result.num_abnormal_value += 1

                if 0 <= self.result.list_crossing[i].mean_speed_head_lateral <= 300:
                    self.result.list_crossing[i].mean_speed_head_lateral_in_range = True
                else:
                    self.result.list_crossing[i].mean_speed_head_lateral_in_range = False
                    self.result.num_abnormal_value += 1

                if 0 <= self.result.list_crossing[i].sd_speed_head_lateral <= 300:
                    self.result.list_crossing[i].sd_speed_head_lateral_in_range = True
                else:
                    self.result.list_crossing[i].sd_speed_head_lateral_in_range = False
                    self.result.num_abnormal_value += 1

                if 0 <= self.result.list_crossing[i].mean_speed_head_longitudinal <= 2000:
                    self.result.list_crossing[i].mean_speed_head_longitudinal_in_range = True
                else:
                    self.result.list_crossing[i].mean_speed_head_longitudinal_in_range = False
                    self.result.num_abnormal_value += 1

                if 0 <= self.result.list_crossing[i].sd_speed_head_longitudinal <= 400:
                    self.result.list_crossing[i].sd_speed_head_longitudinal_in_range = True
                else:
                    self.result.list_crossing[i].sd_speed_head_longitudinal_in_range = False
                    self.result.num_abnormal_value += 1

                if -10 <= self.result.list_crossing[i].head_pitch_frame_lead_foot_toe_off_before <= 40:
                    self.result.list_crossing[i].head_pitch_frame_lead_foot_toe_off_before_in_range = True
                else:
                    self.result.list_crossing[i].head_pitch_frame_lead_foot_toe_off_before_in_range = False
                    self.result.num_abnormal_value += 1

                if -10 <= self.result.list_crossing[i].head_pitch_frame_applomb_lead_foot <= 40:
                    self.result.list_crossing[i].head_pitch_frame_applomb_lead_foot_in_range = True
                else:
                    self.result.list_crossing[i].head_pitch_frame_applomb_lead_foot_in_range = False
                    self.result.num_abnormal_value += 1

                if -10 <= self.result.list_crossing[i].head_pitch_frame_trail_foot_heel_strike_after <= 40:
                    self.result.list_crossing[i].head_pitch_frame_trail_foot_heel_strike_after_in_range = True
                else:
                    self.result.list_crossing[i].head_pitch_frame_trail_foot_heel_strike_after_in_range = False
                    self.result.num_abnormal_value += 1

                if 0 <= self.result.list_crossing[i].MOS_AP_y <= 1:
                    self.result.list_crossing[i].MOS_AP_y_in_range = True
                else:
                    self.result.list_crossing[i].MOS_AP_y_in_range = False
                    self.result.num_abnormal_value += 1

                if -1 <= self.result.list_crossing[i].MOS_ML_x <= 1:
                    self.result.list_crossing[i].MOS_ML_x_in_range = True
                else:
                    self.result.list_crossing[i].MOS_ML_x_in_range = False
                    self.result.num_abnormal_value += 1

            else:
                self.result.num_abnormal_value += 21


class CrossingTrialConfig(preparation.TrialConfig):
    def __init__(self, trial_, index_, category_, light_condition_lux_, task_type_, step_1_height_, step_1_position_, step_2_height_, step_2_position_, index_in_phase_):
        # Call the parent method
        super().__init__()

        # The TrialData instance
        self.trial = trial_
        self.index = index_
        self.category = category_
        self.light_condition_lux = light_condition_lux_
        self.task_type = task_type_
        self.step_1_height = step_1_height_
        self.step_1_position = step_1_position_
        self.step_2_height = step_2_height_
        self.step_2_position = step_2_position_
        self.index_in_phase = index_in_phase_
        # The name of the trial in the Data folder, for example, Trial_6_CrossingET_STV
        self.name = 'Trial_' + str(self.index) + '_' + str(self.category) + '_' + str(self.task_type)

        # The list of config elements, used for exporting result
        self.list_config_elements = {'session_name': self.trial.session.name,
                                     'trial_name': self.name,
                                     'index': self.index,
                                     'category': self.category,
                                     'light_condition_lux': self.light_condition_lux,
                                     'task_type': self.task_type,
                                     'step_1_height': self.step_1_height,
                                     'step_1_position': self.step_1_position,
                                     'step_2_height': self.step_2_height,
                                     'step_2_position': self.step_2_position,
                                     'index_in_phase': self.index_in_phase
                                     }


# The instance that contains all results for a trial
class CrossingTrialResult(preparation.TrialResult):
    def __init__(self, trial_):

        # Call the parent method
        super().__init__(trial_)

        # Create the two crossings
        self.list_crossing = [Crossing(1, trial_), Crossing(2, trial_)]

    # Generate trial result data for global variables and spatial temporal variables for each crossing
    def generate_trial_result(self):

        # Call the parent method
        super().generate_trial_result()

        # Add _in_range for some general trial result
        if 6 <= self.trial_duration <= 18:
            self.df_trial_result['trial_duration_in_range'] = True
        else:
            self.df_trial_result['trial_duration_in_range'] = False
            self.num_abnormal_value += 1

        if 2.5 <= self.walking_duration <= 7:
            self.df_trial_result['walking_duration_in_range'] = True
        else:
            self.df_trial_result['walking_duration_in_range'] = False
            self.num_abnormal_value += 1

        if 5450 <= self.walking_distance <= 5550:
            self.df_trial_result['walking_distance_in_range'] = True
        else:
            self.df_trial_result['walking_distance_in_range'] = False
            self.num_abnormal_value += 1

        if 700 <= self.walking_speed <= 1900:
            self.df_trial_result['walking_speed_in_range'] = True
        else:
            self.df_trial_result['walking_speed_in_range'] = False
            self.num_abnormal_value += 1

        if 70 <= self.sd_walking_speed <= 350:
            self.df_trial_result['sd_walking_speed_in_range'] = True
        else:
            self.df_trial_result['sd_walking_speed_in_range'] = False
            self.num_abnormal_value += 1

        if -10 <= self.pitch_head_mean <= 40:
            self.df_trial_result['pitch_head_mean_in_range'] = True
        else:
            self.df_trial_result['pitch_head_mean_in_range'] = False
            self.num_abnormal_value += 1

        if 0 <= self.pitch_head_sd <= 9:
            self.df_trial_result['pitch_head_sd_in_range'] = True
        else:
            self.df_trial_result['pitch_head_sd_in_range'] = False
            self.num_abnormal_value += 1

        # Feed the spatial temporal variables for each crossing into the result dataframe
        for crossing in self.list_crossing:
            crossing.generate_spatial_temporal_variables_result()
            for key, item in crossing.dic_crossing_variables.items():
                self.df_trial_result['crossing_' + str(crossing.index) + '_' + key] = item

        # Add response result to the csv
        path_response_file = self.trial.trial_folder_path + 'Result/Result.csv'
        response_file_text = open(path_response_file, "r").read()
        array_response_file_text = response_file_text.split(';')

        self.df_trial_result['number_correct'] = array_response_file_text[1]
        self.df_trial_result['number_response'] = array_response_file_text[3]
        self.df_trial_result['bool_response'] = array_response_file_text[5].replace('\n', '')

        # Save the number of total abnormal values
        print('self.num_abnormal_value', self.num_abnormal_value)
        self.df_trial_result['num_abnormal_value'] = self.num_abnormal_value

        # Save the number of total number of peak detected for some main rigidbody
        print('self.num_peak_deleted', self.num_peak_deleted)
        self.df_trial_result['num_peak_deleted'] = self.num_peak_deleted

        # Save trial result dataframe to csv file
        self.df_trial_result.to_csv(self.trial.trial_folder_path + 'Result/' + self.trial.config.name + '_result.csv', sep=';')


# The instance of a crossing which contains all spatial and temporal variables
class Crossing:
    def __init__(self, index_crossing_, trial_):
        # The trial instance
        self.trial = trial_

        """
        Spatial variables, exported in result file
        """

        # Anterior-posterior distance between the toe of the lead foot and the obstacle before crossing (y axis)
        self.penultimate_foot_placement = None
        self.penultimate_foot_placement_in_range = None

        # Anterior-posterior distance between the toe of the trail foot and the obstacle before crossing (y axis)
        self.final_foot_placement = None
        self.final_foot_placement_in_range = None

        # Vertical distance between the toe of the lead foot and the top of the obstacle while crossing (z axis)
        self.lead_vertical_toe_clearance = None
        self.lead_vertical_toe_clearance_in_range = None

        # Vertical distance between the toe of the trail foot and the top of the obstacle while crossing (z axis)
        self.trail_vertical_toe_clearance = None
        self.trail_vertical_toe_clearance_in_range = None

        # Anterior-posterior distance between the toe of the lead foot and the obstacle after crossing (y axis)
        self.lead_foot_placement_toe = None
        self.lead_foot_placement_toe_in_range = None

        # Anterior-posterior distance between the heel of the lead foot and the obstacle after crossing (y axis)
        self.lead_foot_placement_heel = None
        self.lead_foot_placement_heel_in_range = None

        # Anterior-posterior distance between the toe of the trail foot and the obstacle after crossing (y axis)
        self.trail_foot_placement_toe = None
        self.trail_foot_placement_toe_in_range = None

        # Anterior-posterior distance between the heel of the trail foot and the obstacle after crossing (y axis)
        self.trail_foot_placement_heel = None
        self.trail_foot_placement_heel_in_range = None

        # Anterior-posterior distance from the toe of the trial foot before crossing to the toe of lead foot after crossing (y axis)
        self.step_length_crossing = None
        self.step_length_crossing_in_range = None

        # Lateral distance from the heel of the trial foot before crossing to the heel of the lead foot after crossing (x axis)
        self.step_width_crossing = None
        self.step_width_crossing_in_range = None

        """
        Temporal variables, exported in result file
        """

        # The last double support time before the crossing (from the heel strike of the trail foot before crossing to the toe off of the lead foot before crossing)
        self.double_support_before_crossing = None
        self.double_support_before_crossing_in_range = None

        # The single support time of the trail foot before the crossing (from the toe off of the lead foot before crossing to the heel strike of the lead foot after crossing)
        self.single_support_trail = None
        self.single_support_trail_in_range = None

        # The double support time during the crossing (from the heel strike of the lead foot after crossing to the toe off of the trail foot before crossing)
        self.double_support_crossing = None
        self.double_support_crossing_in_range = None

        # The single support time of the lead foot after the crossing (from the toe off of the trail foot before crossing to the heel strike of the trail foot after crossing)
        self.single_support_lead = None
        self.single_support_lead_in_range = None

        """
        Stability index, exported in result file
        """

        # The mean scalar speed (all axis) of toe strike for lead foot after the crossing. (the frame of heel strike to 10 frames after the heel strike)
        self.mean_speed_lead_foot_toe_strike_after = None

        # The mean scalar speed (all axis) of heel strike for lead foot after the crossing. (10 frames before the heel strike to the frame of heel strike)
        self.mean_speed_lead_foot_heel_strike_after = None

        # The mean lateral scalar speed (x axis) of head between the toe off of the lead foot before crossing and heel strike of trail foot after crossing
        self.mean_speed_head_lateral = None
        self.mean_speed_head_lateral_in_range = None

        # The standard deviation of the lateral scalar speed (x axis) of head of all frames between the toe off of the lead foot before crossing and heel strike of trail foot after crossing
        self.sd_speed_head_lateral = None
        self.sd_speed_head_lateral_in_range = None

        # The mean longitudinal scalar speed (y axis) of head between the toe off of the lead foot before crossing and heel strike of trail foot after crossing
        self.mean_speed_head_longitudinal = None
        self.mean_speed_head_longitudinal_in_range = None

        # The standard deviation of the longitudinal scalar speed (y axis) of head of all frames between the toe off of the lead foot before crossing and heel strike of trail foot after crossing
        self.sd_speed_head_longitudinal = None
        self.sd_speed_head_longitudinal_in_range = None

        """
        Head pitch orientation, exported in result file
        """

        # The head orientation in pitch at the frame of toe off of the lead foot before crossing
        self.head_pitch_frame_lead_foot_toe_off_before = None
        self.head_pitch_frame_lead_foot_toe_off_before_in_range = None

        # The head orientation in pitch at the frame when the toe of the lead foot is above the obstacle
        self.head_pitch_frame_applomb_lead_foot = None
        self.head_pitch_frame_applomb_lead_foot_in_range = None

        # The head orientation in pitch at the frame of heel strike of the trail foot after crossing
        self.head_pitch_frame_trail_foot_heel_strike_after = None
        self.head_pitch_frame_trail_foot_heel_strike_after_in_range = None

        """
        For information, exported in result file
        """

        # The lead foot used for the crossing (Left or Right)
        self.lead_foot = None

        # True if the result can be calculated with the implemented algorithm, False if otherwise
        self.can_calculate_result = False

        # Error code representing the errors for obstacle placement
        self.error_code = ''

        """
        For calculation, not exported in result file
        """

        # The first crossing or the second crossing (1 or 2)
        self.index = index_crossing_

        # The dictionary of spatial temporal variables for the crossing
        self.dic_crossing_variables = None

        # The frame of Toe Off lead foot before the crossing
        self.frame_lead_foot_toe_off_before = None

        # The index of frame for left toe applomb (the moment when the participant cross the obstacle with left foot)
        self.frame_applomb_LTOE = None

        # The index of frame for right toe applomb (the moment when the participant cross the obstacle with right foot)
        self.frame_applomb_RTOE = None

        # The index of frame for the lead foot toe applomb (the moment when the participant cross the obstacle with lead foot)
        self.frame_applomb_lead_foot = None

        # The frame of Heel Strike lead foot after the crossing
        self.frame_lead_foot_heel_strike_after = None

        # The frame of Heel Strike trail foot after the crossing
        self.frame_trail_foot_heel_strike_after = None

        # The temporal window of the crossing
        self.start_frame = None
        self.end_frame = None

        # The name of the obstacle (Step051 / Step052 / Step151 / Step152)
        self.obstacle_name = None

        # The top center position of the obstacle (mean value of the first 3 seconds of the trial to avoid markers' inversion)
        self.pos_top_center_obstacle = None

        # Path for saving plots
        self.save_plot_path = ''

        # MOS margin of stability variables
        self.MOS_AP_x = None
        self.MOS_AP_y = None
        self.MOS_AP_z = None
        self.MOS_ML_x = None
        self.MOS_ML_y = None
        self.MOS_ML_z = None

        self.MOS_AP_y_in_range = None
        self.MOS_ML_x_in_range = None

        # Quality of MOS
        self.Quality_MOS_AP_y = None
        self.Quality_MOS_ML_x = None

        # Archived trial information
        self.Number_Archived_Trial = None
        self.Reasons_Archived_Trial = None

    # Detect the index of frame of applomb for left and right toe for the crossing
    # (applomb : the moment that the toe is closest to the obstacle in y axis)
    def detect_applomb(self):
        self.frame_applomb_RTOE = np.nanargmin(np.sqrt((self.trial.df_vicon_filtered.loc[:, (preparation.name_right_toe_marker, 'PosY')] - self.pos_top_center_obstacle[1]) ** 2))
        self.frame_applomb_LTOE = np.nanargmin(np.sqrt((self.trial.df_vicon_filtered.loc[:, (preparation.name_left_toe_marker, 'PosY')] - self.pos_top_center_obstacle[1]) ** 2))

        self.lead_foot = 'Right'
        if self.frame_applomb_RTOE < self.frame_applomb_LTOE:
            self.lead_foot = 'Right'
            self.frame_applomb_lead_foot = self.frame_applomb_RTOE
        else:
            self.lead_foot = 'Left'
            self.frame_applomb_lead_foot = self.frame_applomb_LTOE

    # Set the temporal window of the crossing to avoid intersection between two crossings
    def set_temporal_window(self, start_frame_, end_frame_):
        self.start_frame = start_frame_
        self.end_frame = end_frame_

    # Calculate spatial temporal variables using methods in jnj_method_analysis.py
    def calculate_spatial_temporal_variables(self, verbose=False, plot_it=False, save_plot=False):
        if verbose:
            print(' ---------- ' + self.trial.config.name + ' in ' + self.trial.session.name + ' ----------- ')

        self.save_plot_path = self.trial.session.session_plot_folder_path + self.trial.config.name + '_crossing_' + str(
            self.index) + '_'
        print(self.save_plot_path)

        result = analysis.crossing_calculation(
            self.trial.df_vicon_filtered.loc[self.start_frame:self.end_frame, preparation.name_right_heel_marker].to_numpy(),
            self.trial.df_vicon_filtered.loc[self.start_frame:self.end_frame, preparation.name_left_heel_marker].to_numpy(),
            self.trial.df_vicon_filtered.loc[self.start_frame:self.end_frame, preparation.name_right_toe_marker].to_numpy(),
            self.trial.df_vicon_filtered.loc[self.start_frame:self.end_frame, preparation.name_left_toe_marker].to_numpy(),
            self.lead_foot, self.frame_applomb_LTOE, self.frame_applomb_RTOE, self.obstacle_name,
            self.pos_top_center_obstacle, verbose=verbose, plot_it=plot_it, save_plot=save_plot,
            save_plot_path=self.save_plot_path)

        if verbose:
            if result == None:
                print('Crossing error no result')
            else:
                print('Crossing result: ', result)

        if result is None:
            return

        self.penultimate_foot_placement, self.final_foot_placement, self.lead_vertical_toe_clearance, \
        self.trail_vertical_toe_clearance, self.lead_foot_placement_toe, self.lead_foot_placement_heel, \
        self.trail_foot_placement_toe, self.trail_foot_placement_heel, self.step_width_crossing, self.double_support_before_crossing, \
        self.single_support_lead, self.double_support_crossing, self.single_support_trail, self.frame_lead_foot_toe_off_before, \
        self.frame_lead_foot_heel_strike_after, self.frame_trail_foot_heel_strike_after, self.can_calculate_result = result
        self.step_length_crossing = self.final_foot_placement - self.lead_foot_placement_toe

    # Calculate the stability index variables
    def calculate_stability_variables(self, plot_it=False, save_plot=False):
        """
        Calculate mean_speed_lead_foot_heel_strike_after
        """

        # The name of lead foot heel marker
        name_lead_foot_heel_marker = preparation.name_left_heel_marker if self.lead_foot == 'Left' else preparation.name_right_heel_marker

        # df_heel_strike: The lead foot heel marker position dataframe (10 frames before the heel strike to the frame of heel strike)
        df_heel_strike = self.trial.df_vicon_filtered.loc[self.frame_lead_foot_heel_strike_after - 10: self.frame_lead_foot_heel_strike_after, name_lead_foot_heel_marker]

        # calculate scalar speed (all axis) for each frame in df_heel_strike
        diff_heel_strike = df_heel_strike.diff().fillna(0.)
        diff_heel_strike['Distance'] = np.sqrt(diff_heel_strike.PosX ** 2 + diff_heel_strike.PosY ** 2 + diff_heel_strike.PosZ ** 2)
        diff_heel_strike['Speed'] = diff_heel_strike['Distance'] / (1 / 120.)

        # The dataframe of scalar speed (all axis) of heel strike for lead foot after the crossing. (10 frames before the heel strike to the frame of heel strike)
        df_speed_lead_foot_heel_strike_after = diff_heel_strike.loc[self.frame_lead_foot_heel_strike_after - 9:, 'Speed']

        # The mean scalar speed (all axis) of heel strike for lead foot after the crossing. (10 frames before the heel strike to the frame of heel strike)
        self.mean_speed_lead_foot_heel_strike_after = np.mean(df_speed_lead_foot_heel_strike_after.to_numpy())

        """
        Calculate mean_speed_lead_foot_toe_strike_after
        """

        # The name of lead foot toe marker
        name_lead_foot_toe_marker = preparation.name_left_toe_marker if self.lead_foot == 'Left' else preparation.name_right_toe_marker

        # df_toe_strike: The lead foot toe marker position dataframe (the frame of heel strike to 10 frames after the heel strike)
        df_toe_strike = self.trial.df_vicon_filtered.loc[self.frame_lead_foot_heel_strike_after:self.frame_lead_foot_heel_strike_after + 10, name_lead_foot_toe_marker]

        # calculate scalar speed (all axis) for each frame in df_toe_strike
        diff_toe_strike = df_toe_strike.diff().fillna(0.)
        diff_toe_strike['Distance'] = np.sqrt(diff_toe_strike.PosX ** 2 + diff_toe_strike.PosY ** 2 + diff_toe_strike.PosZ ** 2)
        diff_toe_strike['Speed'] = diff_toe_strike['Distance'] / (1 / 120.)

        # The dataframe of scalar speed (all axis) of toe strike for lead foot after the crossing. (the frame of heel strike to 10 frames after the heel strike)
        df_speed_lead_foot_toe_strike_after = diff_toe_strike.loc[self.frame_lead_foot_heel_strike_after + 1:, 'Speed']

        # The mean scalar speed (all axis) of toe strike for lead foot after the crossing. (the frame of heel strike to 10 frames after the heel strike)
        self.mean_speed_lead_foot_toe_strike_after = np.mean(df_speed_lead_foot_toe_strike_after.to_numpy())

        """
        Calculate mean_speed_head_lateral, sd_speed_head_lateral, mean_speed_head_longitudinal and sd_speed_head_longitudinal
        """

        # df_head_crossing: The head calibrated rigidbody position dataframe (between the toe off of the lead foot before crossing and heel strike of trail foot after crossing)
        df_head_crossing = self.trial.df_vicon_filtered.loc[self.frame_lead_foot_toe_off_before:self.frame_trail_foot_heel_strike_after, ('HeadCalibrated', preparation.position_iterables)]

        # calculate scalar speed (x and y axis) for each frame in df_head_crossing
        diff_head_crossing = df_head_crossing.diff().fillna(0.)

        diff_head_crossing['DistanceX'] = np.abs(diff_head_crossing.loc[:, ('HeadCalibrated', 'PosX')])
        diff_head_crossing['DistanceY'] = np.abs(diff_head_crossing.loc[:, ('HeadCalibrated', 'PosY')])
        diff_head_crossing['SpeedX'] = diff_head_crossing['DistanceX'] / (1 / 120.)
        diff_head_crossing['SpeedY'] = diff_head_crossing['DistanceY'] / (1 / 120.)

        # The dataframe of lateral scalar speed (x axis) of head between the toe off of the lead foot before crossing and heel strike of trail foot after crossing
        df_speed_head_lateral = diff_head_crossing.loc[self.frame_lead_foot_toe_off_before + 1:, 'SpeedX']

        # The mean lateral scalar speed (x axis) of head between the toe off of the lead foot before crossing and heel strike of trail foot after crossing
        self.mean_speed_head_lateral = np.mean(df_speed_head_lateral.to_numpy())

        # The standard deviation of the lateral scalar speed (x axis) of head of all frames between the toe off of the lead foot before crossing and heel strike of trail foot after crossing
        self.sd_speed_head_lateral = np.std(df_speed_head_lateral.to_numpy())

        # The dataframe of longitudinal scalar speed (y axis) of head between the toe off of the lead foot before crossing and heel strike of trail foot after crossing
        df_speed_head_longitudinal = diff_head_crossing.loc[self.frame_lead_foot_toe_off_before + 1:, 'SpeedY']

        # The mean longitudinal scalar speed (y axis) of head between the toe off of the lead foot before crossing and heel strike of trail foot after crossing
        self.mean_speed_head_longitudinal = np.mean(df_speed_head_longitudinal.to_numpy())

        # The standard deviation of the longitudinal scalar speed (y axis) of head of all frames between the toe off of the lead foot before crossing and heel strike of trail foot after crossing
        self.sd_speed_head_longitudinal = np.std(df_speed_head_longitudinal.to_numpy())

        """
        Calculate MOS margin of stability for every frame and MOS_AP & MOS_ML for the frame self.frame_lead_foot_heel_strike_after
        """
        # Calculate MOS margin of stability for every frame
        for axis in preparation.position_iterables:
            # Calculate CoM - Centre of Mass - definition - barycenter of the 4 markers (in meter)
            '''self.trial.df_vicon_filtered['COM_' + axis] = (self.trial.df_vicon_filtered.loc[:, (preparation.name_LASI_marker, axis)] +
                                                           self.trial.df_vicon_filtered.loc[:, (preparation.name_LPSI_marker, axis)] +
                                                           self.trial.df_vicon_filtered.loc[:, (preparation.name_RASI_marker, axis)] +
                                                           self.trial.df_vicon_filtered.loc[:, (preparation.name_RPSI_marker, axis)]) / 4000.'''
            self.trial.df_vicon_filtered['COM_' + axis] = (self.trial.df_vicon_filtered.loc[:, (preparation.name_pelvis_marker, axis)]) / 1000.

            # Calculate distanceCoM - the distance of the CoM in each frame (in meter)
            self.trial.df_vicon_filtered['DistanceCOM_' + axis] = self.trial.df_vicon_filtered['COM_' + axis].diff().fillna(0.)

            # Calculate velocityCoM - the velocity of the CoM (in m/s)
            self.trial.df_vicon_filtered['VelocityCOM_' + axis] = self.trial.df_vicon_filtered['DistanceCOM_' + axis] / (1 / 120.)

            # Calculate AccelerationCoM - the acceleration of the CoM (in m/s^2)
            self.trial.df_vicon_filtered['AccelerationCOM_' + axis] = self.trial.df_vicon_filtered['VelocityCOM_' + axis].diff().fillna(0.) / (1 / 120.)

            # Calculate the Distance between the CoM and Heel (in meter)
            self.trial.df_vicon_filtered['DistanceCOMLeftHeel_' + axis] = self.trial.df_vicon_filtered['COM_' + axis] - self.trial.df_vicon_filtered.loc[:,
                                                                                                                        (preparation.name_left_heel_marker, axis)] / 1000.
            self.trial.df_vicon_filtered['DistanceCOMRightHeel_' + axis] = self.trial.df_vicon_filtered['COM_' + axis] - self.trial.df_vicon_filtered.loc[:,
                                                                                                                         (preparation.name_right_heel_marker, axis)] / 1000.

        # Calculate the Distance between the CoM and left foot Heel (in meter)
        self.trial.df_vicon_filtered['DistanceCOMLeftHeel'] = np.sqrt(self.trial.df_vicon_filtered['DistanceCOMLeftHeel_PosX'] ** 2
                                                                      + self.trial.df_vicon_filtered['DistanceCOMLeftHeel_PosY'] ** 2
                                                                      + self.trial.df_vicon_filtered['DistanceCOMLeftHeel_PosZ'] ** 2)

        # Calculate the Distance between the CoM and right foot Heel (in meter)
        self.trial.df_vicon_filtered['DistanceCOMRightHeel'] = np.sqrt(self.trial.df_vicon_filtered['DistanceCOMRightHeel_PosX'] ** 2
                                                                       + self.trial.df_vicon_filtered['DistanceCOMRightHeel_PosY'] ** 2
                                                                       + self.trial.df_vicon_filtered['DistanceCOMRightHeel_PosZ'] ** 2)

        # Calculate XCoM - Extrapolated center of mass
        for axis in preparation.position_iterables:
            # Left foot
            self.trial.df_vicon_filtered['XCOM_Left_' + axis] = self.trial.df_vicon_filtered['COM_' + axis]
            + self.trial.df_vicon_filtered['VelocityCOM_' + axis] / np.sqrt(9.81 / self.trial.df_vicon_filtered['DistanceCOMLeftHeel'])

            # Right foot
            self.trial.df_vicon_filtered['XCOM_Right_' + axis] = self.trial.df_vicon_filtered['COM_' + axis]
            + self.trial.df_vicon_filtered['VelocityCOM_' + axis] / np.sqrt(9.81 / self.trial.df_vicon_filtered['DistanceCOMRightHeel'])

        # Calculate the MOS_AP
        # reverse the two sides of calculation so that all the MOS_AP of right foot to positive
        # Left foot
        self.trial.df_vicon_filtered['MOS_AP_Left_PosX'] = self.trial.df_vicon_filtered['XCOM_Left_PosX'] - self.trial.df_vicon_filtered[(preparation.name_left_toe_marker, 'PosX')] / 1000.
        self.trial.df_vicon_filtered['MOS_AP_Left_PosY'] = self.trial.df_vicon_filtered['XCOM_Left_PosY'] - self.trial.df_vicon_filtered[(preparation.name_left_toe_marker, 'PosY')] / 1000.
        self.trial.df_vicon_filtered['MOS_AP_Left_PosZ'] = self.trial.df_vicon_filtered['XCOM_Left_PosZ'] - self.trial.df_vicon_filtered[(preparation.name_left_toe_marker, 'PosZ')] / 1000.
        # self.trial.df_vicon_filtered['MOS_AP_Left_norm'] = np.sqrt(self.trial.df_vicon_filtered['MOS_AP_Left_PosX'] ** 2 + self.trial.df_vicon_filtered['MOS_AP_Left_PosY'] ** 2 + self.trial.df_vicon_filtered['MOS_AP_Left_PosZ'] ** 2)

        # Right foot
        self.trial.df_vicon_filtered['MOS_AP_Right_PosX'] = self.trial.df_vicon_filtered['XCOM_Right_PosX'] - self.trial.df_vicon_filtered[(preparation.name_right_toe_marker, 'PosX')] / 1000.
        self.trial.df_vicon_filtered['MOS_AP_Right_PosY'] = self.trial.df_vicon_filtered['XCOM_Right_PosY'] - self.trial.df_vicon_filtered[(preparation.name_right_toe_marker, 'PosY')] / 1000.
        self.trial.df_vicon_filtered['MOS_AP_Right_PosZ'] = self.trial.df_vicon_filtered['XCOM_Right_PosZ'] - self.trial.df_vicon_filtered[(preparation.name_right_toe_marker, 'PosZ')] / 1000.
        # self.trial.df_vicon_filtered['MOS_AP_Right_norm'] = np.sqrt(self.trial.df_vicon_filtered['MOS_AP_Right_PosX'] ** 2 + self.trial.df_vicon_filtered['MOS_AP_Right_PosY'] ** 2 + self.trial.df_vicon_filtered['MOS_AP_Right_PosZ'] ** 2)

        # For the case if the left foot is the lead foot
        if self.lead_foot == 'Left':
            # Calculate MOS_AP & MOS_ML for the frame self.frame_lead_foot_heel_strike_after

            self.MOS_AP_x = float(self.trial.df_vicon_filtered.loc[self.frame_lead_foot_heel_strike_after, 'MOS_AP_Left_PosX'])
            self.MOS_AP_y = float(self.trial.df_vicon_filtered.loc[self.frame_lead_foot_heel_strike_after, 'MOS_AP_Left_PosY'])
            self.MOS_AP_z = float(self.trial.df_vicon_filtered.loc[self.frame_lead_foot_heel_strike_after, 'MOS_AP_Left_PosZ'])

            # Calculate BOSML_MTP_Joint - Extrapolated metatarsophalangeal joint  with Toe and Heel position - maximum lateral excursion
            vector_left_foot_toe = np.array(
                [self.trial.df_vicon_filtered.loc[self.frame_lead_foot_heel_strike_after, (preparation.name_left_toe_marker, 'PosX')] / 1000.,
                 self.trial.df_vicon_filtered.loc[self.frame_lead_foot_heel_strike_after, (preparation.name_left_toe_marker, 'PosY')] / 1000.,
                 self.trial.df_vicon_filtered.loc[self.frame_lead_foot_heel_strike_after, (preparation.name_left_toe_marker, 'PosZ')] / 1000., ])

            vector_left_foot_heel = np.array(
                [self.trial.df_vicon_filtered.loc[self.frame_lead_foot_heel_strike_after, (preparation.name_left_heel_marker, 'PosX')] / 1000.,
                 self.trial.df_vicon_filtered.loc[self.frame_lead_foot_heel_strike_after, (preparation.name_left_heel_marker, 'PosY')] / 1000.,
                 self.trial.df_vicon_filtered.loc[self.frame_lead_foot_heel_strike_after, (preparation.name_left_heel_marker, 'PosZ')] / 1000., ])

            norme_toe_to_heel = np.linalg.norm(vector_left_foot_toe - vector_left_foot_heel)

            vector_left_foot_BOSML = vector_left_foot_heel + ((vector_left_foot_toe - vector_left_foot_heel) * (13 / 19)) + \
                                     np.dot(((vector_left_foot_toe - vector_left_foot_heel) / norme_toe_to_heel), (np.array([0, 0, 1]))) * \
                                     30 * norme_toe_to_heel / 190

            # Calculate the MOS_ML
            self.MOS_ML_x = float(vector_left_foot_BOSML[0] - self.trial.df_vicon_filtered.loc[self.frame_lead_foot_heel_strike_after, 'XCOM_Left_PosX'])
            self.MOS_ML_y = float(vector_left_foot_BOSML[1] - self.trial.df_vicon_filtered.loc[self.frame_lead_foot_heel_strike_after, 'XCOM_Left_PosY'])
            self.MOS_ML_z = float(vector_left_foot_BOSML[2] - self.trial.df_vicon_filtered.loc[self.frame_lead_foot_heel_strike_after, 'XCOM_Left_PosZ'])

        # For the case if the right foot is the lead foot
        if self.lead_foot == 'Right':
            # Calculate MOS_AP & MOS_ML for the frame self.frame_lead_foot_heel_strike_after

            self.MOS_AP_x = float(self.trial.df_vicon_filtered.loc[self.frame_lead_foot_heel_strike_after, 'MOS_AP_Right_PosX'])
            self.MOS_AP_y = float(self.trial.df_vicon_filtered.loc[self.frame_lead_foot_heel_strike_after, 'MOS_AP_Right_PosY'])
            self.MOS_AP_z = float(self.trial.df_vicon_filtered.loc[self.frame_lead_foot_heel_strike_after, 'MOS_AP_Right_PosZ'])

            # Calculate BOSML_MTP_Joint - Extrapolated metatarsophalangeal joint  with Toe and Heel position - maximum lateral excursion
            vector_right_foot_toe = np.array(
                [self.trial.df_vicon_filtered.loc[self.frame_lead_foot_heel_strike_after, (preparation.name_right_toe_marker, 'PosX')] / 1000.,
                 self.trial.df_vicon_filtered.loc[self.frame_lead_foot_heel_strike_after, (preparation.name_right_toe_marker, 'PosY')] / 1000.,
                 self.trial.df_vicon_filtered.loc[self.frame_lead_foot_heel_strike_after, (preparation.name_right_toe_marker, 'PosZ')] / 1000., ])

            vector_right_foot_heel = np.array(
                [self.trial.df_vicon_filtered.loc[self.frame_lead_foot_heel_strike_after, (preparation.name_right_heel_marker, 'PosX')] / 1000.,
                 self.trial.df_vicon_filtered.loc[self.frame_lead_foot_heel_strike_after, (preparation.name_right_heel_marker, 'PosY')] / 1000.,
                 self.trial.df_vicon_filtered.loc[self.frame_lead_foot_heel_strike_after, (preparation.name_right_heel_marker, 'PosZ')] / 1000., ])

            norme_toe_to_heel = np.linalg.norm(vector_right_foot_toe - vector_right_foot_heel)

            vector_right_foot_BOSML = vector_right_foot_heel + ((vector_right_foot_toe - vector_right_foot_heel) * (13 / 19)) + \
                                      np.dot(((vector_right_foot_toe - vector_right_foot_heel) / norme_toe_to_heel), (np.array([0, 0, 1]))) * 30 * \
                                      norme_toe_to_heel / 190

            # Calculate the MOS_ML
            self.MOS_ML_x = float(vector_right_foot_BOSML[0] - self.trial.df_vicon_filtered.loc[self.frame_lead_foot_heel_strike_after, 'XCOM_Right_PosX'])
            self.MOS_ML_y = float(vector_right_foot_BOSML[1] - self.trial.df_vicon_filtered.loc[self.frame_lead_foot_heel_strike_after, 'XCOM_Right_PosY'])
            self.MOS_ML_z = float(vector_right_foot_BOSML[2] - self.trial.df_vicon_filtered.loc[self.frame_lead_foot_heel_strike_after, 'XCOM_Right_PosZ'])

        # Quality of MOS
        if self.MOS_AP_y < 0:
            self.Quality_MOS_AP_y = 'negative'
        elif self.MOS_AP_y > 0.7:
            self.Quality_MOS_AP_y = 'outside'
        else:
            self.Quality_MOS_AP_y = 'normal'

        if self.lead_foot == 'Left':
            if self.MOS_ML_x < 0:
                self.Quality_MOS_ML_x = 'negative'
            elif self.MOS_ML_x > 0.35:
                self.Quality_MOS_ML_x = 'outside'
            else:
                self.Quality_MOS_ML_x = 'normal'

        if self.lead_foot == 'Right':
            self.MOS_ML_x = - self.MOS_ML_x
            if self.MOS_ML_x < 0:
                self.Quality_MOS_ML_x = 'negative'
            elif self.MOS_ML_x > 0.35:
                self.Quality_MOS_ML_x = 'outside'
            else:
                self.Quality_MOS_ML_x = 'normal'

        # Archived trial control
        self.Number_Archived_Trial = None
        self.Reasons_Archived_Trial = None

        # """
        # For plots
        # """
        # if plot_it or save_plot:
        #     plt.plot(df_speed_lead_foot_heel_strike_after.index, df_speed_lead_foot_heel_strike_after.to_numpy(), color='k', label="speed lead foot heel strike after (all axis)")
        #     plt.plot(df_speed_lead_foot_toe_strike_after.index, df_speed_lead_foot_toe_strike_after.to_numpy(), color='navy', label="speed lead foot toe strike after (all axis)")
        #     plt.plot(df_speed_head_lateral.index, df_speed_head_lateral.to_numpy(), color='maroon', label="speed head lateral (axis x)")
        #     plt.plot(df_speed_head_longitudinal.index, df_speed_head_longitudinal.to_numpy(), color='deeppink', label="speed head longitudinal (axis y)")
        #     plt.axvline(x=self.frame_lead_foot_heel_strike_after, color='y', label="lead foot heel strike after")
        #     plt.axvline(x=self.frame_lead_foot_toe_off_before, color='r', label="lead foot toe off before")
        #     plt.axvline(x=self.frame_applomb_lead_foot, color='g', label="applomb lead foot")
        #     plt.axvline(x=self.frame_trail_foot_heel_strike_after, color='b', label="trail foot heel strike after")
        #     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #     if plot_it:
        #         plt.show()
        #     if save_plot:
        #         plt.savefig(self.save_plot_path + 'Speed_Head_Heel_Toe.png', bbox_inches='tight', dpi=300)
        #         plt.close('all')

    # Calculate the head orientation variables (pitch in degree. When positive, look up; when negative, look down)
    def calculate_head_orientation(self, plot_it=False, save_plot=False):
        self.head_pitch_frame_lead_foot_toe_off_before = \
            analysis.convert_quaternion_to_pitch(analysis.get_quaternion(self.trial.df_vicon_structured, 'HeadCalibrated', self.frame_lead_foot_toe_off_before))
        self.head_pitch_frame_applomb_lead_foot = \
            analysis.convert_quaternion_to_pitch(analysis.get_quaternion(self.trial.df_vicon_structured, 'HeadCalibrated', self.frame_applomb_lead_foot))
        self.head_pitch_frame_trail_foot_heel_strike_after = \
            analysis.convert_quaternion_to_pitch(analysis.get_quaternion(self.trial.df_vicon_structured, 'HeadCalibrated', self.frame_trail_foot_heel_strike_after))

        # For plots
        if plot_it or save_plot:
            all_pitch = []
            for i in range(len(self.trial.df_vicon_structured.index)):
                all_pitch.append(analysis.convert_quaternion_to_pitch(
                    analysis.get_quaternion(self.trial.df_vicon_structured, 'HeadCalibrated', i)))
            all_pitch = np.array(all_pitch)
            plt.plot(self.trial.df_vicon_structured.index, all_pitch, color='k', label="pitch in degree")
            plt.axvline(x=self.frame_lead_foot_toe_off_before, color='r', label="lead foot toe off before")
            plt.axvline(x=self.frame_applomb_lead_foot, color='g', label="applomb lead foot")
            plt.axvline(x=self.frame_trail_foot_heel_strike_after, color='b', label="trail foot heel strike after")
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            if plot_it:
                plt.show()
            if save_plot:
                plt.savefig(self.save_plot_path + 'Head_Orientation.png', bbox_inches='tight', dpi=300)
                plt.close('all')

    # For one trial, check if there are some folders with the same in the folder Archived
    def calculate_archived_data(self, plot_it=False, save_plot=False):
        # print(ppretty(self.trial, seq_length=10))

        # Check if there are some folders with the same in the folder Archived
        session_folder_path = self.trial.session.session_folder_path
        session_archived_folder_path = session_folder_path + '/Archived'
        trial_name = 'Trial_' + str(self.trial.config.index) + '_Crossing'

        number_trial_archived_found = 0
        trial_archived_folders = []
        for path in os.listdir(session_archived_folder_path):
            if trial_name in path:
                number_trial_archived_found += 1
                trial_archived_folders.append(session_archived_folder_path + '/' + path);

        self.Number_Archived_Trial = number_trial_archived_found

        # Grouping all the comment.txt from each found archived folder
        reasons_archive = ''
        index = 0
        for folder in trial_archived_folders:
            # print(folder)
            comment_file_path = folder + '/Comment/Comment.txt'
            comment_file = open(comment_file_path, mode='r', encoding='utf-8')
            comment_text = comment_file.read()
            reasons_archive += str(index) + ': ' + comment_text + '; '
            index += 1

        self.Reasons_Archived_Trial = reasons_archive


    # Generate the dictionary of spatial temporal variables result for the crossing
    def generate_spatial_temporal_variables_result(self):
        self.dic_crossing_variables = {'lead_foot': self.lead_foot,
                                       'penultimate_foot_placement': self.penultimate_foot_placement,
                                       'penultimate_foot_placement_in_range': self.penultimate_foot_placement_in_range,
                                       'final_foot_placement': self.final_foot_placement,
                                       'final_foot_placement_in_range': self.final_foot_placement_in_range,
                                       'lead_vertical_toe_clearance': self.lead_vertical_toe_clearance,
                                       'lead_vertical_toe_clearance_in_range': self.lead_vertical_toe_clearance_in_range,
                                       'trail_vertical_toe_clearance': self.trail_vertical_toe_clearance,
                                       'trail_vertical_toe_clearance_in_range': self.trail_vertical_toe_clearance_in_range,
                                       'lead_foot_placement_toe': self.lead_foot_placement_toe,
                                       'lead_foot_placement_toe_in_range': self.lead_foot_placement_toe_in_range,
                                       'lead_foot_placement_heel': self.lead_foot_placement_heel,
                                       'lead_foot_placement_heel_in_range': self.lead_foot_placement_heel_in_range,
                                       'trail_foot_placement_toe': self.trail_foot_placement_toe,
                                       'trail_foot_placement_toe_in_range': self.trail_foot_placement_toe_in_range,
                                       'trail_foot_placement_heel': self.trail_foot_placement_heel,
                                       'trail_foot_placement_heel_in_range': self.trail_foot_placement_heel_in_range,
                                       'step_length_crossing': self.step_length_crossing,
                                       'step_length_crossing_in_range': self.step_length_crossing_in_range,
                                       'step_width_crossing': self.step_width_crossing,
                                       'step_width_crossing_in_range': self.step_width_crossing_in_range,
                                       'double_support_before_crossing': self.double_support_before_crossing,
                                       'double_support_before_crossing_in_range': self.double_support_before_crossing_in_range,
                                       'single_support_trail': self.single_support_trail,
                                       'single_support_trail_in_range': self.single_support_trail_in_range,
                                       'double_support_crossing': self.double_support_crossing,
                                       'double_support_crossing_in_range': self.double_support_crossing_in_range,
                                       'single_support_lead': self.single_support_lead,
                                       'single_support_lead_in_range': self.single_support_lead_in_range,
                                       'mean_speed_lead_foot_heel_strike_after': self.mean_speed_lead_foot_heel_strike_after,
                                       'mean_speed_lead_foot_toe_strike_after': self.mean_speed_lead_foot_toe_strike_after,
                                       'mean_speed_head_lateral': self.mean_speed_head_lateral,
                                       'mean_speed_head_lateral_in_range': self.mean_speed_head_lateral_in_range,
                                       'sd_speed_head_lateral': self.sd_speed_head_lateral,
                                       'sd_speed_head_lateral_in_range': self.sd_speed_head_lateral_in_range,
                                       'mean_speed_head_longitudinal': self.mean_speed_head_longitudinal,
                                       'mean_speed_head_longitudinal_in_range': self.mean_speed_head_longitudinal_in_range,
                                       'sd_speed_head_longitudinal': self.sd_speed_head_longitudinal,
                                       'sd_speed_head_longitudinal_in_range': self.sd_speed_head_longitudinal_in_range,
                                       'head_pitch_frame_lead_foot_toe_off_before': self.head_pitch_frame_lead_foot_toe_off_before,
                                       'head_pitch_frame_lead_foot_toe_off_before_in_range': self.head_pitch_frame_lead_foot_toe_off_before_in_range,
                                       'head_pitch_frame_applomb_lead_foot': self.head_pitch_frame_applomb_lead_foot,
                                       'head_pitch_frame_applomb_lead_foot_in_range': self.head_pitch_frame_applomb_lead_foot_in_range,
                                       'head_pitch_frame_trail_foot_heel_strike_after': self.head_pitch_frame_trail_foot_heel_strike_after,
                                       'head_pitch_frame_trail_foot_heel_strike_after_in_range': self.head_pitch_frame_trail_foot_heel_strike_after_in_range,
                                       'can_calculate_result': self.can_calculate_result,
                                       'error_code': self.error_code,
                                       'frame_lead_foot_heel_strike_after': self.frame_lead_foot_heel_strike_after,
                                       'MOS_AP_Y': self.MOS_AP_y,
                                       'MOS_AP_Y_in_range': self.MOS_AP_y_in_range,
                                       'MOS_ML_X': self.MOS_ML_x,
                                       'MOS_ML_X_in_range': self.MOS_ML_x_in_range,
                                       'Quality_MOS_AP_Y': self.Quality_MOS_AP_y,
                                       'Quality_MOS_ML_X': self.Quality_MOS_ML_x,
                                       'Number_Archived_Trial': self.Number_Archived_Trial,
                                       'Reasons_Archived_Trial': self.Reasons_Archived_Trial,
                                       }


# In case of analysing all the session in one folder
for path in os.listdir(root_data_path):
    full_path = os.path.join(root_data_path, path)
    participant_name = path.rsplit('_', 1)[0]
    session_name = path.rsplit('_', 1)[1]

    # Create a session
    session = CrossingSessionData(root_data_path, participant_name, session_name)

    # Basic analyse each trial in Archived folder
    for trial in session.list_archived_trials:
        # Create the raw data(all data assembled in a csv), stuctured data(2 layer of index for a fast indexing), filtered data(with interpolation and filer)
        trial.data_preparation()
        trial.export_vicon_to_c3d()
        print('Archived Trial ' + str(trial.trial_folder_path) + ' analyse completed')

    # Analyse each trial in Data folder
    for trial in session.list_trials:

        # Analyse locomotion
        if trial.config.category not in ['PWS', 'Adaptation', 'FamiStat', 'FamiDyn']:

            # Create the raw data(all data assembled in a csv), stuctured data(2 layer of index for a fast indexing), filtered data(with interpolation and filer)
            trial.data_preparation()

            if trial.load_vicon_filtered():
                trial.export_vicon_to_c3d()
                trial.crossing_analysis_routine(verbose=True, plot_it=False, save_plot=True)
                print('Trial ' + str(trial.config.index) + ' analyse completed')

    # Export session result
    session.export_session_result()
