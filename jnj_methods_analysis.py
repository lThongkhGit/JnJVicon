import scipy.interpolate
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation
from numpy import linalg
from transforms3d.euler import quat2euler
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.signal


# Calculate crossing variables for a crossing using the tracking data of left foot toe, left foot heel, right foot toe and right foot heel
# RHEE_, LHEE_, RTOE_, LTOE_ represent tracker data in format numpy list
# lead_foot represents the lead foot for the crossing (the foot that first crossed the obstacle)
# idx_applomb_LTOE, idx_applomb_RTOE represent the index of frame of applomb for left and right toe during the crossing (the moment that the foot is closest to the obstacle in y axis)
#  obstacle_name represents the name of the obstacle (Step051 / Step052 / Step151 / Step152)
# pos_top_center_obstacle represents the top center position of the obstacle (mean value of the first 3 seconds of the trial to avoid markers' inversion)
def crossing_calculation(RHEE_, LHEE_, RTOE_, LTOE_, lead_foot, idx_applomb_LTOE, idx_applomb_RTOE, obstacle_name, pos_top_center_obstacle, plot_it=False, save_plot=True, information='',
                         save_plot_path='', verbose=False):
    '''
    A utiliser pour detecter les appuis dans TAP 2
    Essaye de combiner les deux methodes de detection
    - la pose des appuis stable pour diminuer l'espace de recherche
    - une detection plus fine autour de cet evene   ment

    pos_top_center_obstacle : the top center position of the obstacle
    pos_top_center_obstacle[2] : the top center position of the obstacle - axis z
    pos_top_center_obstacle[1] : the top center position of the obstacle - axis y
    obstacle_name : the name of the obstacle rigidbody

    Librement inspire de l'article ci-dessous
    https://www.sciencedirect.com/science/article/pii/S0966636206001068
    '''
    # debug_plots=True
    # Recuperer les donnees (list of position)
    RHEE, LHEE, RTOE, LTOE = RHEE_, LHEE_, RTOE_, LTOE_

    LMID = (LHEE + LTOE) / 2.
    RMID = (RHEE + RTOE) / 2.

    # The speed in the vertical direction
    speed_RHEE = np.gradient(RHEE[:, 3])
    speed_LHEE = np.gradient(LHEE[:, 3])
    speed_RTOE = np.gradient(RTOE[:, 3])
    speed_LTOE = np.gradient(LTOE[:, 3])

    speed_LMID = np.gradient(LMID[:, 3])
    speed_RMID = np.gradient(RMID[:, 3])

    # Detecter la phase pendant laquelle les pas peuvent etre detectes pour exclure debut/fin de l'essai
    centerfeets = (LMID + RMID) / 2
    # dist_feets_step = np.sqrt((centerfeets[:, 2] - pos_top_center_obstacle[1]) ** 2)
    dist_feets_step = np.abs(centerfeets[:, 2] - pos_top_center_obstacle[2])
    period_data = [np.where(dist_feets_step < 2000.)[0][0], np.where(dist_feets_step < 2000.)[0][-1]]

    # Avoir les appuis sur le sol de chacun des pieds (phases stables, uniquement avec le toe)
    # avec une precision mauvaise (ce ne sont pas Heel strike et Toe off mais des approximations)
    # c'est utilise pour detecter plus tard HS et TO dans une fenetre temporelle plus restreinte
    stable_stepL, speed_toeL, starts_tmpL, stops_tmpL, stable_stepR, speed_toeR, starts_tmpR, stops_tmpR = steps_detection_low_precision(LTOE, RTOE, plotit=plot_it, save_plot=save_plot,
                                                                                                                                         save_plot_path=save_plot_path)

    # Detecter les stable step autour de la marche (step)
    # Left stable steps
    difff = stable_stepL - idx_applomb_LTOE
    difffinf = difff < 0
    stables_steps_around_step_L = stable_stepL[np.argmax(difff[difffinf]):np.argmax(difff[difffinf]) + 2]

    # print('stables_steps_around_step_L--------------------------------')
    # print(stables_steps_around_step_L)
    # Problem: there is not enough steps between two crossing !!!!!!
    if len(stables_steps_around_step_L) < 2:
        print('Problem: there is not enough steps between two crossing !!!!!!')
        return None

    # Right stable steps
    difff = stable_stepR - idx_applomb_RTOE  # regarder les deux derniers stable steps
    difffinf = difff < 0
    stables_steps_around_step_R = stable_stepR[np.argmax(difff[difffinf]):np.argmax(difff[difffinf]) + 2]

    # print('stables_steps_around_step_R--------------------------------')
    # print(stables_steps_around_step_R)

    # Problem: there is not enough steps between two crossing !!!!!!
    if len(stables_steps_around_step_R) < 2:
        print('Problem: there is not enough steps between two crossing !!!!!!')
        return None

    # Nouvelle version de detection des appuis
    # TOE off (inspire de https://www.sciencedirect.com/science/article/pii/S0966636206001068)

    # Nouvelle version - toe off
    idx_R_TO_before = TO_detection('R_Foot_(before_applomb)_Lead_F_' + lead_foot + '_' + information, RTOE, speed_RTOE, speed_RMID, stables_steps_around_step_R[0], period_data, plot_it=plot_it,
                                   save_plot=save_plot, save_plot_path=save_plot_path)
    idx_R_TO_after = TO_detection('R_Foot_(after_applomb)_Lead_F_' + lead_foot + '_' + information, RTOE, speed_RTOE, speed_RMID, stables_steps_around_step_R[1], period_data, plot_it=plot_it,
                                  save_plot=save_plot, save_plot_path=save_plot_path)
    idx_L_TO_before = TO_detection('L_Foot_(before_applomb)_Lead_F_' + lead_foot + '_' + information, LTOE, speed_LTOE, speed_LMID, stables_steps_around_step_L[0], period_data, plot_it=plot_it,
                                   save_plot=save_plot, save_plot_path=save_plot_path)
    idx_L_TO_after = TO_detection('L_Foot_(after_applomb)_Lead_F_' + lead_foot + '_' + information, LTOE, speed_LTOE, speed_LMID, stables_steps_around_step_L[1], period_data, plot_it=plot_it,
                                  save_plot=save_plot, save_plot_path=save_plot_path)

    # HEEL STRIKE
    # Nouvelle version
    idx_R_HS_before = HS_detection('R_Foot_(before_applomb)_Lead_F_' + lead_foot + '_' + information, RHEE, speed_RHEE, speed_RMID, stables_steps_around_step_R[0], period_data, plot_it=plot_it,
                                   save_plot=save_plot, save_plot_path=save_plot_path)
    idx_R_HS_after = HS_detection('R_Foot_(after_applomb)_Lead_F_' + lead_foot + '_' + information, RHEE, speed_RHEE, speed_RMID, stables_steps_around_step_R[1], period_data, plot_it=plot_it,
                                  save_plot=save_plot, save_plot_path=save_plot_path)
    idx_L_HS_before = HS_detection('L_Foot_(before_applomb)_Lead_F_' + lead_foot + '_' + information, LHEE, speed_LHEE, speed_LMID, stables_steps_around_step_L[0], period_data, plot_it=plot_it,
                                   save_plot=save_plot, save_plot_path=save_plot_path)
    idx_L_HS_after = HS_detection('L_Foot_(after_applomb)_Lead_F_' + lead_foot + '_' + information, LHEE, speed_LHEE, speed_LMID, stables_steps_around_step_L[1], period_data, plot_it=plot_it,
                                  save_plot=save_plot, save_plot_path=save_plot_path)

    if verbose:
        print('------------ TOE OFF')
        print('R_TO_before: ' + str(idx_R_TO_before))
        print('R_TO_after: ' + str(idx_R_TO_after))
        print('L_TO_before: ' + str(idx_L_TO_before))
        print('L_TO_after: ' + str(idx_L_TO_after))
        print('------------ HEEL STRIKE')
        print('R_HS_before: ' + str(idx_R_HS_before))
        print('R_HS_after: ' + str(idx_R_HS_after))
        print('L_HS_before: ' + str(idx_L_HS_before))
        print('L_HS_after: ' + str(idx_L_HS_after))

    # # Plot the states of heel strike
    # if plot_it or save_plot:
    #     f1 = plt.figure(figsize=(16, 12))
    #     acc = np.gradient(speed_RHEE)
    #     ax1 = plt.subplot(413)
    #     plt.plot(acc, 'k')
    #     if not math.isnan(idx_R_HS_before):
    #         plt.plot(idx_R_HS_before, acc[idx_R_HS_before], 'x')
    #     if not math.isnan(idx_R_HS_after):
    #         plt.plot(idx_R_HS_after, acc[idx_R_HS_after], 'x')
    #     plt.axvline(x=idx_applomb_RTOE)
    #
    #     plt.subplot(411, sharex=ax1)
    #     plt.title('HS')
    #     plt.plot(RHEE[:, 3], 'k')
    #     if not math.isnan(idx_R_HS_before):
    #         plt.plot(idx_R_HS_before, RHEE[idx_R_HS_before, 3], 'x')
    #     if not math.isnan(idx_R_HS_after):
    #         plt.plot(idx_R_HS_after, RHEE[idx_R_HS_after, 3], 'x')
    #     plt.axvline(x=idx_applomb_RTOE)
    #
    #     plt.subplot(412, sharex=ax1)
    #     plt.plot(speed_RHEE, 'k')
    #     plt.plot(speed_RMID, 'r')
    #     if not math.isnan(idx_R_HS_before):
    #         plt.plot(idx_R_HS_before, speed_RHEE[idx_R_HS_before], 'x')
    #     if not math.isnan(idx_R_HS_after):
    #         plt.plot(idx_R_HS_after, speed_RHEE[idx_R_HS_after], 'x')
    #
    #     # plt.plot(idx_R_TO_after, speed_RTOE[idx_R_TO_after], 'x')
    #     plt.axvline(x=stables_steps_around_step_R[0], color='k')
    #     plt.axvline(x=stables_steps_around_step_R[0] - 0.4 * 120, color='k')
    #     plt.axvline(x=stables_steps_around_step_R[0] + 0.4 * 120, color='k')
    #     plt.subplot(414)
    #     # plt.plot(tmp_speed)
    #     # plt.plot(a1,'r')
    #     # plt.plot(a2,'g')
    #     # plt.plot(a3,'b')
    #     # plt.plot(a4,'k--')
    #
    #     if plot_it:
    #         plt.show()
    #
    #     if save_plot:
    #         f1.savefig(save_plot_path + 'HS.png', bbox_inches='tight', dpi=300)
    #         plt.close('all')
    #
    # # Plot the states of the toe off
    # if plot_it or save_plot:
    #     f2 = plt.figure(figsize=(16, 12))
    #     ax1 = plt.subplot(211)
    #     plt.title('TO')
    #     plt.plot(RTOE[:, 3], 'k')
    #     plt.axvline(x=idx_applomb_RTOE)
    #
    #     if not math.isnan(idx_R_TO_before):
    #         plt.plot(idx_R_TO_before, RTOE[idx_R_TO_before, 3], 'x')
    #
    #     if not math.isnan(idx_R_TO_after):
    #         plt.plot(idx_R_TO_after, RTOE[idx_R_TO_after, 3], 'x')
    #
    #     if not math.isnan(idx_L_TO_before):
    #         plt.plot(idx_L_TO_before, RTOE[idx_L_TO_before, 3], '*')
    #
    #     if not math.isnan(idx_L_TO_after):
    #         plt.plot(idx_L_TO_after, RTOE[idx_L_TO_after, 3], '*')
    #
    #     plt.subplot(212, sharex=ax1)
    #     plt.plot(speed_RTOE, 'k')
    #     plt.plot(speed_RMID, 'r')
    #     if not math.isnan(idx_R_TO_before):
    #         plt.plot(idx_R_TO_before, speed_RTOE[idx_R_TO_before], 'x')
    #     if not math.isnan(idx_R_TO_after):
    #         plt.plot(idx_R_TO_after, speed_RTOE[idx_R_TO_after], 'x')
    #     plt.axvline(x=stables_steps_around_step_R[0], color='k')
    #     plt.axvline(x=idx_applomb_RTOE)
    #     # plt.axvline(x=stables_steps_around_step_R[0]-0.4*120, color='k')
    #     # plt.axvline(x=stables_steps_around_step_R[0]+0.4*120, color='k')
    #
    #     if plot_it:
    #         plt.show()
    #
    #     if save_plot:
    #         f2.savefig(save_plot_path + 'TO.png', bbox_inches='tight', dpi=300)
    #         plt.close('all')

    # draw foot and crossing
    if plot_it or save_plot:
        f3 = plt.figure(figsize=(16, 12))
        # Left foot and right foot on the y-time
        ax1 = plt.subplot(511)
        plt.title('Result Lead foot=' + lead_foot)
        plt.plot(RHEE[:, 2], 'k--', label='RHEE')
        plt.plot(RTOE[:, 2], 'r--', label='RTOE')
        plt.axvline(x=idx_applomb_RTOE)
        # plt.legend(loc='best')
        # plt.subplot(512, sharex=ax1)
        plt.plot(LHEE[:, 2], 'k', label='LHEE')
        plt.plot(LTOE[:, 2], 'r', label='LTOE')
        plt.axvline(x=idx_applomb_LTOE)
        axes = plt.gca()
        axes.set_xlim([period_data[0], period_data[1]])
        plt.legend(loc='best')

        # Right foot on the z-time
        plt.subplot(512, sharex=ax1)
        plt.plot(RHEE[:, 3], 'k', label='RHEE')
        if not math.isnan(idx_R_HS_before):
            plt.plot(idx_R_HS_before, RHEE[idx_R_HS_before, 3], 'g*')
            plt.text(idx_R_HS_before, RHEE[idx_R_HS_before, 3], 'idx_R_HS_before')
        if not math.isnan(idx_R_HS_after):
            plt.plot(idx_R_HS_after, RHEE[idx_R_HS_after, 3], 'g*')
            plt.text(idx_R_HS_after, RHEE[idx_R_HS_after, 3], 'idx_R_HS_after')

        plt.plot(RTOE[:, 3], 'r', label='RTOE')
        if not math.isnan(idx_R_TO_before):
            plt.plot(idx_R_TO_before, RTOE[idx_R_TO_before, 3], 'gx')
            plt.text(idx_R_TO_before, RTOE[idx_R_TO_before, 3], 'idx_R_TO_before')

        if not math.isnan(idx_R_TO_after):
            plt.plot(idx_R_TO_after, RTOE[idx_R_TO_after, 3], 'gx')
            plt.text(idx_R_TO_after, RTOE[idx_R_TO_after, 3], 'idx_R_TO_after')

        plt.axvline(x=idx_applomb_RTOE)
        for el in stables_steps_around_step_R:
            plt.axvline(x=el, color='k')
        plt.legend(loc='best')

        # Left foot on the z-time
        plt.subplot(513, sharex=ax1)
        plt.plot(LHEE[:, 3], 'k', label='LHEE')
        if not math.isnan(idx_L_HS_before):
            plt.plot(idx_L_HS_before, LHEE[idx_L_HS_before, 3], 'g*')
            plt.text(idx_L_HS_before, LHEE[idx_L_HS_before, 3], 'idx_L_HS_before')
        if not math.isnan(idx_L_HS_after):
            plt.plot(idx_L_HS_after, LHEE[idx_L_HS_after, 3], 'g*')
            plt.text(idx_L_HS_after, LHEE[idx_L_HS_after, 3], 'idx_L_HS_after')

        plt.plot(LTOE[:, 3], 'r', label='LTOE')
        if not math.isnan(idx_L_TO_before):
            plt.plot(idx_L_TO_before, LTOE[idx_L_TO_before, 3], 'gx')
            plt.text(idx_L_TO_before, LTOE[idx_L_TO_before, 3], 'idx_L_TO_before')
        if not math.isnan(idx_L_TO_after):
            plt.plot(idx_L_TO_after, LTOE[idx_L_TO_after, 3], 'gx')
            plt.text(idx_L_TO_after, LTOE[idx_L_TO_after, 3], 'idx_L_TO_after')

        for el in stables_steps_around_step_L:
            plt.axvline(x=el, color='k')
        plt.axvline(x=idx_applomb_LTOE)
        plt.legend(loc='best')

        # Left and right foot on the y-z
        ax6 = plt.subplot(5, 1, (4, 5))
        # afficher le sol
        limit_ground = 3 * 1000.
        ax6.fill([pos_top_center_obstacle[1] - limit_ground, pos_top_center_obstacle[1] + limit_ground, pos_top_center_obstacle[1] + limit_ground, pos_top_center_obstacle[1] - limit_ground],
                 [-0.1 * 1000., -0.1 * 1000., 0, 0], fill=False, hatch='\\')
        # Afficher la marche
        if '05' in obstacle_name:
            plt.plot([pos_top_center_obstacle[1], pos_top_center_obstacle[1]], [pos_top_center_obstacle[2], pos_top_center_obstacle[2] - 50], 'b')
        if '15' in obstacle_name:
            plt.plot([pos_top_center_obstacle[1], pos_top_center_obstacle[1]], [pos_top_center_obstacle[2], pos_top_center_obstacle[2] - 150], 'b')

        # Afficher les courbes des pieds
        # le droit en pointille
        plt.plot(RHEE[:, 2], RHEE[:, 3], 'k--', label='RHEE')
        plt.plot(RTOE[:, 2], RTOE[:, 3], 'r--', label='RTOE')

        # left
        plt.plot(LHEE[:, 2], LHEE[:, 3], 'k', label='LHEE')
        plt.plot(LTOE[:, 2], LTOE[:, 3], 'r', label='LTOE')

        # afficher les pieds (de HS)
        if not math.isnan(idx_L_TO_before) and not math.isnan(idx_L_HS_before):
            drawfoot([LHEE[idx_L_HS_before, 2], LHEE[idx_L_HS_before, 3]], [LTOE[idx_L_TO_before, 2], LTOE[idx_L_TO_before, 3]], color='g', offset=30.)
        if not math.isnan(idx_L_TO_after) and not math.isnan(idx_L_HS_after):
            drawfoot([LHEE[idx_L_HS_after, 2], LHEE[idx_L_HS_after, 3]], [LTOE[idx_L_TO_after, 2], LTOE[idx_L_TO_after, 3]], color='g', offset=30.)
        if not math.isnan(idx_R_TO_before) and not math.isnan(idx_R_HS_before):
            drawfoot([RHEE[idx_R_HS_before, 2], RHEE[idx_R_HS_before, 3]], [RTOE[idx_R_TO_before, 2], RTOE[idx_R_TO_before, 3]], color='m', offset=30.)
        if not math.isnan(idx_R_TO_after) and not math.isnan(idx_R_HS_after):
            drawfoot([RHEE[idx_R_HS_after, 2], RHEE[idx_R_HS_after, 3]], [RTOE[idx_R_TO_after, 2], RTOE[idx_R_TO_after, 3]], color='m', offset=30.)

        axes = plt.gca()
        axes.set_xlim([pos_top_center_obstacle[1] - 2000., pos_top_center_obstacle[1] + 2000.])
        axes.set_ylim([-0.1 * 1000., 600])
        plt.legend(loc='best')

        if plot_it:
            plt.show()

        if save_plot:
            f3.savefig(save_plot_path + 'Result_Lead_foot_' + lead_foot + '.png', bbox_inches='tight', dpi=300)
            plt.close('all')

    if np.isnan(idx_R_TO_before) or np.isnan(idx_R_TO_before) or np.isnan(idx_R_TO_after) or np.isnan(
            idx_L_TO_before) or np.isnan(idx_L_TO_after) or np.isnan(idx_R_HS_before) or np.isnan(
        idx_R_HS_after) or np.isnan(idx_L_HS_before) or np.isnan(idx_L_HS_after):
        return None

    if lead_foot == 'Left':  # L lead
        # Anterior-posterior distance between the toe of the lead foot and the obstacle before crossing (y axis)
        penultimate_foot_placement = LTOE[idx_L_TO_before, 2] - pos_top_center_obstacle[1]
        # Anterior-posterior distance between the toe of the trail foot and the obstacle before crossing (y axis)
        final_foot_placement = RTOE[idx_R_TO_before, 2] - pos_top_center_obstacle[1]
        # Vertical distance between the toe of the lead foot and the top of the obstacle while crossing (z axis)
        lead_vertical_toe_clearance = LTOE[idx_applomb_LTOE, 3] - pos_top_center_obstacle[2]
        # Vertical distance between the toe of the trail foot and the top of the obstacle while crossing (z axis)
        trail_vertical_toe_clearance = RTOE[idx_applomb_RTOE, 3] - pos_top_center_obstacle[2]
        # Anterior-posterior distance between the toe of the lead foot and the obstacle after crossing (y axis)
        lead_foot_placement_toe = LTOE[idx_L_TO_after, 2] - pos_top_center_obstacle[1]
        # Anterior-posterior distance between the heel of the lead foot and the obstacle after crossing (y axis)
        lead_foot_placement_heel = LHEE[idx_L_HS_after, 2] - pos_top_center_obstacle[1]
        # Anterior-posterior distance between the toe of the trail foot and the obstacle after crossing (y axis)
        trail_foot_placement_toe = RTOE[idx_R_TO_after, 2] - pos_top_center_obstacle[1]
        # Anterior-posterior distance between the heel of the trail foot and the obstacle after crossing (y axis)
        trail_foot_placement_heel = RHEE[idx_R_HS_after, 2] - pos_top_center_obstacle[1]
        # Lateral distance from the heel of the trial foot before crossing to the heel of the lead foot after crossing (x axis)
        step_width_crossing = np.abs(LHEE[idx_L_HS_after, 1] - RHEE[idx_R_HS_before, 1])
        # The last double support time before the crossing (from the heel strike of the trail foot before crossing to the toe off of the lead foot before crossing)
        double_support_before_crossing = (idx_L_TO_before - idx_R_HS_before) * (1 / 120.)
        # The single support time of the trail foot before the crossing (from the toe off of the lead foot before crossing to the heel strike of the lead foot after crossing)
        single_support_trail = (idx_L_HS_after - idx_L_TO_before) * (1 / 120.)
        # The double support time during the crossing (from the heel strike of the lead foot after crossing to the toe off of the trail foot before crossing)
        double_support_crossing = (idx_R_TO_before - idx_L_HS_after) * (1 / 120.)
        # The single support time of the lead foot after the crossing (from the toe off of the trail foot before crossing to the heel strike of the trail foot after crossing)
        single_support_lead = (idx_R_HS_after - idx_R_TO_before) * (1 / 120.)
        # The frame of Toe Off lead foot before the crossing
        idx_lead_foot_TO_before = idx_L_TO_before
        # The frame of Heel Strike lead foot after the crossing
        idx_lead_foot_HS_after = idx_L_HS_after
        # The frame of Heel Strike trail foot after the crossing
        idx_trail_foot_HS_after = idx_R_HS_after

    elif lead_foot == 'Right':  # R lead
        # Anterior-posterior distance between the toe of the lead foot and the obstacle before crossing (y axis)
        penultimate_foot_placement = RTOE[idx_R_TO_before, 2] - pos_top_center_obstacle[1]
        # Anterior-posterior distance between the toe of the trail foot and the obstacle before crossing (y axis)
        final_foot_placement = LTOE[idx_L_TO_before, 2] - pos_top_center_obstacle[1]
        # Vertical distance between the toe of the lead foot and the top of the obstacle while crossing (z axis)
        lead_vertical_toe_clearance = RTOE[idx_applomb_RTOE, 3] - pos_top_center_obstacle[2]
        # Vertical distance between the toe of the trail foot and the top of the obstacle while crossing (z axis)
        trail_vertical_toe_clearance = LTOE[idx_applomb_LTOE, 3] - pos_top_center_obstacle[2]
        # Anterior-posterior distance between the toe of the lead foot and the obstacle after crossing (y axis)
        lead_foot_placement_toe = RTOE[idx_R_TO_after, 2] - pos_top_center_obstacle[1]
        # Anterior-posterior distance between the heel of the lead foot and the obstacle after crossing (y axis)
        lead_foot_placement_heel = RHEE[idx_R_HS_after, 2] - pos_top_center_obstacle[1]
        # Anterior-posterior distance between the toe of the trail foot and the obstacle after crossing (y axis)
        trail_foot_placement_toe = LTOE[idx_L_TO_after, 2] - pos_top_center_obstacle[1]
        # Anterior-posterior distance between the heel of the trail foot and the obstacle after crossing (y axis)
        trail_foot_placement_heel = LHEE[idx_L_HS_after, 2] - pos_top_center_obstacle[1]
        # Lateral distance from the heel of the trial foot before crossing to the heel of the lead foot after crossing (x axis)
        step_width_crossing = np.abs(RHEE[idx_R_HS_after, 1] - LHEE[idx_L_HS_before, 1])
        # The last double support time before the crossing (from the heel strike of the trail foot before crossing to the toe off of the lead foot before crossing)
        double_support_before_crossing = (idx_R_TO_before - idx_L_HS_before) * (1 / 120.)
        # The single support time of the trail foot before the crossing (from the toe off of the lead foot before crossing to the heel strike of the lead foot after crossing)
        single_support_trail = (idx_R_HS_after - idx_R_TO_before) * (1 / 120.)
        # The double support time during the crossing (from the heel strike of the lead foot after crossing to the toe off of the trail foot before crossing)
        double_support_crossing = (idx_L_TO_before - idx_R_HS_after) * (1 / 120.)
        # The single support time of the lead foot after the crossing (from the toe off of the trail foot before crossing to the heel strike of the trail foot after crossing)
        single_support_lead = (idx_L_HS_after - idx_L_TO_before) * (1 / 120.)
        # The frame of Toe Off lead foot before the crossing
        idx_lead_foot_TO_before = idx_R_TO_before
        # The frame of Heel Strike lead foot after the crossing
        idx_lead_foot_HS_after = idx_R_HS_after
        # The frame of  Heel Strike trail foot after the crossing
        idx_trail_foot_HS_after = idx_L_HS_after

    return penultimate_foot_placement, final_foot_placement, lead_vertical_toe_clearance, trail_vertical_toe_clearance, lead_foot_placement_toe, \
           lead_foot_placement_heel, trail_foot_placement_toe, trail_foot_placement_heel, step_width_crossing, double_support_before_crossing, single_support_lead, \
           double_support_crossing, single_support_trail, idx_lead_foot_TO_before, idx_lead_foot_HS_after, idx_trail_foot_HS_after, True


# Detect all Heel Strike, Toe off frames in a trial for complex terrain, and calculate the precise stable step using these two list
def complex_calculation(RHEE_, LHEE_, RTOE_, LTOE_, plot_it=False, save_plot=True, save_plot_path='', verbose=True):
    # Recuperer les donnees (list of position)
    RHEE, LHEE, RTOE, LTOE = RHEE_, LHEE_, RTOE_, LTOE_

    LMID = (LHEE + LTOE) / 2.
    RMID = (RHEE + RTOE) / 2.

    # The speed in the vertical direction
    speed_RHEE = np.gradient(RHEE[:, 3])
    speed_LHEE = np.gradient(LHEE[:, 3])
    speed_RTOE = np.gradient(RTOE[:, 3])
    speed_LTOE = np.gradient(LTOE[:, 3])

    speed_LMID = np.gradient(LMID[:, 3])
    speed_RMID = np.gradient(RMID[:, 3])

    # Remove first 10 frames and last 10 frames
    period_data = [10, len(RHEE) - 10]

    # Avoir les appuis sur le sol de chacun des pieds (phases stables, uniquement avec le toe)
    # avec une precision mauvaise (ce ne sont pas Heel strike et Toe off mais des approximations)
    # c'est utilise pour detecter plus tard HS et TO dans une fenetre temporelle plus restreinte
    stable_stepL, speed_toeL, starts_tmpL, stops_tmpL, stable_stepR, speed_toeR, starts_tmpR, stops_tmpR = steps_detection_low_precision(LTOE, RTOE, plotit=plot_it, save_plot=save_plot,save_plot_path=save_plot_path)

    print('lowP_stable_stepL: ', stable_stepL)
    print('lowP_stable_stepR: ', stable_stepR)

    # HEEL STRIKE
    list_frame_HS_left = []
    list_frame_HS_right = []

    for i in range(len(stable_stepL)):
        frame_hs = HS_detection('L_HS_' + str(i), LHEE, speed_LHEE, speed_LMID, stable_stepL[i], period_data)
        # Add the heel strike frame to the list if the frame is not Nan and the frame is at least 50 frames after the precedent heel strike frame
        if not np.isnan(frame_hs):
            if not (len(list_frame_HS_left) > 0 and frame_hs - 50 < list_frame_HS_left[-1]):
                list_frame_HS_left.append(frame_hs)

    for i in range(len(stable_stepR)):
        frame_hs = HS_detection('R_HS_' + str(i), RHEE, speed_RHEE, speed_RMID, stable_stepR[i], period_data)
        # Add the heel strike frame to the list if the frame is not Nan and the frame is at least 50 frames after the precedent heel strike frame
        if not np.isnan(frame_hs):
            if not (len(list_frame_HS_right) > 0 and frame_hs - 50 < list_frame_HS_right[-1]):
                list_frame_HS_right.append(frame_hs)

    if verbose:
        print('------------ HEEL STRIKE LEFT FOOT')
        print(list_frame_HS_left)
        print('------------ HEEL STRIKE RIGHT FOOT')
        print(list_frame_HS_right)

    # TOE OFF
    list_frame_TO_left = []
    list_frame_TO_right = []

    for i in range(len(stable_stepL)):
        frame_to = TO_detection('L_TO_' + str(i), LTOE, speed_LTOE, speed_LMID, stable_stepL[i], period_data)
        # Add the toe off frame to the list if the frame is not Nan and the frame is at least 50 frames after the precedent toe off frame
        if not np.isnan(frame_to):
            if not (len(list_frame_TO_left) > 0 and frame_to - 50 < list_frame_TO_left[-1]):
                list_frame_TO_left.append(frame_to)

    for i in range(len(stable_stepR)):
        frame_to = TO_detection('R_TO_' + str(i), RTOE, speed_RTOE, speed_RMID, stable_stepR[i], period_data)
        # Add the toe off frame to the list if the frame is not Nan and the frame is at least 50 frames after the precedent toe off frame
        if not np.isnan(frame_to):
            if not (len(list_frame_TO_right) > 0 and frame_to - 50 < list_frame_TO_right[-1]):
                list_frame_TO_right.append(frame_to)

    if verbose:
        print('------------ TOE OFF LEFT FOOT')
        print(list_frame_TO_left)
        print('------------ TOE OFF RIGHT FOOT')
        print(list_frame_TO_right)

    # # Calculate the precise stable step frame using the list_frame_HS and list_frame_TO
    # list_frame_stable_precise_left = []
    # list_frame_stable_precise_right = []
    #
    # if len(list_frame_HS_left) == len(list_frame_TO_left):
    #     for i in range(len(list_frame_HS_left)):
    #         list_frame_stable_precise_left.append((list_frame_TO_left[i] + list_frame_HS_left[i]) / 2)
    #
    # if len(list_frame_HS_right) == len(list_frame_TO_right):
    #     for i in range(len(list_frame_HS_right)):
    #         list_frame_stable_precise_right.append((list_frame_TO_right[i] - list_frame_HS_right[i]) / 2)
    #
    # if verbose:
    #     print('------------ PRECISE STABLE STEP LEFT FOOT')
    #     print(list_frame_stable_precise_left)
    #     print('------------ PRECISE STABLE STEP RIGHT FOOT')
    #     print(list_frame_stable_precise_right)

    if len(list_frame_HS_left) == 0 or len(list_frame_HS_right) == 0 or len(list_frame_TO_left) == 0 or len(list_frame_TO_right) == 0:
        return None

    return list_frame_HS_left, list_frame_HS_right, list_frame_TO_left, list_frame_TO_right


# This is a private method which is only used internally to detect heel strike
def HS_detection(information, HEE_pos, HEEE_speed, MID_speed, idx_stable, period_data, plot_it=False, save_plot=False, save_plot_path=''):
    steps_stable = 30  # frames pedant lesquelles je calcule la position moyenne du pied
    percent = 0.6  # 0.45 initialement pour TAP
    win_before_heel_strike = 0.7 * 120.
    distance_from_stable = 90.  # 60 initialement pour TAP

    median_pos_around_step = np.nanmean(HEE_pos[idx_stable - steps_stable:idx_stable + steps_stable, :], axis=0, dtype='float32')  # position moyenne du pied pendant la phase stable

    # rule1 = HEE_pos[:, 2]<(np.nanmean(HEE_pos[:, 2])*percent)  # etre proche du sol en position Z
    rule1 = HEE_pos[:, 3] < (np.nanmean(HEE_pos[period_data[0]:period_data[1], 3], dtype='float32') * percent)  # version permettant d'enlever les valeurs negatives au debut
    rule2 = np.logical_and(np.arange(HEE_pos.shape[0]) < idx_stable, np.arange(HEE_pos.shape[0]) > (idx_stable - int(win_before_heel_strike)))  # etre avant le stable step
    rule3 = np.abs(HEE_pos[:, 2] - median_pos_around_step[2]) < distance_from_stable  # etre proche en distance du stable step (en Y)

    # print('rule1: ', rule1, 'rule2: ', rule2, 'rule3', rule3)

    rule123 = np.logical_and.reduce((rule1, rule2, rule3))

    tmp_MID_speed = MID_speed.copy()
    tmp_MID_speed[np.logical_not(rule123)] = np.NaN

    idx_heel_strike = np.NaN if np.all(np.isnan(tmp_MID_speed)) else np.nanargmin(tmp_MID_speed)

    if plot_it or save_plot:
        f1 = plt.figure(figsize=(16, 12))
        ax1 = plt.subplot(311)
        plt.title('heel_strike_detection_' + information)
        plt.plot(MID_speed, color='k', label='MID_speed')
        plt.plot(tmp_MID_speed, color='b', label='tmp_MID_speed')
        plt.plot(HEEE_speed, color='r', label='HEEE_speed')
        plt.axvline(color='g', x=idx_stable, label='stable_step')
        plt.axvline(color='m', x=period_data[0], label='start_period')
        plt.axvline(color='y', x=period_data[1], label='end_period')
        plt.legend(loc='best')

        plt.subplot(312, sharex=ax1)
        plt.plot(rule1, color='r', label='rule1')
        plt.plot(rule2, color='g', label='rule2')
        plt.plot(rule3, color='b', label='rule3')
        plt.plot(rule123, color='k', label='rule1 & 2 & 3')
        plt.legend(loc='best')

        plt.subplot(313, sharex=ax1)
        plt.axvline(color='g', x=idx_stable, label='stable_step')
        plt.axvline(color='m', x=period_data[0], label='start_period')
        plt.axvline(color='y', x=period_data[1], label='end_period')
        if idx_heel_strike is not np.NaN:
            plt.axvline(color='k', x=idx_heel_strike, label='idx_heel_strike')
        plt.plot(np.abs(HEE_pos[:, 2] - median_pos_around_step[2]), color='b', label='abs_relative_heel_pos_to_stable_heel_pos')
        plt.axhline(color='r', y=distance_from_stable, label='threshold_distance_from_stable')
        plt.legend(loc='best')

        if plot_it:
            plt.show()

        if save_plot:
            f1.savefig(save_plot_path + 'HEEL DETECTION ' + information + '.png', bbox_inches='tight', dpi=300)
            plt.close('all')

    return idx_heel_strike


# This is a private method which is only used internally to detect toe off.
def TO_detection(information, TOE_pos, TOE_speed, MID_speed, idx_stable, period_data, plot_it=False, save_plot=False, save_plot_path=''):
    steps_stable = 30  # frames pedant lesquelles je calcule la position moyenne du pied
    percent = 0.45  # 0.45 initialement pour TAP
    win_before_heel_strike = 0.7 * 120.
    distance_from_stable = 60.  # mm en vertical

    if len(TOE_pos[idx_stable - steps_stable:idx_stable + steps_stable, :]) == 0:
        median_pos_around_step = [0, 0, 0]
    else:
        median_pos_around_step = np.nanmean(TOE_pos[idx_stable - steps_stable:idx_stable + steps_stable, :], axis=0)  # position moyenne du toe pendant la phase stable

    # rule1 = TOE_pos[:,2]<(np.nanmean(TOE_pos[:,2])*percent) # etre proche du sol en position Z
    # version permettant d'enlever les valeurs negatives au debut
    rule1 = TOE_pos[:, 3] < (np.nanmean(TOE_pos[period_data[0]:period_data[1], 3], dtype='float32') * percent)
    # etre apres le stable step
    rule2 = np.logical_and(np.arange(TOE_pos.shape[0]) > idx_stable, np.arange(TOE_pos.shape[0]) < (idx_stable + int(win_before_heel_strike)))
    # rule2 = np.logical_and(np.arange(TOE_pos.shape[0]) < idx_stable, np.arange(TOE_pos.shape[0]) > (idx_stable - int(win_before_heel_strike)))
    # rule2 = np.logical_and(np.arange(TOE_pos.shape[0]) > (idx_stable - int(win_before_heel_strike)), np.arange(TOE_pos.shape[0]) < (idx_stable + int(win_before_heel_strike)))
    # etre proche en distance du stable step (en Y)
    rule3 = np.abs(TOE_pos[:, 2] - median_pos_around_step[2]) < distance_from_stable

    rule123 = np.logical_and.reduce((rule1, rule2, rule3))

    tmp_MID_speed = MID_speed.copy()
    tmp_MID_speed[np.logical_not(rule123)] = np.NaN

    idx_toe_off = np.NaN if np.all(np.isnan(tmp_MID_speed)) else np.nanargmax(tmp_MID_speed)

    # if ('L_TO_' in information):
        # print(rule3)
        # print(TOE_pos, TOE_speed, MID_speed, idx_stable, period_data)

    # if 'L_Foot_(after_applomb)_Lead_F' in information:
    #     print('rule1: ', rule1)
    #     print('rule2: ', rule2)
    #     print('rule3: ', rule3)
    #     print('np.abs(TOE_pos[:, 2] - median_pos_around_step[2])    ', np.abs(TOE_pos[:, 2] - median_pos_around_step[2]))
    #     print('TOE_pos[:, 2]    ', TOE_pos[:, 2])
    #     print('idx_stable    ', idx_stable)
    #     print('np.arange(TOE_pos.shape[0])    ', np.arange(TOE_pos.shape[0]))

    if plot_it or save_plot:
        f1 = plt.figure(figsize=(16, 12))
        plt.subplot(311)
        plt.title('toe_off_detection_' + information)
        plt.plot(MID_speed, color='k', label='MID_speed')
        plt.plot(tmp_MID_speed, color='b', label='tmp_MID_speed')
        plt.plot(TOE_speed, 'r', label='TOE_speed')
        plt.axvline(color='g', x=idx_stable, label='stable_step')
        plt.axvline(color='m', x=period_data[0], label='start_period')
        plt.axvline(color='y', x=period_data[1], label='end_period')
        plt.legend(loc='best')

        plt.subplot(312)
        plt.plot(rule1, color='r', label='rule1')
        plt.plot(rule2, color='g', label='rule2')
        plt.plot(rule3, color='b', label='rule3')
        plt.plot(rule123, color='k', label='rule1 & 2 & 3')
        plt.legend(loc='best')

        plt.subplot(313)
        # Rule 1
        plt.plot(TOE_pos[:, 3], color='c', label='toe_pos_z')
        plt.axhline(color='limegreen', y=np.nanmean(TOE_pos[:, 3], dtype='float32') * percent, label='threshold_toe_pos_z')
        # rule 3
        plt.axvline(color='g', x=idx_stable, label='stable_step')
        plt.axvline(color='m', x=period_data[0], label='start_period')
        plt.axvline(color='y', x=period_data[1], label='end_period')
        if idx_toe_off is not np.NaN:
            plt.axvline(color='k', x=idx_toe_off, label='idx_toe_off')
        plt.plot(np.abs(TOE_pos[:, 2] - median_pos_around_step[2]), color='b', label='abs_relative_toe_pos_to_stable_toe_pos_y')
        plt.axhline(color='r', y=distance_from_stable, label='threshold_distance_from_stable_y')
        plt.legend(loc='best')

        if plot_it:
            plt.show()

        if save_plot:
            f1.savefig(save_plot_path + 'TOE DETECTION ' + information + '.png', bbox_inches='tight', dpi=300)
            plt.close('all')

    return idx_toe_off


def drawfoot(posheel, postoe, color='r', offset=30.):
    coord = [[posheel[0], posheel[1]], [postoe[0], postoe[1]],
             [(posheel[0] + postoe[0]) / 2., (posheel[1] + postoe[1]) / 2 + offset]]
    coord.append(coord[0])
    xs, ys = zip(*coord)  # create lists of x and y values
    plt.plot(xs, ys, color=color, linewidth=2)


# This is a private method which is only used internally to detect steps with low precision
def steps_detection_low_precision(L_toepos, R_toepos, plotit=False, save_plot=False, save_plot_path=''):
    '''
    T_footSpeed : mm/s
    '''

    ###to modify: is this necessary ???
    # Ajout pour essayer de gerer le bruit
    '''
    L_toepos[:, 0] = filt_my_signal_nan_inside(L_toepos[:, 0], freq=120., lp=7., n=4, plotit=False)
    L_toepos[:, 1] = filt_my_signal_nan_inside(L_toepos[:, 1], freq=120., lp=7., n=4, plotit=False)
    L_toepos[:, 2] = filt_my_signal_nan_inside(L_toepos[:, 2], freq=120., lp=7., n=4, plotit=False)
    R_toepos[:, 0] = filt_my_signal_nan_inside(R_toepos[:, 0], freq=120., lp=7., n=4, plotit=False)
    R_toepos[:, 1] = filt_my_signal_nan_inside(R_toepos[:, 1], freq=120., lp=7., n=4, plotit=False)
    R_toepos[:, 2] = filt_my_signal_nan_inside(R_toepos[:, 2], freq=120., lp=7., n=4, plotit=False)
    '''

    speed_toeL, starts_tmpL, stops_tmpL, stable_stepL = one_side_steps_detection_low_precision(L_toepos)
    speed_toeR, starts_tmpR, stops_tmpR, stable_stepR = one_side_steps_detection_low_precision(R_toepos)

    if plotit or save_plot:
        main_figure = plt.figure(figsize=(16, 12))
        plt.subplot(211)
        plt.xlabel('index_frame')
        plt.ylabel('axis_y')
        plt.plot(L_toepos[:, 2], 'k', label='left_toe_y_position')
        plt.plot(stable_stepL, L_toepos[stable_stepL, 2], 'ko', label='stable_stepL')
        plt.plot(R_toepos[:, 2], 'r', label='right_toe_y_position')
        plt.plot(stable_stepR, R_toepos[stable_stepR, 2], 'ro', label='stable_stepR')
        plt.legend(loc='best')

        plt.subplot(212)
        plt.xlabel('index_frame')
        plt.ylabel('speed_axis_y')
        plt.plot(speed_toeL, 'k', label='left_toe_y_speed')
        plt.plot(starts_tmpL, speed_toeL[starts_tmpL], 'go', label='starts_tmpL')
        plt.plot(stops_tmpL, speed_toeL[stops_tmpL], 'ro', label='stops_tmpL')

        plt.plot(speed_toeR, 'r', label='right_toe_y_speed')
        plt.plot(starts_tmpR, speed_toeR[starts_tmpR], 'gx', label='starts_tmpR')
        plt.plot(stops_tmpR, speed_toeR[stops_tmpR], 'rx', label='stops_tmpR')
        plt.legend(loc='best')

        if plotit:
            plt.show()

        if save_plot:
            main_figure.savefig(save_plot_path + 'steps_detection_low_precision' + '.png', bbox_inches='tight', dpi=300)
            plt.close('all')
    return stable_stepL, speed_toeL, starts_tmpL, stops_tmpL, stable_stepR, speed_toeR, starts_tmpR, stops_tmpR


# This method calculates the list of index of frame where the right foot is stable on the ground using the toe position
def one_side_steps_detection_low_precision(toepos, T_footSpeed=10):
    '''
    T_footSpeed : mm/s
    '''

    '''
    toepos: 3D foot pos
    
    out:
    speed_toe: foot speed (y axis - in walking direction)
    starts_tmp: une estimation des moments ou le pied se pose au sol (liste)
    stops_tmp: une estimation des moments ou le pied se leve du sol (liste)
    stable_step: les moments ou le pied est totalement au sol (mileu - au sens du temps - des deux instants precedents) (liste)

    '''
    speed_toe = np.abs(np.gradient(toepos[:, 2]))
    wh = np.where(speed_toe > T_footSpeed)[0]
    # The moment that the foot starts walking

    starts_tmp = np.hstack([0, wh[np.hstack([np.where(np.diff(wh) > 10)[0], len(wh) - 1])]])
    stops_tmp = np.hstack([wh[np.hstack([0, np.where(np.diff(wh) > 10)[0] + 1])], len(speed_toe) - 1])
    stable_step = (starts_tmp + stops_tmp) / 2

    # plt.figure()
    # plt.plot(speed_toe, 'k')
    # plt.axhline(y=T_footSpeed, color='r')
    # plt.plot(stops_tmp, speed_toe[stops_tmp.astype(np.int64)], 'ok')
    # plt.plot(starts_tmp, speed_toe[starts_tmp.astype(np.int64)], 'og')
    # plt.show()
    # dsf

    return speed_toe, starts_tmp.astype(np.int64), stops_tmp.astype(np.int64), stable_step.astype(np.int64)


# Detect the peak in the signal and delete them
def detect_peak_delete(trial_data, x3D, freq=100, plotit=False, plot_axis='', plot_path='', plot_name=''):
    # Convert the y to speed
    # print(x3D)
    np_positions_120hz = x3D
    vectors_speed_120hz = np.gradient(np_positions_120hz)[0]
    vectors_acc_120hz = np.gradient(vectors_speed_120hz)[0]

    acc_norm = []
    for vector_acc in vectors_acc_120hz:
        # print(LA.norm(vector_acc))
        acc_norm.append(linalg.norm(vector_acc) * freq * freq / 1000)
        # y_speed.to_csv('C:\\Users\\yicha\\Desktop\\QuickText.csv', sep=';')

    # # Plot the acceleration
    # if plotit:
    #     plt.figure(figsize=(16, 9))
    #     plt.title("Norme of acceleration")
    #     plt.plot(acc_norm)
    #     plt.ylim(0, 120)
    #
    #     plt.savefig(plot_path)
    #     plt.close('all')

    # # Height is the limit of human movement acceleration 70m/s2
    # peaks, _ = scipy.signal.find_peaks(acc_norm, height=70)

    # Delete all the peaks with acceleration bigger than 100m/s2
    peaks = []
    for acc in acc_norm:
        if acc >= 150:
            peaks.append(acc_norm.index(acc))

    # Delete the peak and one point before and after
    x3D_result = x3D.copy()
    for peak in peaks:
        for Dim in range(3):
            x3D_result[peak, Dim] = np.nan
            if (peak - 1) > 0 and (peak + 1) < len(x3D_result):
                x3D_result[peak - 1, Dim] = np.nan
                x3D_result[peak + 1, Dim] = np.nan
            if (peak - 2) > 0 and (peak + 2) < len(x3D_result):
                x3D_result[peak - 2, Dim] = np.nan
                x3D_result[peak + 2, Dim] = np.nan
            if (peak - 3) > 0 and (peak + 3) < len(x3D_result):
                x3D_result[peak - 3, Dim] = np.nan
                x3D_result[peak + 3, Dim] = np.nan
            if (peak - 4) > 0 and (peak + 4) < len(x3D_result):
                x3D_result[peak - 4, Dim] = np.nan
                x3D_result[peak + 4, Dim] = np.nan

    # Delete all the isolated points
    list_index = range(len(x3D_result))
    for index in list_index:
        for Dim in range(3):
            if (index - 2) > 0 and (index + 2) < len(list_index):
                if math.isnan(x3D_result[index - 1, Dim]) or math.isnan(x3D_result[index - 2, Dim]):
                    if math.isnan(x3D_result[index + 1, Dim]) or math.isnan(x3D_result[index + 2, Dim]):
                        x3D_result[index, Dim] = np.nan

    # Plot the difference if the plotit is True
    if plotit:
        if plot_axis == 'PosX':
            index_axis = 0
        if plot_axis == 'PosY':
            index_axis = 1
        if plot_axis == 'PosZ':
            index_axis = 2

        plt.figure(figsize=(16, 9))
        plt.suptitle(plot_name)
        plt.subplot(211).set_title("Before delete peaks of acceleration")
        plt.plot(x3D[:, index_axis])
        # print(x[peaks])
        plt.plot(peaks, x3D[:, index_axis][peaks], "x")
        plt.subplot(212).set_title("After delete peaks of acceleration")
        plt.plot(x3D_result[:, index_axis])

        plt.savefig(plot_path)
        plt.close('all')

    if plotit:
        trial_data.num_peak_deleted += len(peaks)

    return x3D_result


# Fill the gap with spline for tracker and rigidbody position data
def fill_gap_trash(x3Dini, kind='cubic', plotit=False, plot_axis='', plot_path='', plot_name=''):
    '''
    Faire un bouchage de nan sur des donnees 3D (uniquement spline maintenant)
    avec une copie
    '''
    #    import scipy as sc
    x3D = x3Dini.copy()

    # print(x3D[0, :])

    # si tous les valeurs sont nan, return
    if np.count_nonzero(np.isnan(x3D[:, 0])) == len(x3D[:, 0]):
        return x3D

    # enlever les nans au tout debut
    if np.isnan(x3D[0, 0]):
        nans, x = nan_helper(x3D[:, 0])
        nans.nonzero()[0]
        iiii = 0
        while np.isnan(x3D[iiii, 0]):
            valuecorrect = x3D[iiii + 1, :]
            iiii += 1
        iiii = 0
        while np.isnan(x3D[iiii, 0]):
            x3D[iiii, :] = valuecorrect
            iiii += 1

    # Meme chose pour les nans a la fin
    if np.isnan(x3D[-1, 0]):
        iiii = len(x3D[:, 0]) - 1
        while np.isnan(x3D[iiii, 0]):
            valuecorrect = x3D[iiii - 1, :]
            iiii -= 1
        iiii = len(x3D[:, 0]) - 1
        while np.isnan(x3D[iiii, 0]):
            x3D[iiii, :] = valuecorrect
            iiii -= 1

    # Il peut aussi y avoir des nan plus tard qu'il faudrait interpoler
    nans, trash = nan_helper(x3D[:, 0])
    if nans.nonzero()[0].__len__() > 0:
        for Dim in range(3):
            nans, trash = nan_helper(x3D[:, Dim])
            xnew = np.array(range(len(x3D[:, Dim])))
            f = scipy.interpolate.InterpolatedUnivariateSpline(xnew[~nans], x3D[~nans, Dim], k=3)
            x3D[nans, Dim] = f(xnew[nans])

    # Plot the difference if the plotit is True
    if plotit:
        if plot_axis == 'PosX':
            index_axis = 0
        if plot_axis == 'PosY':
            index_axis = 1
        if plot_axis == 'PosZ':
            index_axis = 2

        plt.figure(figsize=(16, 9))
        plt.suptitle(plot_name)
        plt.subplot(211).set_title("Before fill gap trash")
        plt.plot(x3Dini[:, index_axis])
        # print(x[peaks])
        plt.subplot(212).set_title("After fill gap trash")
        plt.plot(x3D[:, index_axis])

        plt.savefig(plot_path)
        plt.close('all')

    return x3D


# Filter the data with butterworth filter for tracker and rigidbody position data
def filt_my_signal(y, freq=100., lp=5., n=4, plotit=False, plot_axis='', plot_path='', plot_name=''):
    b, a = butter(n, (lp / (freq / 2.)), btype='low')  # Butterworth filter
    yfilts = filtfilt(b, a, y)  # filter with phase shift correction

    # if plotit:
    #     t = range(0, len(y))
    #     fig, ax1 = plt.subplots(1, 1, figsize=(10, 4))
    #     plt.title = plot_name
    #     ax1.plot(t, y, 'r.-', linewidth=2, label='raw data')
    #     ax1.plot(t, yfilts, 'g.-', linewidth=2, label='filtfilt @ ' + str(lp) + ' Hz')
    #     # ax1.legend(frameon=False, fontsize=14)
    #     ax1.legend()
    #     ax1.set_xlabel("Time [s]");
    #     ax1.set_ylabel("Amplitude")
    #     plt.show()

    # Plot the difference if the plotit is True
    if plotit:
        if plot_axis == 'PosX':
            index_axis = 0
        if plot_axis == 'PosY':
            index_axis = 1
        if plot_axis == 'PosZ':
            index_axis = 2

        plt.figure(figsize=(16, 9))
        plt.suptitle(plot_name)
        plt.subplot(211).set_title("Before filt my signal")
        plt.plot(y)
        # print(x[peaks])
        plt.subplot(212).set_title("After filt my signal")
        plt.plot(yfilts)

        plt.savefig(plot_path)
        plt.close('all')

    return yfilts


def normalize(v):
    return v / np.linalg.norm(v)


def find_additional_vertical_vector(vector):
    ez = np.array([0, 0, 1])
    look_at_vector = normalize(vector)
    up_vector = normalize(ez - np.dot(look_at_vector, ez) * look_at_vector)
    return up_vector


def calc_rotation_matrix(v1_start, v2_start, v1_target, v2_target):
    """
    calculating M the rotation matrix from base U to base V
    M @ U = V
    M = V @ U^-1
    """

    def get_base_matrices():
        u1_start = normalize(v1_start)
        u2_start = normalize(v2_start)
        u3_start = normalize(np.cross(u1_start, u2_start))

        u1_target = normalize(v1_target)
        u2_target = normalize(v2_target)
        u3_target = normalize(np.cross(u1_target, u2_target))

        U = np.hstack([u1_start.reshape(3, 1), u2_start.reshape(3, 1), u3_start.reshape(3, 1)])
        V = np.hstack([u1_target.reshape(3, 1), u2_target.reshape(3, 1), u3_target.reshape(3, 1)])

        return U, V

    def calc_base_transition_matrix():
        return np.dot(V, np.linalg.inv(U))

    if not np.isclose(np.dot(v1_target, v2_target), 0, atol=1e-03):
        raise ValueError("v1_target and v2_target must be vertical")

    U, V = get_base_matrices()
    return calc_base_transition_matrix()


def get_euler_rotation_angles(start_look_at_vector, target_look_at_vector, start_up_vector=None, target_up_vector=None):
    if start_up_vector is None:
        start_up_vector = find_additional_vertical_vector(start_look_at_vector)

    if target_up_vector is None:
        target_up_vector = find_additional_vertical_vector(target_look_at_vector)

    rot_mat = calc_rotation_matrix(start_look_at_vector, start_up_vector, target_look_at_vector, target_up_vector)
    is_equal = np.allclose(rot_mat @ start_look_at_vector, target_look_at_vector, atol=1e-03)
    # print(f"rot_mat @ start_look_at_vector1 == target_look_at_vector1 is {is_equal}")
    rotation = Rotation.from_matrix(rot_mat)
    return rotation.as_euler(seq="xyz", degrees=True)


# Convert the 2 directional vector to a euler rotation vector 3
def from_to_rotation(start_look_at_vector, target_look_at_vector):
    phi, theta, psi = get_euler_rotation_angles(start_look_at_vector, target_look_at_vector)
    # print(f"phi_x_rotation={phi}, theta_y_rotation={theta}, psi_z_rotation={psi}")
    euler_rotation = [phi, theta, psi]
    return euler_rotation


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


# Get the quaternion in (w, x, y, z) of an object at a specific frame in a dataframe
def get_quaternion(df, object_name, frame):
    w = df.loc[frame, (object_name, ['QuaW'])].values[0]
    x = df.loc[frame, (object_name, ['QuaX'])].values[0]
    y = df.loc[frame, (object_name, ['QuaY'])].values[0]
    z = df.loc[frame, (object_name, ['QuaZ'])].values[0]
    return [w, x, y, z]


# Convert the quaternion to pitch (in degree) for vicon coordinate system
# The vicon coordinate system is right handed, the axes order is 'syxz'
# The radian should be converted to angle
# euler_degree[0] represents roll. When positive, tilt the head to the right; when negative, tilt the head to the left (according to coordinate system)
# euler_degree[1] represents pitch. When positive, look up; when negative, look down
# euler_degree[2] represents yaw. When positive, turn left; when negative, turn right (according to coordinate system)
def convert_quaternion_to_pitch(quaternion):
    euler_radian = list(quat2euler(quaternion, axes='syxz'))
    euler_degree = [math.degrees(euler_radian[i]) for i in range(3)]
    return euler_degree[1]


def convert_quaternion_to_roll_pitch_yaw(quaternion):
    euler_radian = list(quat2euler(quaternion, axes='syxz'))
    euler_degree = [math.degrees(euler_radian[i]) for i in range(3)]
    return euler_degree
