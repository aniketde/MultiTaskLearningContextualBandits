


from __future__ import division
import numpy as np
import DataCreate.DataCreate as DC
import BanditModels.Unified_Bandit as Bandits

# Thic function runs the bandit algorithm for T steps
def ContextBanditUCBRunForTSteps(DataXY,T,data_flag,bw_x,bw_prob,bw_prod,gamma,alpha,algorithm_flag):
    accuracy_UCB = 0.0
    #rewardAccu = np.zeros([T])
    regretUCB = np.zeros([T])
    Bandit_Algo = Bandits.Unified_Bandit(bw_x, bw_prod, gamma, bw_prob, alpha, data_flag,
                                 algorithm_flag)

    if (data_flag == 3):
        Bandit_Algo.set_All_User_Context(DataXY['UserContext'])
        Bandit_Algo.set_All_Arm_Context(DataXY['ArmContext'])
        Bandit_Algo.set_Reward_Distribution(DataXY['rewardDist'])

    Selected_Arm_T = np.zeros([T])
    Exact_Arm_T =  np.zeros([T])
    Task_sim_dict = dict()
    for tt in range(0, T):
        arm_tt, X_test, DataXY = Bandit_Algo.get_Arm_And_X_test_And_Data(DataXY, tt)
        if data_flag == 3:
            Bandit_Algo.update_Collected_Rewards(DataXY, tt, [])
            RewardforAllArms = Bandit_Algo.get_Reward_For_All_Arms(DataXY, tt)

            ind_arm = np.argmax(RewardforAllArms)
            true_reward =  RewardforAllArms[ind_arm]

            rewardAccu = Bandit_Algo.get_Collected_Rewards()

        elif (data_flag == 7):
            armTest = DataXY['armTest']
            ind_arm = armTest[tt]
            true_reward = 1.0

            Bandit_Algo.update_Collected_Rewards(DataXY, tt, ind_arm)
            rewardAccu = Bandit_Algo.get_Collected_Rewards()

        Selected_Arm_T[tt] = arm_tt
        Exact_Arm_T[tt] = ind_arm

        if int(ind_arm) == int(arm_tt):
            accuracy_UCB += 1

        if data_flag == 3:
            regretUCB[tt] = RewardforAllArms[ind_arm] - rewardAccu[tt]
        elif (data_flag == 7):
            regretUCB[tt] = 1.0 - rewardAccu[tt]

        #Add Data
        DataXY = DC.AddData(DataXY, arm_tt, algorithm_flag, X_test, rewardAccu[tt], tt)

        if tt % 50 == 0 and tt != 0:
            print "iteration number, true class, UCB class,true reward, UCB reward, Algorithm "
            print tt, int(ind_arm), int(arm_tt), true_reward, rewardAccu[tt], algorithm_flag
            print str(tt) + " Accuracy of "+ algorithm_flag + " :"+ str( accuracy_UCB / tt)

        if tt == T:
            print "iteration number, true class, UCB class,true reward, UCB reward, Algorithm "
            print tt, int(ind_arm), int(arm_tt), true_reward, rewardAccu[tt], algorithm_flag
            print str(tt) + " Accuracy of "+ algorithm_flag + " :"+ str( accuracy_UCB / tt)
        if tt %(T/4) == 0:
            Task_sim_dict[algorithm_flag + '_Task_Sim_' + str(tt)] = DataXY[algorithm_flag + '_TaskSim']
    AverageRegret = np.sum(regretUCB) / float(T)
    AverageAccuracy = accuracy_UCB / float(T)

    return AverageRegret,AverageAccuracy,regretUCB,Selected_Arm_T,Exact_Arm_T,Task_sim_dict
