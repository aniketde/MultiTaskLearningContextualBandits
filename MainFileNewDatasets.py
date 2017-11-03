from __future__ import division
import numpy as np
import pickle
import DataCreate.DataCreate as DC
import CrossValidation.CrossValidation as CV
import Algorithms.UCBTimeT as UCBT
import os
import warnings
import bandwidthselection.bandwidths as bwest

warnings.filterwarnings("ignore")
import time

start_time = time.time()

########################################################################################################################
# This is a main file.
#
#
#
#
########################################################################################################################


### Initialize the algorithm
########################################################################################################################
A = 5 #No of arms for synthetic data (data_flag = 3) Redundant for other flag
d = 2 #Dimension for synthetic data. Currently results shown are with d = 2
N_valid = 50 #No. of data points per arm in validation set
N = 1 #Algorithm starts with one random example assigned to each arm. This is a cold start problem.
alpha = 0.1

data_flag = 3 # 3 is syntehtic TL dataset, 7 real multiclass datasets (please see data_flag_multiclass)
if data_flag == 3:
    data_flag_multiclass = "synthetic1" #Results will be stored in this folder for synthetic data
elif data_flag == 7:
    data_flag_multiclass = 'segment' # 'Digits',  'mnsit', 'pendigits', 'segment','letter',  'usps',

crossval_flag = 0  # 0 if you don't need cross validation, 2 for batch learning way of doing cross valid, 3 for silverman's rule
Main_Program_flag = 0  # This variable is 0 when doing cross validation and 1 when running main block. Don't change this.
# In future release we will get rid of this.

#Random seeds so that we use same sequence of data points while evaluating algorithms
randomSeedsTrain = np.array([15485867, 15486277, 15486727, 15487039,
                             15485917, 15486281, 15486739, 15487049,
                             15485927, 15486283, 15486749, 15487061,
                             15485933, 15486287, 15486769, 15487067,
                             15485941, 15486347, 15486773, 15487097,
                             15485959, 15486421, 15486781, 15487103,
                             15485989, 15486433, 15486791, 15487139,
                             15485993, 15486437, 15486803, 15487151,
                             15486013, 15486451, 15486827, 15487177,
                             15486041, 15486469, 15486833, 15487237,
                             15486047, 15486481, 15486857, 15487243,
                             15486059, 15486487, 15486869, 15487249,
                             15486071, 15486491, 15486871, 15487253])

randomSeedsTest = np.array([15486101, 15486511, 15486883, 15487271,
                            15486139, 15486517, 15486893, 15487291,
                            15486157, 15486533, 15486907, 15487309,
                            15486173, 15486557, 15486917, 15487313,
                            15486181, 15486571, 15486929, 15487319,
                            15486193, 15486589, 15486931, 15487331,
                            15486209, 15486649, 15486953, 15487361,
                            15486221, 15486671, 15486967, 15487399,
                            15486227, 15486673, 15486997, 15487403,
                            15486241, 15486703, 15487001, 15487429,
                            15486257, 15486707, 15487007, 15487457,
                            15486259, 15486719, 15487019, 15487469])

########################################################################################################################


### List of algorithms
########################################################################################################################
# These are different algorithm you could run and compare. You can add your own aglorithm in the list by modifying DataCreate
# and GaussianKernels files in the respective libraries.
algorithm_list = ['KTL-UCB-TaskSimEst', 'Lin-UCB-Ind', 'KTL-UCB-TaskSim', 'Lin-UCB-Pool']
#algorithm_list = ['KTL-UCB-TaskSimEst', 'Lin-UCB-Ind']
#algorithm_list = ['KTL-UCB-TaskSim', 'Lin-UCB-Ind']
########################################################################################################################


### cross validation Block
########################################################################################################################

Parameter_Dict = dict()

# Use selected hyperparamertes without doing cross validation
if crossval_flag == 0:

    if data_flag == 3:
        Parameter_Dict['KTL-UCB-TaskSim'] = np.array([2.1544346900318843, 0.001, 27.825594022071257, 0.001, alpha])
        Parameter_Dict['KTL-UCB-TaskSimEst'] = np.array([2.1544346900318843, 2.1544346900318843, 100.0, 0.001, alpha])
        Parameter_Dict['Lin-UCB-Ind'] = np.array([2.1544346900318843, 0.001, 0.001, 0.001, alpha])
        Parameter_Dict['Lin-UCB-Pool'] = np.array([2.1544346900318843, 0.001, 0.001, 0.001, alpha])

    elif (data_flag == 4):
        #Data_flag = 4
        Parameter_Dict['KTL-UCB-TaskSim'] = np.array([0.16681005372, 0.001, 0.1, 0.0464158883361, alpha])
        Parameter_Dict['KTL-UCB-TaskSimEst'] = np.array([0.16681005372, 0.001, 0.1, 0.0464158883361, alpha])
        Parameter_Dict['Lin-UCB-Ind'] = np.array([0.0464158883361, 0.001, 0.001, 0.16681005372, alpha])
        Parameter_Dict['Lin-UCB-Pool'] = np.array([0.001, 0.001, 0.001, 0.16681005372, alpha])
    elif (data_flag == 7):

        if data_flag_multiclass == "Digits":
            Parameter_Dict['KTL-UCB-TaskSimEst'] = np.array([0.001, 0.01291549665014884, 100.0, 0.001, alpha])
            Parameter_Dict['Lin-UCB-Ind'] = np.array([0.001, 0.001, 0.001, 0.001, alpha])

        elif data_flag_multiclass == "mnsit":
            Parameter_Dict['KTL-UCB-TaskSimEst'] = np.array([0.0035938136638046258, 7.7426368268112773, 100.0, 0.001,alpha])
            Parameter_Dict['Lin-UCB-Ind'] = np.array([0.0035938136638046258, 0.001, 0.001, 0.001, alpha])

        elif data_flag_multiclass == "pendigits":
            Parameter_Dict['KTL-UCB-TaskSimEst'] = np.array([0.001, 0.001, 100.0, 0.001, alpha])
            Parameter_Dict['Lin-UCB-Ind'] = np.array([0.001, 0.001, 0.001, 0.001, alpha])

        elif data_flag_multiclass == "segment":
            Parameter_Dict['KTL-UCB-TaskSimEst'] = np.array(
                [0.59948425031894093, 100.0, 27.825594022071257, 0.0035938136638046258, alpha])
            Parameter_Dict['Lin-UCB-Ind'] = np.array([0.1668100537200059, 0.001, 0.001, 0.0035938136638046258,alpha])

        elif data_flag_multiclass == "usps":
            Parameter_Dict['KTL-UCB-TaskSimEst'] = np.array(
                [0.01291549665014884, 0.1668100537200059, 100.0, 0.001, alpha])
            Parameter_Dict['Lin-UCB-Ind'] = np.array([0.01291549665014884, 0.001, 0.001, 0.001, alpha])

        elif data_flag_multiclass == "letter":
            Parameter_Dict['KTL-UCB-TaskSimEst'] = np.array([2.1544346900318843, 27.825594022071257, 100.0, 0.001, alpha])
            Parameter_Dict['Lin-UCB-Ind'] = np.array([2.1544346900318843, 0.001, 0.001, 0.001, alpha])

# Select the hyperparamertes using five fold cross validation
elif crossval_flag == 2:
    if data_flag == 3:
        N = N_valid
    T = 200
    fold_cv = 5
    n_grid = 10
    nStart = -3
    nEnd = 2
    bw_x_grid = np.logspace(nStart, nEnd, n_grid)
    gamma_grid = np.logspace(nStart, nEnd, n_grid)

    for algorithm_flag in algorithm_list:
        if (algorithm_flag == 'KTL-UCB-TaskSimEst' or algorithm_flag == 'KTL-UCB-TaskSimEstRew'):
            bw_prod_grid = np.logspace(nStart, nEnd, n_grid)
            bw_prob_grid = np.logspace(nStart, nEnd, n_grid)
        elif (algorithm_flag == 'KTL-UCB-TaskSim'):
            bw_prod_grid = np.logspace(nStart, nEnd, n_grid)
            bw_prob_grid = np.logspace(nStart, nEnd, 1)
        else:
            bw_prod_grid = np.logspace(nStart, nEnd, 1)
            bw_prob_grid = np.logspace(nStart, nEnd, 1)
        bw_x_est, bw_prob_est, bw_prod_est, lambda_reg_est = CV.CrossValidRegression(bw_x_grid, gamma_grid,
                                                                                     bw_prod_grid, bw_prob_grid,
                                                                                     fold_cv, algorithm_flag, data_flag,
                                                                                     data_flag_multiclass, A,
                                                                                     d, N_valid, N, T, randomSeedsTrain,
                                                                                     Main_Program_flag)

        Parameter_Dict[algorithm_flag] = bw_x_est, bw_prob_est, bw_prod_est, lambda_reg_est, alpha

#Use silverman's rule to select bandwidth
#Code taken from https://github.com/statsmodels/statsmodels/blob/master/statsmodels/nonparametric/bandwidths.py
elif crossval_flag == 3:
    T = 150
    # rngTrain = np.random.RandomState(randomSeedsTrain[0])
    DataXY = DC.TrainDataCollect(data_flag, data_flag_multiclass, A, d, N_valid, N, T, randomSeedsTrain[0], Main_Program_flag)
    Testfeatures = np.copy(DataXY['Testfeatures'])
    bw_x = bwest.bw_silverman(Testfeatures[:, 0], kernel=None)
    bw_prob = bwest.bw_silverman(Testfeatures[:, 0], kernel=None)
    bw_prod = bwest.bw_silverman(Testfeatures[:, 0], kernel=None)
    gamma = 0.15

    for algorithm_flag in algorithm_list:
        Parameter_Dict[algorithm_flag] = bw_x, bw_prob, bw_prod, gamma, alpha

print Parameter_Dict

# pickle.dump(Parameter_Dict, open(str(data_flag) + 'parameter_results.p', "wb"))
CV_time = time.time() - start_time
print("Cross Validation took %s seconds ---" % (CV_time))

results_Folder = 'data_flag_' + str(data_flag)

# Save hyperparameters
Parameter_Dict_File_Name = 'cv_' + str(crossval_flag) + '_parameter_results.p'
Parameter_Dict_File_Path = os.path.join("ExperimentResults", results_Folder)
Parameter_Dict_File_Path = os.path.join(Parameter_Dict_File_Path, data_flag_multiclass)
Parameter_Dict_File_Path = os.path.join(Parameter_Dict_File_Path, Parameter_Dict_File_Name)
pickle.dump(Parameter_Dict, open(Parameter_Dict_File_Path, "wb"))

########################################################################################################################


### Main Algorithm Block
########################################################################################################################

start_time = time.time()
T = 1000
if data_flag == 7:
    if data_flag_multiclass == 'Digits':
        T = 1250
    elif data_flag_multiclass == 'mnsit':
        T = 4000  # We can go upto 53750
    elif data_flag_multiclass == 'letter':
        T = 4000  # We can go upto 12740
    elif data_flag_multiclass == 'segment':
        T = 1000  # We can go upto 1953
    elif data_flag_multiclass == 'pendigits':
        T = 4000  # We can go upto 6984, A is 10
    elif data_flag_multiclass == 'usps':
        T = 4000 # We can go upto 6781, A is 10
elif data_flag == 3:
    N = 1

Main_Program_flag = 1 # This variable is 0 when doing cross validation and 1 when running main block. Don't change this.
# In future release we will get rid of this.
Runs = 10 #Number of times we repeat the experiments.
# Results_dict = dict()

#Initialization
AverageRegretRuns = np.zeros([len(algorithm_list), Runs])
AverageAccuracyRuns = np.zeros([len(algorithm_list), Runs])

for RunNumber in range(0,10):
    algo = 0
    regretUCBRuns = np.zeros([len(algorithm_list), T])
    for algorithm_flag in algorithm_list:
        #random seed
        rngTest = np.random.RandomState(randomSeedsTest[RunNumber])
        #Get the train data. This is just one example assigned to each arm randomly when N = 1 (cold start)
        DataXY = DC.TrainDataCollect(data_flag, data_flag_multiclass, A, d, N_valid, N, T, randomSeedsTest[RunNumber], Main_Program_flag)
        #Get the parameters
        bw_x, bw_prob, bw_prod, gamma, alpha = Parameter_Dict[algorithm_flag]
        print "Algorithm " + "Run number" + str(RunNumber)
        print algorithm_flag, RunNumber, bw_x, bw_prob, bw_prod, gamma, alpha
        #Run the bandit algorithm and get regret/reward with selected arm
        AverageRegret, AverageAccuracy, regretUCB, Selected_Arm_T, Exact_Arm_T, Task_sim_dict = UCBT.ContextBanditUCBRunForTSteps(
            DataXY, T, data_flag, bw_x, bw_prob, bw_prod, gamma, alpha, algorithm_flag)
        #Store the result
        Results_dict = {'AverageRegret': AverageRegret, 'AverageAccuracy': AverageAccuracy, 'regretUCB': regretUCB,
                        'Selected_Arm_T': Selected_Arm_T, 'Exact_Arm_T': Exact_Arm_T, 'Task_sim_dict': Task_sim_dict}
        all_Results_File_Name = 'dataset_' + data_flag_multiclass + '_Run_' + str(
            RunNumber) + '_algorithm_' + algorithm_flag + '_Results_Dict.p'

        all_Results_File_Path = os.path.join("ExperimentResults", results_Folder)
        all_Results_File_Path = os.path.join(all_Results_File_Path, data_flag_multiclass)
        all_Results_File_Path = os.path.join(all_Results_File_Path, all_Results_File_Name)

        pickle.dump(Results_dict, open(all_Results_File_Path, "wb"))
        AverageRegretRuns[algo, RunNumber] = AverageRegret
        AverageAccuracyRuns[algo, RunNumber] = AverageAccuracy
        regretUCBRuns[algo, :] = regretUCB
        print AverageAccuracyRuns
        # print "A = " + str(A)
        print "T = " + str(T)
        print "N = " + str(N)
        print "d = " + str(d)

        #Move to next algorithm
        algo += 1

    save_Folder = os.path.join("ExperimentResults", results_Folder)
    save_Folder = os.path.join(save_Folder, data_flag_multiclass)
    if not os.path.exists(save_Folder):
        os.makedirs(save_Folder)
    saveLocation = os.path.join(save_Folder, str(RunNumber) + ".csv")
    resultsFile = open(saveLocation, 'w')
    for algoCounter in range(0, len(algorithm_list)):
        regrets = regretUCBRuns[algoCounter, :]
        regrets = np.cumsum(regrets).tolist()
        regrets_str = ",".join(map(str, regrets))
        resultsFile.write(regrets_str)
        resultsFile.write("\n")
    resultsFile.close()

CV_Time_File_Name = 'cv_' + str(crossval_flag) + '_cross_val_time_required.p'
CV_Time_File_Name_Path = os.path.join("ExperimentResults", results_Folder)
CV_Time_File_Name_Path = os.path.join(CV_Time_File_Name_Path, data_flag_multiclass)
CV_Time_File_Name_Path = os.path.join(CV_Time_File_Name_Path, CV_Time_File_Name)
pickle.dump(CV_time, open(CV_Time_File_Name_Path, "wb"))

print AverageAccuracyRuns

Main_Block_Algo_time = time.time() - start_time

Training_Time_File_Name = 'dataset_' + data_flag_multiclass + '_Main_Block_time_required.p'

Training_Time_File_Path = os.path.join("ExperimentResults", results_Folder)
Training_Time_File_Path = os.path.join(Training_Time_File_Path, data_flag_multiclass)
Training_Time_File_Path = os.path.join(Training_Time_File_Path, Training_Time_File_Name)

pickle.dump(Main_Block_Algo_time, open(Training_Time_File_Path, "wb"))

print("Main Block of algorithm took %s seconds ---" % (Main_Block_Algo_time))