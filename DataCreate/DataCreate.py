

from __future__ import division
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
import sklearn.datasets as datasets
from sklearn.cross_validation import train_test_split


def reward_from_labels_Multiclass(true_label, est_label):

    if true_label == est_label:
        rew = 1
    else:
        rew = 0
    rew = np.ones([1]) * rew
    # print true_label, est_label, rew
    return rew


def TrainDataCollect(data_flag,data_flag_multiclass,A,d,N_valid,N,T,RandomSeedNumber,Main_Program_flag):
    rng = np.random.RandomState(RandomSeedNumber)
    if data_flag == 3:
        theta_select = np.arange(0, A)
        theta = np.zeros([A, 1])
        rewardDist_select = np.arange(0, A)
        rewardDist = np.zeros([A, 1])
        x = rng.rand(100000, 2)
        x = x * 2 - 1
        z = np.sum(x ** 2, axis=1)
        x = x[z <= 1]
        X = x[0:(N + T), :]
        X[:, 1] = X[:, 1] * 0.5
        Ydummy = np.copy(X[:, 1])
        XTrain = X[0:N, :]
        XTest = X[N:, :]
        UserContext = np.copy(XTest)
        # YTest = Y[N:]
        DataXY = dict()
        for aa in range(0, A):
            theta[aa] = theta_select[aa] * (np.pi / 25.0)
            # theta[0] = 0.0
            R = np.array([[np.cos(theta[aa, 0]), -np.sin(theta[aa, 0])], [np.sin(theta[aa, 0]), np.cos(theta[aa, 0])]])
            rewardDist[aa] = rewardDist_select[aa] / A
            print theta[aa], rewardDist[aa]
            Y = (1.0 - (Ydummy - rewardDist[aa] + 0.5) ** 2)  # + 0.01*np.random.randn(N+T)
            XTrain = R.dot(XTrain.T).T
            YTrain = Y[0:N]

            Total_Features = np.copy(XTrain)
            Arm_rewards = np.copy(YTrain)

            # Save training data for KTL UCB
            train_datasetKTLUCB = 'Train_Datasets_KTLUCB' + str(int(aa))
            DataXY[train_datasetKTLUCB] = np.copy(Total_Features)

            train_labelsKTLUCB = 'Train_Labels_KTLUCB' + str(int(aa))
            DataXY[train_labelsKTLUCB] = np.copy(Arm_rewards)

            # Save training data for KTLEst UCB
            train_datasetKTLEstUCB = 'Train_Datasets_KTLEstUCB' + str(int(aa))
            DataXY[train_datasetKTLEstUCB] = np.copy(Total_Features)

            train_labelsKTLEstUCB = 'Train_Labels_KTLEstUCB' + str(int(aa))
            DataXY[train_labelsKTLEstUCB] = np.copy(Arm_rewards)


            # Save training data for Lin UCB
            train_datasetLinUCB = 'Train_Datasets_LinUCB' + str(int(aa))
            DataXY[train_datasetLinUCB] = np.copy(Total_Features)

            train_labelsLinUCB = 'Train_Labels_LinUCB' + str(int(aa))
            DataXY[train_labelsLinUCB] = np.copy(Arm_rewards)

            # Save training data for Pool UCB
            train_datasetPoolUCB = 'Train_Datasets_PoolUCB' + str(int(aa))
            DataXY[train_datasetPoolUCB] = np.copy(Total_Features)

            train_labelsPoolUCB = 'Train_Labels_PoolUCB' + str(int(aa))
            DataXY[train_labelsPoolUCB] = np.copy(Arm_rewards)
        ArmContext = np.copy(theta)  # don't need it but just to pass through the functions
        DataXY['theta'] = np.copy(theta)
        DataXY['ArmContext'] = np.copy(ArmContext)
        DataXY['UserContext'] = np.copy(UserContext)
        DataXY['rewardDist'] = np.copy(rewardDist)

    elif data_flag == 7:
        if data_flag_multiclass == 'Digits':
            Z = datasets.load_digits()
            Features = Z['data']
            Labels = Z['target']
        elif data_flag_multiclass == 'mnsit':
            mnsit = datasets.load_svmlight_file('./mnist.scale.bz2')
            Features = mnsit[0]
            Features = Features.todense()
            Labels = mnsit[1]
        elif data_flag_multiclass == 'letter':
            letter = datasets.load_svmlight_file('./letter.scale.txt')
            Features = letter[0]
            Features = Features.todense()
            Labels = letter[1]
        elif data_flag_multiclass == 'segment':
            segment = datasets.load_svmlight_file('./segment.scale.txt')
            Features = segment[0]
            Features = Features.todense()
            Labels = segment[1]
        elif data_flag_multiclass == 'pendigits':
            pendigits = datasets.load_svmlight_file('./pendigits.txt')
            Features = pendigits[0]
            Features = Features.todense()
            Labels = pendigits[1]
        elif data_flag_multiclass == 'vehicle':
            vehicle = datasets.load_svmlight_file('./vehicle.scale.txt')
            Features = vehicle[0]
            Features = Features.todense()
            Labels = vehicle[1]
        elif data_flag_multiclass == 'usps':
            usps = datasets.load_svmlight_file('./usps.bz2')
            Features = usps[0]
            Features = Features.todense()
            Labels = usps[1]

        if np.min(Labels) == 1:
            Labels = Labels - 1
        elif np.min(Labels) == -1:
            Labels = Labels + 1
        A = np.unique(Labels).shape[0]  # number of classes
        Features_valid, Features_train_test, Labels_valid, Labels_train_test = train_test_split(Features, Labels, train_size= int(A*N_valid), random_state = 3,stratify = Labels)
        if Main_Program_flag == 0:
            Features_train = np.copy(Features_valid)
            Features_test = np.copy(Features_valid)
            Labels_train = np.copy(Labels_valid)
            Labels_test = np.copy(Labels_valid)
        else:
            Features_train, Features_test, Labels_train, Labels_test = train_test_split(Features_train_test, Labels_train_test,
                                                                                                    train_size=int(
                                                                                                        A * N),
                                                                                                    random_state=RandomSeedNumber+17,
                                                                                                    stratify=Labels_train_test)
            # Features_train.shape, Features_test.shape, A*N, Labels_train
        if Main_Program_flag == 0:
            M = N_valid
        else:
            M = N

        idx = rng.permutation(A * M)
        Features_train = Features_train[idx, :]
        Labels_train = Labels_train[idx]


        Labels_test_hist = np.histogram(Labels_test,A)
        minimum_number_label = np.min(Labels_test_hist[0])
        Features_test_dummy = np.zeros([A*minimum_number_label,Features_test.shape[1]])
        Labels_test_dummy = np.zeros([A * minimum_number_label])

        for aa in range(0,A):
            idx = np.where(Labels_test == aa)[0]
            #idx = np.asarray(idx)
            #idx = idx[0,:]
            #idx.astype(int)
            Features_test_dummy[aa*minimum_number_label:(aa+1)*minimum_number_label,:] = Features_test[idx[:minimum_number_label],:]
            Labels_test_dummy[aa * minimum_number_label:(aa + 1) * minimum_number_label] = Labels_test[idx[:minimum_number_label]]
        idx = rng.permutation(A * minimum_number_label)
        Features_test = Features_test_dummy[idx, :]
        Labels_test = Labels_test_dummy[idx]


        print A * minimum_number_label
        DataXY = dict()
        for aa in range(0, A):
            XTrain = Features_train[aa*M:(aa+1)*M,:]
            LabelsTrain = Labels_train[aa * M:(aa + 1) * M]
            YTrain = np.zeros([M])
            YTrain[ LabelsTrain == aa] = 1
            print LabelsTrain, aa, YTrain
            Total_Features = np.copy(XTrain)
            Arm_rewards = np.copy(YTrain)
            # Save training data for KTL UCB
            train_datasetKTLUCB = 'Train_Datasets_KTLUCB' + str(int(aa))
            DataXY[train_datasetKTLUCB] = np.copy(Total_Features)

            train_labelsKTLUCB = 'Train_Labels_KTLUCB' + str(int(aa))
            DataXY[train_labelsKTLUCB] = np.copy(Arm_rewards)

            # Save training data for KTLEst UCB
            train_datasetKTLEstUCB = 'Train_Datasets_KTLEstUCB' + str(int(aa))
            DataXY[train_datasetKTLEstUCB] = np.copy(Total_Features)

            train_labelsKTLEstUCB = 'Train_Labels_KTLEstUCB' + str(int(aa))
            DataXY[train_labelsKTLEstUCB] = np.copy(Arm_rewards)


            # Save training data for Lin UCB
            train_datasetLinUCB = 'Train_Datasets_LinUCB' + str(int(aa))
            DataXY[train_datasetLinUCB] = np.copy(Total_Features)

            train_labelsLinUCB = 'Train_Labels_LinUCB' + str(int(aa))
            DataXY[train_labelsLinUCB] = np.copy(Arm_rewards)

            # Save training data for Pool UCB
            train_datasetPoolUCB = 'Train_Datasets_PoolUCB' + str(int(aa))
            DataXY[train_datasetPoolUCB] = np.copy(Total_Features)

            train_labelsPoolUCB = 'Train_Labels_PoolUCB' + str(int(aa))
            DataXY[train_labelsPoolUCB] = np.copy(Arm_rewards)

        DataXY['Testfeatures'] = np.copy(Features_test)
        DataXY['theta'] = 0
        DataXY['armTest'] = np.copy(Labels_test)
        print Labels_test.shape
    DataXY['NoOfArms'] = A
    return DataXY



def AllDataCollect(DataXY,algorithm_flag):
    # Get total samples and samples in each dataset
    A = DataXY['NoOfArms']
    total_samples = 0
    samples_per_task = np.zeros([A, 1])
    for i in range(0, A):
        if algorithm_flag == 'KTL-UCB-TaskSim':
            train_dataset = 'Train_Datasets_KTLUCB' + str(i)
        elif algorithm_flag == 'KTL-UCB-TaskSimEst':
            train_dataset = 'Train_Datasets_KTLEstUCB' + str(i)
        elif algorithm_flag == 'Lin-UCB-Ind':
            train_dataset = 'Train_Datasets_LinUCB' + str(i)
        elif algorithm_flag == 'Lin-UCB-Pool':
            train_dataset = 'Train_Datasets_PoolUCB' + str(i)
        #print DataXY.keys()
        X = np.copy(DataXY[train_dataset])
        total_samples = total_samples + X.shape[0]
        samples_per_task[i] = X.shape[0]

    # Collect all labels and all features
    y = np.zeros(total_samples)
    X_total = np.zeros([total_samples, X.shape[1]])
    rr = 0
    for i in range(0, A):
        if algorithm_flag == 'KTL-UCB-TaskSim':
            train_labels = 'Train_Labels_KTLUCB' + str(i)
            train_dataset = 'Train_Datasets_KTLUCB' + str(i)
        elif algorithm_flag == 'KTL-UCB-TaskSimEst':
            train_labels = 'Train_Labels_KTLEstUCB' + str(i)
            train_dataset = 'Train_Datasets_KTLEstUCB' + str(i)
        elif algorithm_flag == 'Lin-UCB-Ind':
            train_labels = 'Train_Labels_LinUCB' + str(i)
            train_dataset = 'Train_Datasets_LinUCB' + str(i)
        elif algorithm_flag == 'Lin-UCB-Pool':
            train_labels = 'Train_Labels_PoolUCB' + str(i)
            train_dataset = 'Train_Datasets_PoolUCB' + str(i)

        labels = np.copy(DataXY[train_labels])
        y[rr:rr + labels.shape[0]] = np.copy(DataXY[train_labels])
        X_total[rr:rr + labels.shape[0], :] = np.copy(DataXY[train_dataset])
        rr = rr + labels.shape[0]

    return total_samples, samples_per_task, y, X_total


def AddData(DataXY,arm_tt,algorithm_flag,X_test,reward_test,tt):
    if algorithm_flag == 'KTL-UCB-TaskSim':
        train_labels = 'Train_Labels_KTLUCB' + str(arm_tt)
        train_dataset = 'Train_Datasets_KTLUCB' + str(arm_tt)
        test_label = 'Test_Labels_KTLUCB'
        last_roundXTest =  'Test_Datasets_KTLUCB'
    elif algorithm_flag == 'KTL-UCB-TaskSimEst':
        train_labels = 'Train_Labels_KTLEstUCB' + str(arm_tt)
        train_dataset = 'Train_Datasets_KTLEstUCB' + str(arm_tt)
        test_label = 'Test_Labels_KTLEstUCB'
        last_roundXTest = 'Test_Datasets_KTLEstUCB'
    elif algorithm_flag == 'Lin-UCB-Ind':
        train_labels = 'Train_Labels_LinUCB' + str(arm_tt)
        train_dataset = 'Train_Datasets_LinUCB' + str(arm_tt)
        test_label = 'Test_Labels_LinUCB'
        last_roundXTest = 'Test_Datasets_LinUCB'
    elif algorithm_flag == 'Lin-UCB-Pool':
        train_labels = 'Train_Labels_PoolUCB' + str(arm_tt)
        train_dataset = 'Train_Datasets_PoolUCB' + str(arm_tt)
        test_label = 'Test_Labels_PoolUCB'
        last_roundXTest = 'Test_Datasets_PoolUCB'

    Total_Features = np.copy(DataXY[train_dataset])
    Arm_rewards = np.copy(DataXY[train_labels])

    Total_Features = np.append(Total_Features, X_test, axis=0)
    reward_test = np.ones([1])*reward_test
    Arm_rewards = np.append(Arm_rewards, reward_test, axis=0)

    DataXY[train_dataset] = np.copy(Total_Features)
    DataXY[train_labels] = np.copy(Arm_rewards)
    DataXY[last_roundXTest] = np.copy(X_test)

    if tt == 0:
        armSelectedTT =  np.ones([1])*arm_tt #np.empty([0])
    else:
        armSelectedTT = np.copy(DataXY[test_label])
        armSelectedTT = np.append(armSelectedTT, np.ones([1])*arm_tt, axis=0)
    armSelectedTT = armSelectedTT.astype(int)

    DataXY[test_label] = np.copy(armSelectedTT)

    return DataXY