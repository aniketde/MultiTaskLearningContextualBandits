
from __future__ import division
import numpy as np
import sklearn.metrics.pairwise as Kern

# Get final product kernel matrix for crossvalidation block
def GetKernelMatrixWithoutCorrectionTerm(X_total,A,total_samples,samples_per_task_train,bw_x, bw_prob, bw_prod,algorithm_flag,theta):
    #This is only for cross-validation
    #Calculate embedding

    # Calculate Task Similarity Kernel
    if algorithm_flag == 'KTL-UCB-TaskSimEst':
        Task_emb = np.zeros([A, A])
        rr = 0
        for i in range(0, A):
            Xi = X_total[rr:rr + samples_per_task_train[i], :]
            cc = 0
            for j in range(0, A):
                Xj = X_total[cc:cc + samples_per_task_train[j], :]
                K_task = Kern.rbf_kernel(Xi, Xj, bw_prob)
                Task_emb[i, j] = np.mean(K_task)
                cc = cc + samples_per_task_train[j]
            rr = rr + samples_per_task_train[i]
        Task_sim = np.zeros([A, A])
        rr = 0
        for i in range(0, A):
            #Xi = X_total[rr:rr + samples_per_task_train[i], :]
            cc = 0
            for j in range(0, A):
                # Xj = X_total[cc:cc + samples_per_task_train[j], :]
                sim = Task_emb[i,i] + Task_emb[j,j] - 2*Task_emb[i,j]
                Task_sim[i, j] = np.exp(-sim*bw_prod)
                cc = cc + samples_per_task_train[j]
            rr = rr + samples_per_task_train[i]
    elif algorithm_flag == 'KTL-UCB-TaskSim':
        ## ??? Write task similarity based on graph instead of theta
        Task_sim = Kern.rbf_kernel(theta, theta, bw_prod)
    elif algorithm_flag == 'Lin-UCB-Ind':
        Task_sim = np.identity(A)
    elif algorithm_flag == 'Lin-UCB-Pool':
        Task_sim = np.ones([A, A])

    # Calculate final kernel
    K_sim =  np.zeros([total_samples, total_samples])
    rr = 0
    for i in range(0, A):
        Xi = X_total[rr:rr + samples_per_task_train[i], :]
        cc = 0
        for j in range(0, A):
            Xj = X_total[cc:cc + samples_per_task_train[j], :]
            K_sim[rr:rr + samples_per_task_train[i],cc:cc + samples_per_task_train[j]] = Task_sim[i, j]*Kern.rbf_kernel(Xi, Xj, bw_x)
            cc = cc + samples_per_task_train[j]
        rr = rr + samples_per_task_train[i]


    return K_sim,Task_sim

# Get final product kernel matrix for main block. In future release both these functions will be combined
def GetTestKernelMatrixWithoutCorrectionTerm(X_train,X_test,A,Task_sim,samples_per_task_train,samples_per_task_test,bw_x):
    K_sim = np.zeros([X_train.shape[0], X_test.shape[0]])
    rr = 0
    for i in range(0, A):
        Xi = X_train[rr:rr + samples_per_task_train[i], :]
        cc = 0
        for j in range(0, A):
            Xj = X_test[cc:cc + samples_per_task_test[j], :]
            K_sim[rr:rr + samples_per_task_train[i], cc:cc + samples_per_task_test[j]] = Task_sim[
                                                                                             i, j] * Kern.rbf_kernel(Xi,
                                                                                                                     Xj,
                                                                                                                     bw_x)
            cc = cc + samples_per_task_test[j]
        rr = rr + samples_per_task_train[i]

    return K_sim


def GetKernelMatrix(DataXY,X_total,A,total_samples,samples_per_task,bw_x, theta, bw_prod,algorithm_flag,bw_prob,gamma):
    Task_emb = np.zeros([A, A])
    if algorithm_flag == 'KTL-UCB-TaskSim':
        ## ??? Write task similarity based on graph instead of theta
        Task_sim = Kern.rbf_kernel(theta, theta, bw_prod) #+ 0.75*np.random.rand(theta.shape[0],theta.shape[0])
        #plt.imshow(Task_sim)
        #plt.show()
    elif algorithm_flag == 'KTL-UCB-TaskSimEst':
        rr = 0
        for i in range(0, A):
            Xi = X_total[rr:rr + samples_per_task[i], :]
            cc = 0
            for j in range(0, A):
                Xj = X_total[cc:cc + samples_per_task[j], :]
                K_task = Kern.rbf_kernel(Xi, Xj, bw_prob)
                Task_emb[i, j] = np.mean(K_task)
                cc = cc + samples_per_task[j]
            rr = rr + samples_per_task[i]

        Task_sim = np.zeros([A, A])
        rr = 0
        for i in range(0, A):
            # Xi = X_total[rr:rr + samples_per_task_train[i], :]
            cc = 0
            for j in range(0, A):
                # Xj = X_total[cc:cc + samples_per_task_train[j], :]
                sim = Task_emb[i, i] + Task_emb[j, j] - 2 * Task_emb[i, j]
                Task_sim[i, j] = np.exp(-sim * bw_prod)
                cc = cc + samples_per_task[j]
            rr = rr + samples_per_task[i]
    elif algorithm_flag == 'Lin-UCB-Ind':
        Task_sim = np.identity(A)
    elif algorithm_flag == 'Lin-UCB-Pool':
        Task_sim = np.ones([A,A])

    # Calculate final kernel

    K_sim =  np.zeros([total_samples, total_samples])
    rr = 0
    for i in range(0, A):
        Xi = X_total[rr:rr + samples_per_task[i], :]
        cc = 0
        for j in range(0, A):
            Xj = X_total[cc:cc + samples_per_task[j], :]
            K_sim[rr:rr + samples_per_task[i],cc:cc + samples_per_task[j]] = Task_sim[i, j]*Kern.rbf_kernel(Xi, Xj, bw_x)
            #print Kern.rbf_kernel(Xi, Xj, bw_x)
            cc = cc + samples_per_task[j]
        rr = rr + samples_per_task[i]

    #correction term
    eta_arm = np.zeros([X_total.shape[0]])
    rr = 0
    for i in range(0, A):
        if algorithm_flag == 'KTL-UCB-TaskSim':
            train_dataset = 'Train_Datasets_KTLUCB' + str(i)
        elif algorithm_flag == 'KTL-UCB-TaskSimEst':
            train_dataset = 'Train_Datasets_KTLEstUCB' + str(i)
        elif algorithm_flag == 'Lin-UCB-Ind':
            train_dataset = 'Train_Datasets_LinUCB' + str(i)
        elif algorithm_flag == 'Lin-UCB-Pool':
            train_dataset = 'Train_Datasets_PoolUCB' + str(i)
        X = np.copy(DataXY[train_dataset])
        eta_arm[rr:rr + X.shape[0]] = np.kron(1 / X.shape[0], np.ones(X.shape[0]))
        rr = rr + X.shape[0]

    eta = np.diag(eta_arm)
    InvTerm = np.linalg.inv(eta.dot(K_sim) + gamma * np.identity(K_sim.shape[0]))

    DataXY[algorithm_flag+'_TaskSim'] = np.copy(Task_sim)
    DataXY[algorithm_flag + '_TaskEmb'] = np.copy(Task_emb)
    DataXY[algorithm_flag + 'KSim'] = np.copy(K_sim)
    DataXY[algorithm_flag + 'etaArm'] = np.copy(eta_arm)
    DataXY[algorithm_flag + 'InvMat'] = np.copy(InvTerm)

    return K_sim, eta_arm,Task_sim,DataXY
