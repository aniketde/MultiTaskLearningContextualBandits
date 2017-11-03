
from __future__ import division
import numpy as np
import sklearn.metrics.pairwise as Kern
import DataCreate.DataCreate as DC
import KernelCalculation.GaussianKernels as GK


#This function runs the bandit algorithm at time t
def ContextBanditUCB(DataXY, tt, bw_x, bw_prod, gamma,bw_prob, alpha, data_flag,algorithm_flag):
    #KTLUCB
    total_samples, samples_per_task, y, X_total = DC.AllDataCollect(DataXY,algorithm_flag)

    #print algorithm_flag, X_total.shape
    samples_per_task = samples_per_task.astype(int)
    A = DataXY['NoOfArms']
    theta = DataXY['theta']

    K_sim, eta_arm,Task_sim,DataXY = GK.GetKernelMatrix(DataXY,X_total,A,total_samples,samples_per_task,bw_x, theta, bw_prod,algorithm_flag,bw_prob,gamma)

    # Run the UCB estimate
    # KRR Estimate using training data and direct inverse
    eta = np.diag(eta_arm)

    InvTerm = np.linalg.inv(eta.dot(K_sim) + gamma * np.identity(K_sim.shape[0]))

    reward = np.zeros([A, 1])
    reward_est = np.zeros([A])

    if data_flag == 4:
        #aa = tt % A
        armTest = DataXY['armTest']
        aa = armTest[tt]
        #X_test = theta[aa, :] + rng.randn(1, X_total.shape[1])
        Testfeatures = DataXY['Testfeatures']
        X_test = Testfeatures[tt, :]
        X_test = X_test[np.newaxis, :]
    elif  data_flag == 7:
        armTest = DataXY['armTest']
        aa = armTest[tt]
        Testfeatures = DataXY['Testfeatures']
        X_test = Testfeatures[tt, :]
        X_test = X_test[np.newaxis, :]


    for aa in range(0, A):
        if data_flag == 3:
            UserContext = DataXY['UserContext']
            X_dummy = UserContext[tt,:]
            X_dummy = X_dummy[np.newaxis, :]
            R = np.array([[np.cos(theta[aa, 0]), -np.sin(theta[aa, 0])], [np.sin(theta[aa, 0]), np.cos(theta[aa, 0])]])
            X_test = R.dot(X_dummy.T).T

        K_x = np.zeros([X_total.shape[0], X_test.shape[0]])
        rr = 0
        for i in range(0, A):
            Xi = X_total[rr:rr + samples_per_task[i], :]
            K_x[rr:rr + samples_per_task[i], :] = Task_sim[i, aa] * Kern.rbf_kernel(Xi,X_test,bw_x)
            rr = rr + samples_per_task[i]


        k_x_a = Kern.rbf_kernel(X_test, X_test, bw_x)

        reward_est[aa] = np.transpose(K_x).dot(InvTerm).dot(eta).dot(y)
        reward_conf = k_x_a - np.transpose(K_x).dot(InvTerm).dot(eta).dot(K_x)

        '''
        if  k_x_a - np.transpose(K_x).dot(InvTerm).dot(eta).dot(K_x) < 0:
            print tt, aa, reward_conf
            reward_conf = 0.0
        '''

        reward[aa] = reward_est[aa] + alpha * np.sqrt(reward_conf)

    if data_flag == 3:
        UserContext = DataXY['UserContext']
        X_dummy = UserContext[tt,:]
        X_dummy = X_dummy[np.newaxis,:]
        selected_arm = np.argmax(reward)
        R = np.array([[np.cos(theta[selected_arm, 0]), -np.sin(theta[selected_arm, 0])], [np.sin(theta[selected_arm, 0]), np.cos(theta[selected_arm, 0])]])
        X_test = R.dot(X_dummy.T).T

    return reward,X_test,DataXY