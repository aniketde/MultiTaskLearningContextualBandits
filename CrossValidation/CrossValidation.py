
from __future__ import division
import numpy as np
import DataCreate.DataCreate as DC
import KernelCalculation.GaussianKernels as GK
from sklearn.cross_validation import KFold

#alpha estimation for Kernel Ridge Regression
def alphaEst(X_train,lambda_reg,Y_train,samples_per_task_train,K_train,A):
    eta_arm_train = np.zeros([X_train.shape[0]])
    rr = 0
    for i in range(0, A):
        eta_arm_train[rr:rr + samples_per_task_train[i]] = np.kron(1 / samples_per_task_train[i],
                                                                   np.ones(samples_per_task_train[i]))
        rr = rr + samples_per_task_train[i]

    # K_similarity_of_Arms = X_train.dot(np.transpose(X_train))
    # sanity = np.amin(K_similarity_of_Arms)
    # print sanity, bb1, ll, cv
    eta_train = np.diag(eta_arm_train)
    inv_term_est = np.linalg.inv(eta_train.dot(K_train) + lambda_reg * np.identity(K_train.shape[0]))

    alpha_est = np.dot(inv_term_est, eta_train.dot(Y_train))

    alpha_est = alpha_est[np.newaxis, :]

    return alpha_est

#Main cross-validation block
def CrossValidRegression(bw_x_grid,lambda_reg_grid,bw_prod_grid,bw_prob_grid,fold_cv,algorithm_flag,data_flag, data_flag_multiclass,A, d, N_valid,N, T,randomSeeds,Main_Program_flag):
    #It is assumed that A is a multiple of 5 and
    DataXY = DC.TrainDataCollect(data_flag,data_flag_multiclass, A, d, N_valid,N, T,randomSeeds[0],Main_Program_flag)
    A = DataXY['NoOfArms']
    theta = DataXY['theta']
    total_samples, samples_per_task, y, X_total = DC.AllDataCollect(DataXY,algorithm_flag)
    samples_per_task = samples_per_task.astype(int)
    kf = KFold(samples_per_task[0], n_folds=fold_cv)
    err = np.zeros([bw_x_grid.shape[0],bw_prod_grid.shape[0],bw_prob_grid.shape[0] ,lambda_reg_grid.shape[0]])
    for bb1 in range(0, bw_x_grid.shape[0]):
        bw_x = bw_x_grid[bb1]
        for bb3 in range(0, bw_prod_grid.shape[0]):
            bw_prod = bw_prod_grid[bb3]
            for bb2 in range(0, bw_prob_grid.shape[0]):
                bw_prob = bw_prob_grid[bb2]
                for ll in range(0, lambda_reg_grid.shape[0]):
                    lambda_reg = lambda_reg_grid[ll]
                    err_cv = np.zeros([fold_cv, 1])
                    cv = 0
                    print "parameters"
                    print bb1,bb3,bb2,ll
                    for train_index, test_index in kf:
                        ind_all = np.linspace(0, total_samples - 1, total_samples)
                        ind_all = ind_all.astype(int)

                        ind_test = np.zeros([A*test_index.shape[0]]).astype(int)
                        samples_per_task_test = np.copy(samples_per_task)
                        samples_per_task_train = np.copy(samples_per_task)
                        for ii in range(0,A):
                            #print  ii,ind_test[ii*test_index.shape[0]:(ii+1)*test_index.shape[0]].shape,ind_all[ii*samples_per_task[ii]:(test_index.shape[0]+ii*samples_per_task[ii])].shape
                            ind_test[ii*test_index.shape[0]:(ii+1)*test_index.shape[0]] = ind_all[ii*samples_per_task[ii]:(test_index.shape[0]+ii*samples_per_task[ii])]
                            samples_per_task_test[ii] = int(test_index.shape[0])
                            samples_per_task_train[ii] = int(train_index.shape[0])

                        ind_train = np.delete(ind_all, ind_test)
                        X_train = X_total[ind_train, :]
                        X_test = X_total[ind_test, :]
                        Y_train = y[ind_train]
                        Y_test = y[ind_test]

                        K_train,Task_sim = GK.GetKernelMatrixWithoutCorrectionTerm(X_train, A,X_train.shape[0], samples_per_task_train,bw_x, bw_prob, bw_prod,algorithm_flag,theta)


                        alpha_est = alphaEst(X_train,lambda_reg,Y_train,samples_per_task_train,K_train,A)

                        K_test = GK.GetTestKernelMatrixWithoutCorrectionTerm(X_train, X_test, A,Task_sim, samples_per_task_train,samples_per_task_test, bw_x)
                        #print alpha_est.shape, K_test.shape
                        Y_est = np.dot(alpha_est, K_test)
                        Y_est = Y_est[0,:]

                        err_cv[cv] = np.linalg.norm(Y_test - Y_est)
                        cv = cv+1
                    err[bb1,bb3,bb2, ll] = np.mean(err_cv)

    bb1_min, bb3_min,bb2_min,ll_min = np.where(err == err.min())
    bb1_min = np.ones([1])* bb1_min
    bb3_min = np.ones([1]) * bb3_min
    bb2_min = np.ones([1]) * bb2_min
    ll_min = np.ones([1]) * ll_min
    bw_x_est = bw_x_grid[bb1_min[0]]
    bw_prob_est = bw_prob_grid[bb2_min[0]]
    bw_prod_est = bw_prod_grid[bb3_min[0]]
    lambda_reg_est = lambda_reg_grid[ll_min[0]]
    print "minimium Error is: " + str(err.min()/Y_test.shape[0])
    return bw_x_est, bw_prob_est, bw_prod_est, lambda_reg_est
