import numpy as np
import os

import matplotlib.pyplot as plt


def meanAndStd(data):
    n = len(data)
    mean =  sum(data)/n
    std = sum((x-mean)**2 for x in data)
    std = std/n
    std = std ** 0.5
    return mean, std

def main():

    mainDataFolder = "C:\Users\Aniket\Desktop\KMTLUCB_NIPSCode\ExperimentResults\data_flag_3\synthetic1_Excel"
    #expFolder = "A_16_T_1600_N_1_d_2_alpha_0.01"
    #expFolder = "csv"
    #expFolder = "A_16_T_1600_N_1_d_2_alpha_0.5"

    fullExperimentFolderPath = mainDataFolder#os.path.join(mainDataFolder,expFolder)
    print os.getcwd()
    print mainDataFolder
    print fullExperimentFolderPath
    print os.listdir(fullExperimentFolderPath)
    filesInTheFolder = [f for f in os.listdir(fullExperimentFolderPath) if os.path.isfile(os.path.join(fullExperimentFolderPath, f))]
    
    numberOfFiles = len(filesInTheFolder)
    results = {}
    for file in filesInTheFolder:
        fullPathOfFile = os.path.join(fullExperimentFolderPath,file)
        data = np.genfromtxt(fullPathOfFile, dtype=float, delimiter=',')
        results[file] = data



    dataShape = results[filesInTheFolder[0]].shape
    numberOfTrials = dataShape[1]
    
   

    mean_KTL_Est = []
    std_KTL_Est = []

    mean_KTL_LinUCB = []
    std_KTL_LinUCB = []

    mean_KTL = []
    std_KTL = []

    mean_KTL_PoolUCB = []
    std_KTL_PoolUCB = []

    #mean_KTL_Rew_Est = []
    #std_KTL_Rew_Est = []

    for t in range(0, numberOfTrials):
      
        data_Vec_Est = []
        data_Vec_Rew_Est = []
        data_Vec_LinUCB = []
        data_Vec_KTL = []
        data_Vec_PoolUCB = []
        for file in filesInTheFolder:
            #data_Vec_KTL.append(results[file][0][t])
            data_Vec_Est.append(results[file][0][t])
            data_Vec_LinUCB.append(results[file][1][t])
            data_Vec_KTL.append(results[file][2][t])
            data_Vec_PoolUCB.append(results[file][3][t])
            #data_Vec_Rew_Est.append(results[file][4][t])
            
        

        sampleMean_Est, sampleStd_Est = meanAndStd(data_Vec_Est)
        mean_KTL_Est.append(sampleMean_Est)
        std_KTL_Est.append(sampleStd_Est)

        sampleMean_LinUCB, sampleStd_LinUCB = meanAndStd(data_Vec_LinUCB)
        mean_KTL_LinUCB.append(sampleMean_LinUCB)
        std_KTL_LinUCB.append(sampleStd_LinUCB)

        sampleMean_KTL, sampleStd_KTL = meanAndStd(data_Vec_KTL)
        mean_KTL.append(sampleMean_KTL)
        std_KTL.append(sampleStd_KTL)

        sampleMean_PoolUCB, sampleStd_PoolUCB = meanAndStd(data_Vec_PoolUCB)
        mean_KTL_PoolUCB.append(sampleMean_PoolUCB)
        std_KTL_PoolUCB.append(sampleStd_PoolUCB)

        #sampleMean_Rew_Est, sampleStd_Rew_Est = meanAndStd(data_Vec_Rew_Est)
        #mean_KTL_Rew_Est.append(sampleMean_Rew_Est)
        #std_KTL_Rew_Est.append(sampleStd_Rew_Est)

    lower_KTL_Est = []
    upper_KTL_Est = []
    lower_KTL_LinUCB = []
    upper_KTL_LinUCB = []
    lower_KTL = []
    upper_KTL = []
    lower_KTL_PoolUCB = []
    upper_KTL_PoolUCB = []
    #lower_KTL_Rew_Est = []
    #upper_KTL_Rew_Est = []

    for i in range(0, numberOfTrials):
        
        lower_KTL_Est.append(mean_KTL_Est[i] - std_KTL_Est[i])
        upper_KTL_Est.append(mean_KTL_Est[i] + std_KTL_Est[i])

        lower_KTL_LinUCB.append(mean_KTL_LinUCB[i] - std_KTL_LinUCB[i])
        upper_KTL_LinUCB.append(mean_KTL_LinUCB[i] + std_KTL_LinUCB[i])

        lower_KTL.append(mean_KTL[i] - std_KTL[i])
        upper_KTL.append(mean_KTL[i] + std_KTL[i])

        lower_KTL_PoolUCB.append(mean_KTL_PoolUCB[i] - std_KTL_PoolUCB[i])
        upper_KTL_PoolUCB.append(mean_KTL_PoolUCB[i] + std_KTL_PoolUCB[i])

        #lower_KTL_Rew_Est.append(mean_KTL_Rew_Est[i] - std_KTL_Rew_Est[i])
        #upper_KTL_Rew_Est.append(mean_KTL_Rew_Est[i] + std_KTL_Rew_Est[i])

    plt.rcParams.update({'font.size': 22})

    t_Range =  range(0, numberOfTrials)
    fig, ((ax1, ax2)) = plt.subplots(1,2)
    markers_on = [100, 200, 300, 400, 500, 600, 700, 800, 900]
    ax1.plot(t_Range, mean_KTL_Est,lw=4, label='KMTL-UCB-Est',color='blue',ls='--',marker = 'd',markevery=markers_on,markersize=20)
    ax1.plot(t_Range, mean_KTL_LinUCB,lw=4, label='Kernel-UCB-Ind',color='green',ls='--',marker = 'o',markevery=markers_on,markersize=20)
    ax1.plot(t_Range, mean_KTL,lw=4, label='KMTL-UCB',color='red', ls='--',marker = "v",markevery=markers_on,markersize=20)
    ax1.plot(t_Range, mean_KTL_PoolUCB,lw=4, label='Kernel-UCB-Pool',color='black', ls='--',marker = 's',markevery=markers_on,markersize=20)
    #ax1.plot(t_Range, mean_KTL_Rew_Est, lw=1, label='KTL_Rew_Est', color='yellow', ls='--')
	
    ax1.set_title("Empirical Mean Regrets")
    ax1.legend(loc='upper left') 
    ax1.set_xlabel('Trials')
    ax1.set_ylabel('Cumulative Regret')

      
    #ax2.set_title("LinUCB vs KTL")
    #ax2.plot(t_Range, mean_KTL_LinUCB,lw=4, label='LinUCB',color='green') 
    #ax2.fill_between(t_Range, lower_KTL_LinUCB, upper_KTL_LinUCB, facecolor='green', alpha=0.5)  
    #ax2.plot(t_Range, mean_KTL,lw=4, label='KTL',color='black')
    #ax2.fill_between(t_Range, lower_KTL, upper_KTL, facecolor='black', alpha=0.5)
    #ax2.legend(loc='upper left') 
    #ax2.set_xlabel('Trials')
    #ax2.set_ylabel('Cumulative Regret')

    ax2.set_title("LinUCB vs KTL_Est")
    ax2.plot(t_Range, mean_KTL_LinUCB,lw=4, label='Kernel-UCB-Ind',color='green')
    ax2.fill_between(t_Range, lower_KTL_LinUCB, upper_KTL_LinUCB, facecolor='green', alpha=0.5)  
    ax2.plot(t_Range, mean_KTL_Est,lw=4, label='KMTL-UCB-Est',color='blue')
    ax2.fill_between(t_Range, lower_KTL_Est, upper_KTL_Est, facecolor='blue', alpha=0.5)
    ax2.legend(loc='upper left') 
    ax2.set_xlabel('Trials')
    ax2.set_ylabel('Cumulative Regret')

    #ax4.set_title("LinUCB vs KTL_Rew_Est")
    #ax4.plot(t_Range, mean_KTL_LinUCB,lw=4, label='LinUCB',color='green') 
    #ax4.fill_between(t_Range, lower_KTL_LinUCB, upper_KTL_LinUCB, facecolor='green', alpha=0.5)  
    #ax4.plot(t_Range, mean_KTL_Rew_Est,lw=4, label='KTL_Rew_Est',color='red')
    #ax4.fill_between(t_Range, lower_KTL_Rew_Est, upper_KTL_Rew_Est, facecolor='red', alpha=0.5)
    #ax4.legend(loc='upper left') 
    #ax4.set_xlabel('Trials')
    #ax4.set_ylabel('Cumulative Regret')


    #fig.tight_layout()
    plt.show()
 
    Urun = 1
    


if __name__ == "__main__":
    main()
