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
    mainDataFolder = "C:\Users\Aniket\Downloads\KMTL-UCB Results\\CV_1_200\usps_CV_1_200"

    #expFolder = "A_16_T_1600_N_1_d_2_alpha_0.01"
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

    #mean_KTL = []
    #std_KTL = []

    mean_KTL_Est = []
    std_KTL_Est = []

    #mean_KTL_Rew_Est = []
    #std_KTL_Rew_Est = []

    mean_KTL_LinUCB = []
    std_KTL_LinUCB = []

    for t in range(0, numberOfTrials):
        #data_Vec_KTL = []
        data_Vec_Est = []
        #data_Vec_Rew_Est = []
        data_Vec_LinUCB = []
        for file in filesInTheFolder:
            #data_Vec_KTL.append(results[file][0][t])
            data_Vec_Est.append(results[file][0][t])
            #data_Vec_Rew_Est.append(results[file][2][t])
            data_Vec_LinUCB.append(results[file][1][t])

        #sampleMean_KTL, sampleStd_KTL = meanAndStd(data_Vec_KTL)
        #mean_KTL.append(sampleMean_KTL)
        #std_KTL.append(sampleStd_KTL)

        sampleMean_Est, sampleStd_Est = meanAndStd(data_Vec_Est)
        mean_KTL_Est.append(sampleMean_Est)
        std_KTL_Est.append(sampleStd_Est)

        #sampleMean_Rew_Est, sampleStd_Rew_Est = meanAndStd(data_Vec_Rew_Est)
        #mean_KTL_Rew_Est.append(sampleMean_Rew_Est)
        #std_KTL_Rew_Est.append(sampleStd_Rew_Est)

        sampleMean_LinUCB, sampleStd_LinUCB = meanAndStd(data_Vec_LinUCB)
        mean_KTL_LinUCB.append(sampleMean_LinUCB)
        std_KTL_LinUCB.append(sampleStd_LinUCB)


    #lower_KTL = []
    #upper_KTL = []
    lower_KTL_Est = []
    upper_KTL_Est = []
    #lower_KTL_Rew_Est = []
    #upper_KTL_Rew_Est = []
    lower_KTL_LinUCB = []
    upper_KTL_LinUCB = []

    for i in range(0, numberOfTrials):
        #lower_KTL.append(mean_KTL[i] - std_KTL[i])
        #upper_KTL.append(mean_KTL[i] + std_KTL[i])

        lower_KTL_Est.append(mean_KTL_Est[i] - std_KTL_Est[i])
        upper_KTL_Est.append(mean_KTL_Est[i] + std_KTL_Est[i])

        #lower_KTL_Rew_Est.append(mean_KTL_Rew_Est[i] - std_KTL_Rew_Est[i])
        #upper_KTL_Rew_Est.append(mean_KTL_Rew_Est[i] + std_KTL_Rew_Est[i])

        lower_KTL_LinUCB.append(mean_KTL_LinUCB[i] - std_KTL_LinUCB[i])
        upper_KTL_LinUCB.append(mean_KTL_LinUCB[i] + std_KTL_LinUCB[i])


    plt.rcParams.update({'font.size': 26})

    t_Range =  range(0, numberOfTrials)
    fig, ((ax1, ax2)) = plt.subplots(1,2)
    if mainDataFolder == "C:\Users\Aniket\Downloads\KMTL-UCB Results\USPS2000":
        markers_on = [100, 200, 300, 400, 500, 600,700,800,900,1000,1100,1200,1300,1400,1700]
    elif mainDataFolder == "C:\Users\Aniket\Downloads\KMTL-UCB Results\MNIST2000":
        markers_on = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1700]
    elif mainDataFolder == "C:\Users\Aniket\Downloads\KMTL-UCB Results\MNIST4000":
        markers_on = [100, 200, 300, 500, 700, 900, 1100, 1400, 1700,2000,2500,3000,3500]
    elif mainDataFolder == "C:\Users\Aniket\Downloads\KMTL-UCB Results\Digits":
        markers_on = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
    else:
        markers_on = [100, 200, 300, 400, 500, 600, 700, 800, 900]
    #ax1.plot(t_Range, mean_KTL,lw=4, label='KTL',color='black')
    ax1.plot(t_Range, mean_KTL_Est,marker = 'd',lw=4,label='KMTL-UCB-Est',color='blue',markevery=markers_on,markersize=20)
    #ax1.plot(t_Range, mean_KTL_Rew_Est,lw=4, label='KTL_Rew_Est',color='red')
    ax1.plot(t_Range, mean_KTL_LinUCB,marker = 'o',lw=4,label='Kernel-UCB-Ind',color='green', markevery=markers_on,markersize=20)

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

    ax2.set_title("Confidence Interval")
    ax2.plot(t_Range, mean_KTL_LinUCB,lw=4, label='Kernel-UCB-Ind',color='green',marker = 'o',markevery=markers_on,markersize=20)
    ax2.fill_between(t_Range, lower_KTL_LinUCB, upper_KTL_LinUCB, facecolor='green', alpha=0.5)
    ax2.plot(t_Range, mean_KTL_Est,lw=4, label='KMTL-UCB-Est',color='blue', marker = 'd',markevery=markers_on,markersize=20)
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
