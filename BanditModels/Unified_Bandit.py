from __future__ import division
import numpy as np
import DataCreate.DataCreate as DC
import Algorithms.UCB as UCB
import warnings
warnings.filterwarnings("ignore")

#f_bw <- bw_x_estKTLUCB
#s_bw <- bw_prod_estKTLUCB
#gamma <- gammaKTLUCB
#t_bw <- bw_prob_estKTLUCB
class Unified_Bandit:
    def __init__(self,f_bw, s_bw, gamma, t_bw, alpha, data_flag,type):
        self.type = type
        self.f_bw = f_bw
        self.s_bw = s_bw
        self.gamma = gamma
        self.t_bw = t_bw
        self.alpha = alpha
        self.data_flag = data_flag
        self.collected_rewards = []

    #data <- DataXY
    #time_Step <- tt
    def get_Arm_And_X_test_And_Data(self, data, time_Step):
        estimated_Rewards, X_test, data = UCB.ContextBanditUCB(data, time_Step, self.f_bw , self.s_bw, self.gamma, self.t_bw, self.alpha, self.data_flag, self.type)
        self.selected_arm = np.argmax(estimated_Rewards)
        return self.selected_arm, X_test, data
    
    
    def get_User_Context_Based_Reward(self,user_Context,arm ):
        rotated_Context = self.get_Rotatated_Context(user_Context,arm)
        return  1.0 - (rotated_Context[:, 1] - self.Reward_Distribution[arm] + 0.5) ** 2
        
    def update_Collected_Rewards(self, data, time_Step,ind_arm):
        if self.data_flag == 3:
            user_Context = self.get_User_Context(time_Step)
            self.collected_rewards.append( self.get_User_Context_Based_Reward(user_Context, self.selected_arm))
        elif (self.data_flag == 4):
            #print ind_arm,  self.selected_arm
            self.collected_rewards.append(DC.reward_from_labels(ind_arm, self.selected_arm, self.Hier_Graph, self.Reward_Funct,
                                  self.Hier_Classes))
        elif (self.data_flag == 7):
            #print ind_arm,  self.selected_arm
            self.collected_rewards.append( DC.reward_from_labels_Multiclass(ind_arm, self.selected_arm))
    
    #All these functions are needed for data_flag = 3
    def set_All_User_Context(self,context):
        self.User_Context = context
        
    def set_All_Arm_Context(self,context):
        self.Arm_Context = context
    
    def set_Reward_Distribution(self,rew_Dist):
        self.Reward_Distribution = rew_Dist
    
    def get_User_Context(self,time_Step):
        #Check the size of user context if it is 0, then raise an exception
        x = self.User_Context[time_Step,:]
        return x[np.newaxis, :]
        
    def get_Arm_Context(self):
        #Check the size of arm context if it is 0, then raise an exception
        return  self.Arm_Context
        
    def get_Reward_Distribuiton(self):
        return self.Reward_Distribution
        
    def get_Rotatated_Context(self, user_context, index):
        mat = np.array([[np.cos(self.Arm_Context[index, 0]), -np.sin(self.Arm_Context[index, 0])],[np.sin(self.Arm_Context[index, 0]), np.cos(self.Arm_Context[index, 0])]])
        return mat.dot(user_context.T).T
       

    def get_Reward_For_All_Arms(self,data,time_Step):
        A = data['NoOfArms']
        RewardforAllArms = np.zeros([A])
        user_Context = self.get_User_Context(time_Step)
        for aa in range(0, A):
            RewardforAllArms[aa] =  self.get_User_Context_Based_Reward(user_Context, aa)
        return RewardforAllArms
    
    def get_Collected_Rewards(self):
        return self.collected_rewards


    #All the following functions are needed for data_flag = 4,5

    def set_Reward_Function_Flag(self,reward_funct):
        self.Reward_Funct = reward_funct

    def set_Hier_Graph(self,Graph):
        self.Hier_Graph = Graph

    def set_Hier_Classes(self,Classes):
        self.Hier_Classes = Classes




