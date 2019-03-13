import numpy as np
from utils import load_data,visualize_trajectory_2d,visualize_trajectory_2d_2
from utils_function_EKF import gethatmap, getu_head, getu_vee, getpredict_mu, getpredict_cov, getM, getz, jacobian, getcircle, hatoperation
import matplotlib.pyplot as plt
from scipy.linalg import expm


class data():
    def load27(self):
        filename = "./data/0027.npz"
        return filename
    def load42(self):
        filename = "./data/0042.npz"
        return filename
    def load20(self):
        filename = "./data/0020.npz"
        return filename
    
class EKF():
    def __init__(self):
        mydata = data()
        self.filename = mydata.load27()
        self.t,self.features,self.linear_velocity,self.rotational_velocity,self.K,self.b,self.cam_T_imu = load_data(self.filename)
        self.t_len = len(np.transpose(self.t))
        self.feature_num = len(self.features[:,:,0][0])
        self.D = [[1,0,0],[0,1,0],[0,0,1],[0,0,0]]

    def prediction(self):
        mu_t_t = np.identity(4)
        cov_t_t = np.identity(6)
        velocity = np.transpose(self.linear_velocity)
        omega = np.transpose(self.rotational_velocity)
        time = np.transpose(self.t)
        
        pose_list = []
        pose_list.append(mu_t_t)
        mu_list = []
        cov_list = [] 
        for i in range(self.t_len-1):
            noise_w = np.random.rand(1)
            dt = time[i+1]-time[i]
            omega_hat = gethatmap(omega[i])
            u_head = getu_head(omega_hat,velocity[i])

            v_hat = gethatmap(velocity[i])
            u_vee = getu_vee(omega_hat,v_hat)
           
           
            mu_t_t = getpredict_mu(dt,u_head,mu_t_t) #world to body 
            cov_t_t = getpredict_cov(dt,u_vee,cov_t_t,noise_w)
           
            mu_list.append(mu_t_t)
            pose = np.linalg.inv(mu_t_t)
            pose_list.append(pose)
            cov_list.append(cov_t_t)
        poses_list= np.stack((pose_list), axis = -1)
        fig, ax = visualize_trajectory_2d(poses_list,path_name="imu trajectory",show_ori=True)
        return mu_list
    
    def update(self,mu_list):

        cov_t = np.zeros((self.feature_num,3,3))
        cov_t[:] = np.identity(3)

        M = getM(self.K,self.b)
        
        
        #initialize mu prior of each features at first time
        mu_t = np.zeros((4,self.feature_num))
        for i in range(self.t_len-1):
            first_pic = self.features[:,:,i]
            for j in range(self.feature_num):
                if (first_pic[:,j] == [-1.,-1.,-1.,-1.]).all():
                    continue
                else: 
                    if (mu_t[:,j] == [0,0,0,0]).all():
                        feature = first_pic[:,j]
                        Z = getz(feature,M) 
                        temp = Z * np.dot(np.linalg.pinv(M),feature)
                        inv_o_T_i = np.linalg.pinv(self.cam_T_imu)
                        inv_i_T_t = np.linalg.pinv(mu_list[i])
                        mu_temp = np.dot(np.dot(inv_i_T_t,inv_o_T_i),temp)
                        mu_t[:,j] = mu_temp 
                        
                    else: 
                        continue
            
        plt.scatter(mu_t[0,:],mu_t[1,:],label="prior with raw camera data")
        mu_update = mu_t


        for i in range(self.t_len-1):
            for j in range(self.feature_num):

                pic = self.features[:,:,i]
                if (pic[:,j] == [-1.,-1.,-1.,-1]).all():
                    continue
                
                else:
                    noise_v = np.random.rand(4,4)
                    IV = noise_v * np.identity(4)
                    z = pic[:,j]
                    o_T_i_T_t = np.dot(self.cam_T_imu,mu_list[i])
                    q = np.dot(o_T_i_T_t,mu_update[:,j])
                    pi_function = q/z
                    dpi_function = jacobian(q)
                    
                    H = np.dot(np.dot(np.dot(M,dpi_function),o_T_i_T_t),self.D)
                    HT = np.transpose(H)
                    z_hat = np.dot(M,pi_function)
                    k_gain = np.dot(np.dot(cov_t[j],HT),np.linalg.pinv(np.dot(np.dot(H,cov_t[j]),HT)+IV))
                
                    correction = z - z_hat

                    mu_update[:,j] = mu_update[:,j] + np.dot(np.dot(self.D,k_gain), correction)
                    cov_t[j] = np.dot((np.identity(3)-np.dot(k_gain,H)), cov_t[j])

                    
        plt.scatter(mu_update[0,:],mu_update[1,:],label="updated with imu localization")
        plt.legend()
        plt.show()   
        
    def vslam(self,mu_list):
        M = getM(self.K,self.b)
        velocity = np.transpose(self.linear_velocity)
        omega = np.transpose(self.rotational_velocity)
        time = np.transpose(self.t)
        #initialize robot pose and cov 
        mu_t_pose = np.identity(4)
        cov_t_pose = np.identity(6)
        
        #initialize mean and cov prior of each features at first time
        mu_t_feature = np.zeros((4,self.feature_num))
        cov_t_feature = np.zeros((self.feature_num,3,3))
        cov_t_feature[:] = np.identity(3)

        result_pose = []
        result_pose.append(mu_t_pose)
        #update robot pose by EKF update with observation model
        for i in range(self.t_len-1):
            #new observation to get m
            pic = self.features[:,:,i]
            H = []
            z = []
            z_hat = []
            count = 0 

            for j in range(self.feature_num):
                if (pic[:,j] == [-1.,-1.,-1.,-1.]).all():
                    continue
                else: 
                    feature = pic[:,j]
                    Z = getz(feature,M) 
                    temp = Z * np.dot(np.linalg.pinv(M),feature)
                    
                    inv_o_T_i = np.linalg.pinv(self.cam_T_imu)
                    inv_i_T_t = np.linalg.pinv(mu_t_pose)
                    m = np.dot(inv_i_T_t,np.dot(inv_o_T_i, temp))#feature in world frame 
                    
                    o_T_i_T_t = np.dot(self.cam_T_imu,mu_t_pose)
                    q = np.dot(self.cam_T_imu,np.dot(mu_t_pose,m))
                    pi_function = q/Z
                    dpi_function = jacobian(q)
                    #compute z_hat
                    z_hati = np.dot(M,pi_function)
                            
                    zi = feature
                    #compute H 
                    circle = getcircle(np.dot(mu_t_pose,m))
                    Hi = np.dot(np.dot(np.dot(M,dpi_function),self.cam_T_imu),circle)
                    H.append(Hi)
                    z.append(zi)
                    z_hat.append(z_hati)
                    count += 1 
            if count == 0: 
                mu_t_pose = mu_t_pose
                cov_t_pose = cov_t_pose  
                pose = np.linalg.pinv(mu_t_pose)
                result_pose.append(pose)
            else:
                H = np.vstack((H))
                z = np.vstack((z))
                z_hat = np.vstack((z_hat))
    
       
                #update the robot pose by observation    
                noise_v = 0.01*np.random.rand(1)
                IV = noise_v * np.identity(len(H))     
                HT = np.transpose(H)
                k_gain = np.dot(np.dot(cov_t_pose,HT),np.linalg.pinv(np.dot(np.dot(H,cov_t_pose),HT)+IV))
                
                z = z.reshape([-1,1])
                z_hat = z_hat.reshape([-1,1])
                correction = z - z_hat
                print(correction)
                update_term = np.dot(k_gain,correction)
                update_term_exp = expm(hatoperation(update_term))
                
                mu_t_pose = np.dot(update_term_exp,mu_t_pose)
                cov_t_pose = np.dot((np.identity(6)-np.dot(k_gain,H)),cov_t_pose)
                
                #store poses to plot trajectory
                pose = np.linalg.inv(mu_t_pose)
                result_pose.append(pose)
            
            #update the feature position with the updated pose 
            for j in range(self.feature_num):
                pic = self.features[:,:,i]
                if (pic[:,j] == [-1.,-1.,-1.,-1]).all():
                    continue
                
                else:
                    #if meet [0,0,0,0] ---> set prior
                    if (mu_t_feature[:,j] == [0,0,0,0]).all:
                        feature = pic[:,j]
                        Z = getz(feature,M) 
                        temp = Z * np.dot(np.linalg.pinv(M),feature)
                        inv_o_T_i = np.linalg.pinv(self.cam_T_imu)
                        inv_i_T_t = np.linalg.pinv(mu_t_pose)
                        mu_temp = np.dot(np.dot(inv_i_T_t,inv_o_T_i),temp)
                        mu_t_feature[:,j] = mu_temp 
                    else:
                        noise_vv = 0.01*np.random.rand(1)
                        IV = noise_vv * np.identity(4)
                        z = pic[:,j]
                        o_T_i_T_t = np.dot(self.cam_T_imu,mu_t_pose)
                        q = np.dot(o_T_i_T_t,mu_t_feature[:,j])
                        q3 = q[2]
                        pi_function = q/q3
                        dpi_function = jacobian(q)
                        
                        Hijt = np.dot(np.dot(np.dot(M,dpi_function),o_T_i_T_t),self.D)
                        HT = np.transpose(Hijt)
                        z_hat = np.dot(M,pi_function)
                        k_gain = np.dot(np.dot(cov_t_feature[j],HT),np.linalg.pinv(np.dot(np.dot(Hijt,cov_t_feature[j]),HT)+IV))
                    
                        correction = z - z_hat
    
                        mu_t_feature[:,j] = mu_t_feature[:,j] + np.dot(np.dot(self.D,k_gain), correction)
                        cov_t_feature[j] = np.dot((np.identity(3)-np.dot(k_gain,H)), cov_t_feature[j])  
                        
            #control input update robot pose
            noise_w = 0.1*np.random.rand(1)
            dt = time[i+1]-time[i]
            omega_hat = gethatmap(omega[i])
            u_head = getu_head(omega_hat,velocity[i])

            v_hat = gethatmap(velocity[i])
            u_vee = getu_vee(omega_hat,v_hat)
           
           
            mu_t_pose = getpredict_mu(dt,u_head,mu_t_pose) #world to body 
            cov_t_pose = getpredict_cov(dt,u_vee,cov_t_pose,noise_w)
            
        result_pose= np.stack((result_pose), axis = -1)
        
        
        # assume imu trajectory is correct and update landmarks
        mu_t = np.zeros((4,self.feature_num))
        for i in range(self.t_len-1):
            first_pic = self.features[:,:,i]
            for j in range(self.feature_num):
                if (first_pic[:,j] == [-1.,-1.,-1.,-1.]).all():
                    continue
                else: 
                    if (mu_t[:,j] == [0,0,0,0]).all():
                        feature = first_pic[:,j]
                        Z = getz(feature,M) 
                        temp = Z * np.dot(np.linalg.pinv(M),feature)
                        inv_o_T_i = np.linalg.pinv(self.cam_T_imu)
                        inv_i_T_t = np.linalg.pinv(mu_list[i])
                        mu_temp = np.dot(np.dot(inv_i_T_t,inv_o_T_i),temp)
                        mu_t[:,j] = mu_temp 
                        
                    else: 
                        continue
                    
        # imu data pose          
        mu_t_t = np.identity(4)
        cov_t_t = np.identity(6)
        velocity = np.transpose(self.linear_velocity)
        omega = np.transpose(self.rotational_velocity)
        time = np.transpose(self.t)
        
        pose_list = []
        pose_list.append(mu_t_t)
        mu_list = []
        cov_list = [] 
        for i in range(self.t_len-1):
            noise_w = np.random.rand(1)
            dt = time[i+1]-time[i]
            omega_hat = gethatmap(omega[i])
            u_head = getu_head(omega_hat,velocity[i])

            v_hat = gethatmap(velocity[i])
            u_vee = getu_vee(omega_hat,v_hat)
           
           
            mu_t_t = getpredict_mu(dt,u_head,mu_t_t) #world to body 
            cov_t_t = getpredict_cov(dt,u_vee,cov_t_t,noise_w)
           
            mu_list.append(mu_t_t)
            pose = np.linalg.inv(mu_t_t)
            pose_list.append(pose)
            cov_list.append(cov_t_t)
        poses_list= np.stack((pose_list), axis = -1)
        visualize_trajectory_2d_2(poses_list,result_pose,path_name="imu trajectory with no update",path_name1="imu trajectory with VSLAM",show_ori=True)
#        plt.scatter(mu_t[0,:],mu_t[1,:],label="prior with raw camera data")
        
#        plt.scatter(mu_t_feature[0,:],mu_t_feature[1,:],label="updated feature with VSLAM")
        plt.legend()
        plt.show()   
        return result_pose
        
        
    
                


        
    
    
if __name__ == '__main__':
    myEKF = EKF()
	#IMU Localization via EKF Prediction
    mu_list = myEKF.prediction()
	#Landmark Mapping via EKF Update
    myEKF.update(mu_list)
	#Visual-Inertial SLAM
    result_pose = myEKF.vslam(mu_list)
	# You can use the function below to visualize the robot pose over time
	#visualize_trajectory_2d(world_T_imu,show_ori=True)
