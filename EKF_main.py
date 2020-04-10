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
        #print(self.features)
        
        
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
        noise_w = 1e-7 * np.eye(6)               
        
        for i in range(self.t_len-1):
            
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
        plt.title("Prediction only state")
        plt.show()   
        return mu_list, cov_list
    
    
    def update(self,mu_list):
        poses_list = []
        for i in range(self.t_len-1): 
            poses_list.append(np.linalg.inv(mu_list[i]))
            
        cov_t = np.zeros((self.feature_num,3,3))
        cov_t[:] = np.identity(3)

        M = getM(self.K,self.b)
        noise_v = 1e2 * np.eye(4)    
             
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
                        Z = getz(feature, M) 
                        temp = Z * np.linalg.pinv(M) @ feature
                        imu_T_cam = np.linalg.pinv(self.cam_T_imu)
                        world_T_imu = np.linalg.pinv(mu_list[i])
                        mu_temp = world_T_imu @ imu_T_cam @ temp
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
                    
                    z = pic[:,j]
                    cam_T_imu_T_world = self.cam_T_imu @ mu_list[i]
                    q = cam_T_imu_T_world @ mu_update[:,j]
                    pi_function = q/z
                    dpi_function = jacobian(q)
                    
                    H = M @ dpi_function @ cam_T_imu_T_world @ self.D
                    HT = np.transpose(H)
                    z_hat = M @ pi_function
                    k_gain = cov_t[j] @ HT @ np.linalg.pinv(H@cov_t[j]@HT + noise_v)
                
                    correction = z - z_hat

                    mu_update[:,j] = mu_update[:,j] + self.D @ k_gain @ correction
                    cov_t[j] = (np.identity(3)- k_gain @ H) @ cov_t[j]
        
        #visualize_trajectory_2d(poses_list,path_name="imu trajectory",show_ori=True)
        plt.scatter(mu_update[0,:],mu_update[1,:],label="updated with imu localization")
        plt.legend()
        plt.show()
        
    def VISLAM(self):
        
        time = np.transpose(self.t)
        velocity = np.transpose(self.linear_velocity)
        omega = np.transpose(self.rotational_velocity)
        
        #EKF prediction of the IMU 
        mu_list, cov_list = self.prediction()           #world in imu prediction (imu state)
        noise_w = 1e-5 * np.eye(6)                       #imu noise covariance 
        
        #Initialize EKF update
        M = getM(self.K,self.b)
        mu_world_feature_0 = np.zeros((4,self.feature_num)) #initial state of the landmark --> visualization 
        mu = np.zeros((4, self.feature_num+4))          #fused state: 4 x num of features, 4 x 4 for pose 
        cov = np.eye(3*self.feature_num + 6)                    #fused observation covariance 3 x num of features + 6 DOF 
        noise_v = 1e1 * np.eye(4)                       #landmarks noise covariance 
                          
        #Save the pose world_T_imu
        pose_list = []
        world_T_imu = np.linalg.inv(mu_list[0])
        pose_list.append(world_T_imu)
        
        #Update landmarks for first observation (first frame)
        indices = np.flatnonzero(self.features[0, :, 0] != -1) 
        

        feature = self.features[:, indices, 0]                   #feature in image frame 
        Z = getz(feature, M) 
        x = (feature[0] - M[0,2]) * Z / M[0,0]
        y = (feature[1] - M[1,2]) * Z / M[1,1]
        feature_cam = np.vstack((x, y, Z, np.ones(x.shape)))     #feature in camera frame 
        imu_T_cam = np.linalg.inv(self.cam_T_imu)
        world_T_imu = np.linalg.inv(mu_list[0])
        feature_world = world_T_imu @ imu_T_cam @ feature_cam    #feature in world frame 

        mu_world_feature_0[:, indices] = feature_world
        mu[:, indices] = feature_world
        
        for i in range(self.t_len-2):
            #EKF IMU pose prediction 
            dt = time[i+1]-time[i]
            omega_hat = gethatmap(omega[i])
            u_head = getu_head(omega_hat, velocity[i])

            v_hat = gethatmap(velocity[i])
            u_vee = getu_vee(omega_hat, v_hat)
            
            mu_list[i+1] = getpredict_mu(dt, u_head, mu_list[i]) #world to body 
            cov_list[i+1] = getpredict_cov(dt, u_vee, cov_list[i], noise_w)
            
            #Update covariance in fused state 
            cov[-6:, -6:] = cov_list[i]
            cov[:3*self.feature_num, 3*self.feature_num:] = cov[:3*self.feature_num, 3*self.feature_num:] @ np.transpose(expm(-dt*u_vee))
            cov[3*self.feature_num:, :3*self.feature_num] = np.transpose(cov[:3*self.feature_num, 3*self.feature_num:])
            
            #EKF mean update landmarks and pose 
            indices = np.flatnonzero(self.features[0, :, i+1] != -1)
            z = self.features[:, indices, i+1]
            N_t = len(indices)

            if N_t: 
                #First observed new features 
                indices_new = indices[np.flatnonzero(mu_world_feature_0[-1, indices] == 0)]
                z_new = self.features[:, indices_new, i+1]                     #new feature in image frame
                
                if len(z_new):
                    Z = getz(z_new, M) 
                    x = (z_new[0] - M[0,2]) * Z / M[0,0]
                    y = (z_new[1] - M[1,2]) * Z / M[1,1]
                    feature_cam = np.vstack((x, y, Z, np.ones(x.shape)))     #new feature in camera frame 
                    imu_T_cam = np.linalg.inv(self.cam_T_imu)
                    world_T_imu = np.linalg.inv(mu_list[i+1])
                    feature_world = world_T_imu @ imu_T_cam @ feature_cam    #new feature in world frame 
                    mu_world_feature_0[:, indices_new] = feature_world
                    mu[:, indices_new] = feature_world
                
                #EKF update landmarks and pose 
                H = np.zeros((4*N_t, 3*self.feature_num + 6))
                
                cam_T_imu_T_world = self.cam_T_imu @ mu_list[i+1]
                z_hat = M @ cam_T_imu_T_world @ mu[:,indices]
                z_hat = z_hat/z_hat[2]
                
                for j in range(N_t):
                    #landmarks update of H 
                    H[4*j:4*j+4, 3*indices[j]:3*indices[j]+3] = M @ jacobian(self.cam_T_imu @ mu_list[i+1] @ mu[:,indices[j]].reshape(4,-1)) @ \
                                                                 self.cam_T_imu @ mu_list[i+1] @ self.D
                    #imu pose update of H 
                    H[4*j:4*j+4, -6:] = M @ jacobian(self.cam_T_imu @ mu_list[i+1] @ mu[:, indices[j]]) @ \
                                            self.cam_T_imu @ getcircle(mu_list[i+1] @ mu[:, indices[j]].reshape(4, -1))
                k_gain = cov @ np.transpose(H) @ np.linalg.pinv(H @ cov @ np.transpose(H) + np.kron(np.eye(N_t), noise_v))

                
                #Update landmarks mean 
                mu[:, :self.feature_num] = (mu[:, :self.feature_num].flatten('F') + np.kron(np.eye(self.feature_num), self.D) @ k_gain[:3*self.feature_num, :] @ \
                                           (z - z_hat).flatten('F')).reshape(self.feature_num, 4).T
                
                #Update IMU pose mean 
                mu[:, self.feature_num:] = expm(hatoperation(k_gain[-6:, :] @ (z - z_hat).flatten('F'))) @ mu_list[i+1]
                
                #Update fused state covariance 
                cov = (np.eye(3*self.feature_num + 6) - k_gain @ H) @ cov
                
                #Update all
                mu_list[i+1] = mu[:, self.feature_num:]
                pose_list.append(np.linalg.inv(mu_list[i+1]))
                cov_list[i+1] = cov[-6:, -6:]
             
        feature_list = mu[:, :self.feature_num]
        poses_list = np.stack((pose_list), axis = -1)
        plt.scatter(feature_list[0,:],feature_list[1,:],label="updated feature with VISLAM")
        visualize_trajectory_2d(poses_list,path_name="VISLAM",show_ori=True)                               
        return poses_list
    

            
            
    
if __name__ == '__main__':
    myEKF = EKF()
    #IMU Localization via EKF Prediction (Prediction only)
    mu_list, cov_list = myEKF.prediction()
	
    #Landmark Mapping via EKF Update
    myEKF.update(mu_list)
    
	#Visual-Inertial SLAM
    poses_list = myEKF.VISLAM()
	# You can use the function below to visualize the robot pose over time
    #visualize_trajectory_2d(poses_list,show_ori=True)
