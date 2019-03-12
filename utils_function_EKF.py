#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 22:07:18 2019

@author: LuLienHsi
"""
import numpy as np
from scipy.linalg import expm

def gethatmap(omega):
    hat = [[0, -omega[2], omega[1]],[omega[2],0,-omega[0]],[-omega[1],omega[0],0]]
    return hat

def getu_head(hat,velocity):
    u_head = np.zeros((4,4))
    row1 = np.append(hat[0],velocity[0])
    row2 = np.append(hat[1],velocity[1])
    row3 = np.append(hat[2],velocity[2])
    row4 = np.zeros(4)
    u_head[0] = row1
    u_head[1] = row2
    u_head[2] = row3
    u_head[3] = row4
    return u_head

def getpredict_mu(dt,u_head,mu_t_t):
    output = np.dot(expm(-dt*u_head),mu_t_t)
    return output

def getu_vee(omega_hat,v_hat):
    u_vee = np.zeros((6,6))
    row1 = np.append(omega_hat[0],v_hat[0])
    row2 = np.append(omega_hat[1],v_hat[1])
    row3 = np.append(omega_hat[2],v_hat[2])
    row4 = np.append([0,0,0],omega_hat[0])
    row5 = np.append([0,0,0],omega_hat[1])
    row6 = np.append([0,0,0],omega_hat[2])
    u_vee[0] = row1
    u_vee[1] = row2
    u_vee[2] = row3
    u_vee[3] = row4
    u_vee[4] = row5
    u_vee[5] = row6
    return u_vee

def getpredict_cov(dt,u_vee,cov_t_t,noise_w): 
    output = np.dot(np.dot(expm(-dt*u_vee),cov_t_t),expm(-dt*u_vee)) + (dt**2)*noise_w*np.identity(6)
    return output 

def getM(k,b):
    M = np.zeros((4,4))
    fsu = k[0,0]
    row1 = np.append(k[0],[0])
    row2 = np.append(k[1],[0])
    row3 = np.append(k[0],[-fsu*b])
    row4 = np.append(k[1],[0])
    M[0] = row1
    M[1] = row2
    M[2] = row3 
    M[3] = row4
    return M

def getz(feature,M):
    output =  (-M[2,3])/ (feature[0] - feature[2]) 
    return output

def jacobian(function):
    matrix = np.array([[1,0,-(function[0]/function[2]),0],[0,1,-(function[1]/function[2]),0],[0,0,0,0],[0,0,-(function[3]/function[2]),1]])
    output = matrix / function[2]
    return output

def getcircle(m):
    output = [[1,0,0,0,m[2],-m[1]],[0,1,0,-m[2],0,m[0]],[0,0,1,m[1],-m[0],0],[0,0,0,0,0,0]]
    return output


def hatoperation(term): 
    x = term[0]
    y = term[1]
    z = term[2]
    theta_x = term[3]
    theta_y = term[4]
    theta_z = term[5]
    omega_hat = gethatmap([theta_x,theta_y,theta_z])
    output = getu_head(omega_hat,[x,y,z])
    return output
    


