#!/usr/bin/env python3
# coding: utf-8

# # Test1: SINDy(PDE)
"""
    File name: SINDyPDE-WNI.py
    File source : https://github.com/luckystarufo/pySINDy    
        Author: created by Yuying Liu
        Date created: 11/30/18
    
    Edited: 28/2/2020 by Aaron Xavier
    Python Version: 3.6
"""


import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import time
import cv2

from pySINDy.sindypde import SINDyPDE
from pySINDy import SINDy
from pySINDy.sindybase import SINDyBase
from pySINDy.sindylibr import SINDyLibr
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


def plotUV(Un, Vn, du_dt, dv_dt):
    """
        Function for plotting the UV as a quiver plot
        Created by Aaron X on 27/2/2020
        :param Un: U vectors
        :param Vn: V vectors
        :param du_dt: Incremental changes in U after each iteration
        :param dv_dt: Incremental changes in V after each iteration"""

    ##--SAMPLING--##
    s=(84,56)
    U_ds=np.zeros(s)
    V_ds=np.zeros(s)
    d_u=np.zeros(s)
    d_v=np.zeros(s)
    for i in range(1,84):
        for j in range(1,56):
            U_ds[i,j]=Un[10*i,10*j]
            V_ds[i,j]=Vn[10*i,10*j]
            d_u[i,j]=np.reshape(du_dt,[841,561])[10*i,10*j]
            d_v[i,j]=np.reshape(dv_dt,[841,561])[10*i,10*j]

    fig, ax = plt.subplots(figsize=(56,84))

    x_pos = np.arange(0,56,1)
    y_pos = np.arange(0,84,1)


    ax.quiver(x_pos,y_pos, U_ds[:,:], V_ds[:,:], width=0.001)
    ax.set_title('Plotting motion vectors')
    #plt.show()
    U_ds+=10*d_u
    V_ds+=10*d_v
    fig, ax = plt.subplots(figsize=(56,84))

    x_pos = np.arange(0,56,1) 
    y_pos = np.arange(0,84,1)
    ax.quiver(x_pos,y_pos, U_ds[:,:], V_ds[:,:], width=0.0005)
    ax.set_title('Plotting motion vectors')
    filename="Outputs/Im%i.png"%+(time.time())
    plt.savefig(filename)

    """
    for i in range(0,10):
        U_ds+=10*d_u
        V_ds+=10*d_v
        fig, ax = plt.subplots(figsize=(56,84))

        x_pos = np.arange(0,56,1) 
        y_pos = np.arange(0,84,1)
        ax.quiver(x_pos,y_pos, U_ds[:,:], V_ds[:,:], width=0.0005)
        ax.set_title('Plotting motion vectors')
        filename="Outputs/Im%i.png"%+(int(i)+1)
        plt.savefig(filename)
    """  

def appendUV(Um,Vm,U_nxt,V_nxt):
    """
        Function for appending the new UVs with the prev data
        Created by Aaron X on 27/2/2020
        :param Um: prev U data
        :param Vm: prev V data
        :param U_nxt: Current U to be appended
        :param V_nxt: Current V to be appended
        :return U, V: updated data matrices of U and V respectively
    """
    #Init matrix sizes 
    dim=(841,561,11)
    Um=np.zeros(dim)
    Vm=np.zeros(dim)

    Um[:,:,0:10]=U
    Um[:,:,10]=U_nxt
    
    Vm[:,:,0:10]=V
    Vm[:,:,10]=U_nxt

    return Um[:,:,-10:], Vm[:,:,-10:]

def trainSINDy(U,V,dx,dy,dt):
    """
        Function for training SINDy 
        Created by Aaron X on 22/2/2020
        :param U: U data matrix
        :param V: V data matrix
        :param dx, dy, dt: spatio-temporal grid spacings
        :return U, V: returning U and V for use another func block
        :return model.coefficients: Coefficients generated using SINDy
        """
    
    model = SINDyPDE(name='SINDyPDE model for Reaction-Diffusion Eqn')
    
    start_train=time.time()
    #model.fit(self, data, poly_degree=2, cut_off=1e-3)
    model.fit({'u': U, 'v': V}, dt, [dx, dy], space_deriv_order=2, poly_degree=2, sample_rate=0.01, cut_off=0.05, deriv_acc=2)

    print("\n--- Train time %s seconds ---\n" %(time.time() - start_train))

    #print("\n--- Active terms ---\n" )
    size=np.shape(model.coefficients)
    cnt=0
    for i in range(size[0]):
        for j in range(size[1]):
            if (model.coefficients[i,j])!=0:
                #print(model.coefficients[i,j],"--",model.descriptions[i])
                cnt+=1
    print("--- Active terms %s ---\n" %cnt)
    return U, V, model.coefficients

def testSINDy(Us,Vs,dxs,dys,dts,coeff):
    """
        Function for testing SINDy 
        Created by Aaron X on 22/2/2020
        :param Us: U data matrix (test)
        :param Vs: V data matrix (test)
        :param dxs, dys, dts: spatio-temporal grid spacings
        :return U_nxt, V_nxt: Next U and V vectors
    """
    
    
    model2 = SINDyLibr(name='Derived module from sindybase.py for libr computation')
    libx=model2.libr({'u': Us, 'v': Vs}, dts, [dxs,dys], space_deriv_order=2, poly_degree=2, sample_rate=0.01, cut_off=0.5, deriv_acc=2)

    #Performing Lib*Coeff
    duv_dt=np.matmul(libx,coeff)

    #Splitting dU/dt and dV/dt
    du_dt=duv_dt[:,0]
    dv_dt=duv_dt[:,1]

    #Calc next frame as U_nxt=U+dU and V_nxt=V+dV
    U_nxt=np.reshape(Us,[841,561])+np.reshape(du_dt,[841,561])
    V_nxt=np.reshape(Vs,[841,561])+np.reshape(dv_dt,[841,561])

    plotUV(U_nxt,V_nxt,du_dt,dv_dt)

    return U_nxt, V_nxt


start_prog=time.time()

#Loading the preprocessed training data(20190726:1900hrs-2030hrs)
traindata = sio.loadmat('../datasets/20190726-UV.mat')
traindata.keys()

U = np.real(traindata['u'])
V = np.real(traindata['v'])
t = np.real(traindata['t'].flatten())
x = np.real(traindata['x'].flatten())
y = np.real(traindata['y'].flatten())
dt = t[1] - t[0]
dx = x[1] - x[0]
dy = y[1] - y[0]

#Loading the test data(20190726:2040hrs)
testdata = sio.loadmat('../datasets/20190726/20190726-UVTest.mat')
testdata.keys()

Us = np.real(testdata['u']).reshape(841,561,1)
Vs = np.real(testdata['v']).reshape(841,561,1)
ts = np.real(testdata['t'].flatten())
xs = np.real(testdata['x'].flatten())
ys = np.real(testdata['y'].flatten())
dts = ts[1] - ts[0]
dxs = xs[1] - xs[0]
dys = ys[1] - ys[0]

#Iterating the train-test sequence while moving the 10-frame window 
for i in range(10):
    print("\n--- Sequence %s ---\n" %i)
    U, V, coeff = trainSINDy(U,V,dx,dy,dt)
    U_nxt, V_nxt = testSINDy(Us,Vs,dxs,dys,dts, coeff)
    U,V = appendUV(U,V,U_nxt,V_nxt)

print("\n--- Exec time %s seconds ---\n" %(time.time() - start_prog))

