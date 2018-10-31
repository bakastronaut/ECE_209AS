#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 00:07:20 2018

@author: johndoe

TO DO:
When done debugging, convert phi as control variable to voltage.

"""

import numpy as np
import matplotlib as mpl

def plot_results(state,output,command,title=''):
    # Plot the trajectory
    fig = mpl.pyplot.figure(np.random.randint(low=0,high=100))
    mpl.pyplot.title(title)
    mpl.pyplot.subplot2grid((3,3),(0,0),rowspan=3)
    mpl.pyplot.plot(state[0,:],state[1,:])
    mpl.pyplot.plot(output[0,:],output[1,:],'--')
    
    color = 'tab:grey'
    linestyle = 'dashed'
    mpl.pyplot.plot([-250E-3,250E-3],[375E-3,375E-3],color=color,linestyle=linestyle)
    mpl.pyplot.plot([250E-3,250E-3],[375E-3,-375E-3],color=color,linestyle=linestyle)
    mpl.pyplot.plot([-250E-3,250E-3],[-375E-3,-375E-3],color=color,linestyle=linestyle)
    mpl.pyplot.plot([-250E-3,-250E-3],[-375E-3,375E-3],color=color,linestyle=linestyle)
    
    xmin = np.min([np.min(state[0,:]),-250E-3])
    xmax = np.max([np.max(state[0,:]),250E-3])
    ymin = np.min([np.min(state[1,:]),-375E-3])
    ymax = np.max([np.max(state[1,:]),375E-3])
    k = 1.2
    mpl.pyplot.xlim(k*xmin,k*xmax)
    mpl.pyplot.ylim(k*ymin,k*ymax)
    
    # Plot the translation state and output histories
    xvals = dt*np.arange(0,np.shape(state)[1])
    mpl.pyplot.subplot2grid((3,3),(0,1),colspan=2)
    mpl.pyplot.plot(xvals,state[0,:])
    mpl.pyplot.plot(xvals,state[1,:])
    mpl.pyplot.plot(xvals,output[0,:],'--')
    mpl.pyplot.plot(xvals,output[1,:],'--')
    mpl.pyplot.legend(['state - x','state - y','output - x','output - y'])
    
    # Plot the rotation state and output histories
    mpl.pyplot.subplot2grid((3,3),(1,1),colspan=2)
    mpl.pyplot.plot(xvals,state[2,:])
    mpl.pyplot.plot(xvals,output[2,:],'--')
    ymin = np.min([state[2,:],output[2,:]])
    ymax = np.max([state[2,:],output[2,:]])
    k = 0.1
    mpl.pyplot.ylim([ymin - k,ymax + k])
    mpl.pyplot.ticklabel_format(axis='y',useOffset=False)
    mpl.pyplot.legend(['State - Heading','Output - Heading'])
    
    # Plot the control history
    xvals = np.arange(0,np.shape(command)[1])
    mpl.pyplot.subplot2grid((3,3),(2,1),colspan=2)
    mpl.pyplot.plot(xvals,command[0,:])
    mpl.pyplot.plot(xvals,command[1,:])
    mpl.pyplot.legend(['phi_dot_left','phi_dot_right'])
    
    mpl.pyplot.show()
    
def compose_A():
    a0 = np.array([1,0,0])
    a1 = np.array([0,1,0])
    a2 = np.array([0,0,1])

    A = np.vstack((a0,a1,a2))
    
    return A

def compose_B(s,u):
    accel_l = u[0][0]
    accel_r = u[1][0]
    theta = s[2,0]
    
    gamma = calc_gamma(accel_l,accel_r)
    
    b0 = (1/m_c) * np.array([np.cos(theta),np.cos(theta)])
    b1 = (1/m_c) * np.array([np.sin(theta),np.sin(theta)])
    b2 = (1/I_c) * np.array([[gamma,(gamma-2*delta)]])

    B = (dt * r * m_w) * np.vstack((b0,b1,b2))
    
    return B

def compose_C(k0,k1,k2):
    c0 = np.array([k0,0,0])
    c1 = np.array([0,k1,0])
    c2 = np.array([0,0,k2])
    
    C = np.vstack((c0,c1,c2))
    
    return C

def compose_Q(q00=0,q01=0,q02=0,q10=0,q11=0,q12=0,q20=0,q21=0,q22=0):
    q0 = np.array([q00,q01,q02])
    q1 = np.array([q10,q11,q12])
    q2 = np.array([q20,q21,q22])
    
    Q = np.vstack((q0,q1,q2))
    
    return Q

def compose_V(v00=0,v01=0,v02=0,v10=0,v11=0,v12=0,v20=0,v21=0,v22=0):
    v0 = np.array([v00,v01,v02])
    v1 = np.array([v10,v11,v12])
    v2 = np.array([v20,v21,v22])
    
    V = np.vstack((v0,v1,v2))
    
    return V

def compose_w(Q):
    samples = np.random.multivariate_normal([0,0,0],Q,size=1)
    
    return samples.T

def compose_v(V):
    samples = compose_w(V)
    
    return samples.T

def calc_gamma(accel_l,accel_r):
    if accel_r == 0:
        if accel_l != 0:
            gamma = 2*delta
        elif accel_l == 0:
            gamma = delta
    else:
        F_ratio = abs(accel_l/accel_r)
        gamma = 2*delta*(1 - 1/(F_ratio + 1))
    
    return gamma

def omega(volts):
#    j0 = ((130*2*np.pi)/60 - (100*2*np.pi)/60)/(6-4.8)
    
    # Assuming linear relationship passing through origin.
    j0 = (130*2*np.pi/60)/6
    
    return j0*volts

def ukf_init(x,y,theta,E_x,E_y,E_theta,E_w=np.zeros([3,1]),E_v=np.zeros([3,1])):
    state0 = np.array([[x],[y],[theta]])
    state0_pred = np.array([[E_x],[E_y],[E_theta]])
    P0 = (state0 - state0_pred) @ (state0 - state0_pred).T
    
    state0_a = np.vstack((state0,E_w,E_v)) 
    state0_a_pred = np.vstack((state0_pred,E_w,E_v))    # Augmented initial state vector including noise
    
    P0_a = (state0_a - state0_a_pred) @ (state0_a - state0_a_pred).T    # Augmented initial covariance matrix
    
    assert np.sum(np.equal(P0,P0_a[0:3,0:3])) == 9, 'First block on diagonal should be P0'

#           Px 00 00
#   P0_a =  00 Pw 00
#           00 00 Pv

    P_x = P0_a[0:3,0:3]
    P_w = P0_a[3:6,3:6]
    P_v = P0_a[6:9,6:9]
    
    x_x = state0_a_pred[0:3]
    x_w = state0_a_pred[3:6]
    x_v = state0_a_pred[6:9]
    
    return x_x,x_w,x_v,P_x,P_w,P_v

def ukf_calc_X(state_pred,P,L,alpha,k,beta):
    '''
    Used above within fxn ukf_init()
    '''
    lamda = (alpha**2) * (L + k) - L
    
    N = 2*L + 1         # Number of columns (vectors) in sigma vector
    X = np.zeros([L,N])
    
    X[:,0:1] = state_pred
    for i in range(1,L+1):
        B = np.sqrt( (L+lamda) * P[i-1:i-1+1,:] )
        X[:,i:i+1] = state_pred + B.T
    
    for j in range(L+1,N+1):
        B = np.sqrt( (L+lamda) * P[j-L-1:j-L+1-1,:] )
        X[:,j:j+1] = state_pred - B.T
    
    return X

def ukf_calc_WcWm(L,lamda,alpha,k,beta):
    N = 2*L + 1         # Number of columns (vectors) in sigma vector
    Wm = np.zeros([N,1])
    Wc = np.zeros([N,1])
    for i in range(N):
        if i == 0:
            Wm[0,0] = lamda/(L+lamda)
            Wc[0,0] = lamda/(L+lamda) + ( 1 - alpha**2 + beta)
        
        else:
            Wm[i,0] = 1/( 2*(L+lamda) )
            Wc[i,0] = 1/( 2*(L+lamda) )
    
    k = 1
    assert 1-k <= lamda/(L+lamda) + 2*L * 1/(2*(L+lamda)) <= 1
    
    return Wm,Wc

def ukf_calc_sigma_n_weights(xx_j_pred,xw_j_pred,xv_j_pred,Px,Pw,Pv,alpha,k,beta):
    
    L = np.shape(xx_j_pred)[0]  # Number of rows in state vector
    
    Xx = ukf_calc_X(xx_j_pred,Px,L,alpha=alpha,k=k,beta=beta)
    Xw = ukf_calc_X(xw_j_pred,Pw,L,alpha=alpha,k=k,beta=beta)
    Xv = ukf_calc_X(xv_j_pred,Pv,L,alpha=alpha,k=k,beta=beta)
    
    assert len(xx_j_pred) == len(xw_j_pred) == len(xv_j_pred) == 3
    
    lamda = (alpha**2) * (L + k) - L
    Wm,Wc = ukf_calc_WcWm(L,lamda,alpha=alpha,k=k,beta=beta)
    
    return Xx,Xw,Xv,Wm,Wc

def ukf_time_update(F,H,A,B,u,Xx_j,Xw_j,Xv_j,Wm,Wc):
    '''
    j - time during previous action
    k - time during current action
    '''
    L = np.shape(Xx_j)[0]
    
    Xx_k_given_j = F(A,Xx_j,B,u,Xw_j)
    s_k_minus = np.sum( [Wm[i,0] * Xx_k_given_j[:,i:i+1] for i in range(2*L+1)] ,axis=0)
    
    Y_k_given_j = H(C,Xx_k_given_j,Xv_j)
    y_k_minus = np.sum( [Wm[i,0] * Y_k_given_j[:,i:i+1] for i in range(2*L+1)] ,axis=0)
    
    Z = (Xx_k_given_j - s_k_minus)
    P_k_minus = np.sum( [Wc[i,0] * (Z @ Z.T) for i in range(2*L+1)] ,axis=0)
    
    return P_k_minus,s_k_minus,y_k_minus,Y_k_given_j,Xx_k_given_j

def ukf_msmt_update(H,C,Wc,s_k_minus,y_k_minus,P_k_minus,Y_k_given_j,Xx_k_given_j,v):
    L = np.shape(Xx_k_given_j)[0]
    
    A = Y_k_given_j - y_k_minus
    B = Xx_k_given_j - s_k_minus
    P_yy = np.sum( [[Wc[i,0]] * A @ A.T for i in range(2*L+1)] ,axis=0)
    P_xy = np.sum( [[Wc[i,0]] * B @ A.T for i in range(2*L+1)] ,axis=0)
    
    K = P_xy @ np.linalg.pinv(P_yy)
    
    y_k = H(C,s_k_minus,v)
    
    s_k = s_k_minus + K @ (y_k - y_k_minus)
    
    P_k = P_k_minus - K @ P_yy @ K.T
    
    return s_k,y_k,P_k

def F(A,s,B,u,w):
    return A @ s + B @ u + w

def H(C,s,v):
    return C @ s + v

###############################################################################
###############################################################################
###############################################################################
###############################################################################
    
m_c = 0.3
m_w = 0.05
r = 20E-3       # 20 mm
delta = (85E-3)/2   # 42.5 mm
dt = 0.1
I_c = (1/12) * m_c * (2*delta)**2

phi_dot_l = np.concatenate((np.ones(1),np.ones(20),np.ones(100),0.5*np.ones(100),np.ones(400)),0)
phi_dot_r = np.concatenate((np.ones(1),-1*np.ones(20),np.ones(100),np.ones(100),np.ones(400)),0)

#phi_dot_l = np.concatenate((np.ones(1),np.ones(20)),0)
#phi_dot_r = np.concatenate((np.ones(1),np.ones(20)),0)

u_hist = np.vstack((phi_dot_l,phi_dot_r))
s_hist = np.zeros([3,np.shape(u_hist)[1]+1])
y_hist = np.zeros([3,np.shape(u_hist)[1]+1])

s_hist_ukf = np.zeros([3,np.shape(u_hist)[1]+1])
y_hist_ukf = np.zeros([3,np.shape(u_hist)[1]+1])

  # rad/sec/volt # motor control gain

x_start,y_start,theta_start = 0,0,np.pi/2
s0 = np.array([[x_start,y_start,theta_start]]).T
y0 = 1*s0
u0 = np.array([[u_hist[0][0],u_hist[1][0]]]).T

A = compose_A()

B = compose_B(s0,u0)

k0 = 1
k1 = 1
k2 = 1
C = compose_C(k0,k1,k2)

a = 0.000000
Q = a*compose_Q(q00=1,q11=1,q22=1)

b = 0.000000
V = b*compose_V(v00=1,v11=1,v22=1)

u_hist[:,0:1] = u0

s_hist[:,0:1] = s0
y_hist[:,0:1] = y0

s_hist_ukf[:,0:1] = s0
y_hist_ukf[:,0:1] = y0

temp = []
for n in range(np.shape(u_hist)[1]):
    if n == 0:
        sx_j_pred,sw_j_pred,sv_j_pred,Px,Pw,Pv = ukf_init(x_start,y_start,theta_start,x_start,y_start,theta_start)
    
    else:
        u_k = u_hist[:,n:(n+1)]
        sx_j_pred = s_hist_ukf[:,n:(n+1)]
        sw_j_pred = compose_w(Q)
        sv_j_pred = compose_w(V)
        
        B = compose_B(sx_j_pred,u_k)
    
    Xx_j,Xw_j,Xv_j,Wm,Wc = ukf_calc_sigma_n_weights(sx_j_pred,sw_j_pred,sv_j_pred,Px,Pw,Pv,alpha=1E-1,k=0,beta=2)
    temp.append(Xx_j)

    # Variables at time j go in; variables at time k come out.
    # y_k_minus is state covariance matrix prior to measrement update
    # y_k_minus is output vector prior to measurement update
    P_k_minus,s_k_minus,y_k_minus,Y_k_given_j,Xx_k_given_j = ukf_time_update(F,H,A,B,u_k,Xx_j,Xw_j,Xv_j,Wm,Wc)
    
    # s_k is updated state estimate after measurement
    # P_k is updated output covariance estimate after measurement
    s_k,y_k,P_k = ukf_msmt_update(H,C,Wc,s_k_minus,y_k_minus,P_k_minus,Y_k_given_j,Xx_k_given_j,sv_j_pred)
    
    s_hist_ukf[:,(n+1):(n+2)] = s_k
    y_hist_ukf[:,(n+1):(n+2)] = y_k
    
for n in range(np.shape(u_hist)[1]):
    s = s_hist[:,n:(n+1)]
    u = u_hist[:,n:(n+1)]
    w = compose_w(Q)
    v = compose_w(V)
    
    s = F(A,s,B,u,w)
    y = H(C,s,v)
    
    # Store s in state history
    s_hist[:,(n+1):(n+2)] = s
    y_hist[:,(n+1):(n+2)] = y
    
    # Update u
    u = u_hist[:,n:(n+1)]
    B = compose_B(s,u)

plot_results(s_hist,y_hist,u_hist,'No State Estimation')
plot_results(s_hist_ukf,y_hist_ukf,u_hist,'Unscented Kalman Filter')