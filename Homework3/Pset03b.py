#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 00:07:20 2018

@author: johndoe

"""

import numpy as np
import matplotlib as mpl

def plot_results_ukf(state,output,command,state_true,title=''):
    '''
    Plots the true and predicted state histories, as well as the trajectory and
    the control history.
    '''
    
    color0 = 'C0'
    color1 = 'C1'
    linestyle0 = 'solid'
    linestyle1 = 'dashed'
    # Plot the trajectory
    dims = (5,4)
    mpl.pyplot.figure(np.random.randint(low=0,high=100))
    mpl.pyplot.title(title)
    mpl.pyplot.subplot2grid(dims,(0,0),rowspan=5,colspan=2)
    mpl.pyplot.plot(state_true[0,:],state_true[1,:],color=color0,linestyle=linestyle0)
    mpl.pyplot.plot(state[0,:],state[1,:],color=color0,linestyle=linestyle1)
    mpl.pyplot.legend(['True State','Estimated State'])
    
    color_border = 'C3'
    linestyle_border = 'dotted'
    mpl.pyplot.plot([-250E-3,250E-3],[375E-3,375E-3],color=color_border,linestyle=linestyle_border)
    mpl.pyplot.plot([250E-3,250E-3],[375E-3,-375E-3],color=color_border,linestyle=linestyle_border)
    mpl.pyplot.plot([-250E-3,250E-3],[-375E-3,-375E-3],color=color_border,linestyle=linestyle_border)
    mpl.pyplot.plot([-250E-3,-250E-3],[-375E-3,375E-3],color=color_border,linestyle=linestyle_border)
    
    xmin = np.min([np.min(state[0,:]),-250E-3])
    xmax = np.max([np.max(state[0,:]),250E-3])
    ymin = np.min([np.min(state[1,:]),-375E-3])
    ymax = np.max([np.max(state[1,:]),375E-3])
    k = 1.2
    mpl.pyplot.xlim(k*xmin,k*xmax)
    mpl.pyplot.ylim(k*ymin,k*ymax)
    
    # Plot the translation state history
    xvals = dt*np.arange(0,np.shape(state)[1])
    mpl.pyplot.subplot2grid(dims,(0,2),colspan=2)
    mpl.pyplot.plot(xvals,state_true[0,:],color=color0,linestyle=linestyle0)
    mpl.pyplot.plot(xvals,state_true[1,:],color=color1,linestyle=linestyle0)
    mpl.pyplot.plot(xvals,state[0,:],color=color0,linestyle=linestyle1)
    mpl.pyplot.plot(xvals,state[1,:],color=color1,linestyle=linestyle1)
    mpl.pyplot.legend(['True x','True y','Est x','Est - y'])
    
    # Plot the translation output history
    mpl.pyplot.subplot2grid(dims,(1,2),colspan=2)
    mpl.pyplot.plot(xvals,output[0,:])
    mpl.pyplot.plot(xvals,output[1,:])
    mpl.pyplot.legend(['output - x','output - y'])
    
    # Plot the rotation state history
    k = 0.1
    mpl.pyplot.subplot2grid(dims,(2,2),colspan=2)
    mpl.pyplot.plot(xvals,state_true[2,:],color=color0,linestyle=linestyle0)
    mpl.pyplot.plot(xvals,state[2,:],color=color0,linestyle=linestyle1)
    ymin = np.min(state[2,:])
    ymax = np.max(state[2,:])
    mpl.pyplot.ylim([ymin - k,ymax + k])
    mpl.pyplot.ticklabel_format(axis='y',useOffset=False)
    mpl.pyplot.legend(['True Heading','Est Heading'])
    
    # Plot the rotation output history
    mpl.pyplot.subplot2grid(dims,(3,2),colspan=2)
    mpl.pyplot.plot(xvals,output[2,:])
    ymin = np.min(output[2,:])
    ymax = np.max(output[2,:])
    mpl.pyplot.ylim([ymin - k,ymax + k])
    mpl.pyplot.ticklabel_format(axis='y',useOffset=False)
    mpl.pyplot.legend(['Output - Heading'])
    
    # Plot the control history
    xvals = np.arange(0,np.shape(command)[1])
    mpl.pyplot.subplot2grid(dims,(4,2),colspan=2)
    mpl.pyplot.plot(xvals,command[0,:])
    mpl.pyplot.plot(xvals,command[1,:])
    mpl.pyplot.legend(['phi_dot_left','phi_dot_right'])
    
    mpl.pyplot.show()
    
def compose_A():
    '''
    A matrix from state update
    '''
    
    a0 = np.array([1,0,0])
    a1 = np.array([0,1,0])
    a2 = np.array([0,0,1])

    A = np.vstack((a0,a1,a2))
    
    return A

def compose_B(s,u):
    '''
    Calculates and returns the B matrix in the state update
    '''
    
    accel_l = u[0][0]
    accel_r = u[1][0]
    theta = s[2,0]
    
    gamma = calc_gamma(accel_l,accel_r)
    
    b0 = (m_w/m_c) * np.array([np.cos(theta),np.cos(theta)])
    b1 = (m_w/m_c) * np.array([np.sin(theta),np.sin(theta)])
    b2 = (m_w/I_c) * np.array([[gamma,(gamma-2*delta)]])

    B = (dt * r * m_w) * np.vstack((b0,b1,b2))
    
    return B

def compose_C(k0,k1,k2):
    '''
    C matrix for output calculation
    '''
    
    c0 = np.array([k0,0,0])
    c1 = np.array([0,k1,0])
    c2 = np.array([0,0,k2])
    
    C = np.vstack((c0,c1,c2))
    
    return C

def cov_mat(s00=0,s01=0,s02=0,s10=0,s11=0,s12=0,s20=0,s21=0,s22=0):
    '''
    Creates a 3x3 covariance matrix based on entries for each element.
    '''
    
    s0 = np.array([s00,s01,s02])
    s1 = np.array([s10,s11,s12])
    s2 = np.array([s20,s21,s22])
    
    S = np.vstack((s0,s1,s2))
    
    return S

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

def ukf_init2(distr_state,distr_environ,distr_sensor):
    '''
    Initialize the state estimator
    '''
    
    mean_state = distr_state[0]
    cov_state = distr_state[1]
    mean_environ = distr_environ[0]
    cov_environ = distr_environ[1]
    mean_sensor = distr_sensor[0]
    cov_sensor = distr_sensor[1]
        
    return mean_state,mean_environ,mean_sensor,cov_state,cov_environ,cov_sensor

def ukf_calc_X(state_pred,P,L,alpha,k,beta):
    '''
    Used above within fxn ukf_init() to calculate the sigma vectors
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
    '''
    Calculate weights for update in the mean and covariance
    '''
    
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
    '''
    Calculate both the sigma vectors and weights
    '''
    
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
    Perform the time update
    j - time during previous action
    k - time during current action
    '''
    
    L = np.shape(Xx_j)[0]
    
    Xx_k_given_j = F(A,Xx_j,B,u,Xw_j)
    s_k_minus = np.sum( [Wm[i,0] * Xx_k_given_j[:,i:i+1] for i in range(2*L+1)] ,axis=0)
    
    Y_k_given_j = measurement_model(Xx_k_given_j) + Xv_j #H(C,Xx_k_given_j,Xv_j)
    y_k_minus = np.sum( [Wm[i,0] * Y_k_given_j[:,i:i+1] for i in range(2*L+1)] ,axis=0)
    
    Z = (Xx_k_given_j - s_k_minus)
    P_k_minus = np.sum( [Wc[i,0] * (Z @ Z.T) for i in range(2*L+1)] ,axis=0)
    
    return P_k_minus,s_k_minus,y_k_minus,Y_k_given_j,Xx_k_given_j

def ukf_msmt_update(H,C,Wc,s_k_minus,y_k_minus,P_k_minus,Y_k_given_j,Xx_k_given_j,v):
    '''
    Make the UKF measurement update
    
    Inputs:
        Wc:             Weights for sigma vectors' covariances
        s_k_minus:      State estimate prior to measurement update
        y_k_minus:      Output estimate prior to measurement update
        P_k_minus:      State covariance prior to measurement update
        Y_k_given_j:    New output update
        Xx_k_given_j:   New sigma vector covariance
        v:              Sensor noise vector
        
    Outputs:
        s_k:    Updated state estimate
        y_k:    Updated output estimate
        P_k:    Updated covariance
        
    '''
    L = np.shape(Xx_k_given_j)[0]
    
    A = Y_k_given_j - y_k_minus
    B = Xx_k_given_j - s_k_minus
    P_yy = np.sum( [[Wc[i,0]] * A @ A.T for i in range(2*L+1)] ,axis=0)
    P_xy = np.sum( [[Wc[i,0]] * B @ A.T for i in range(2*L+1)] ,axis=0)
    
    K = P_xy @ np.linalg.pinv(P_yy)
    
    y_k = measurement_model(s_k_minus) + v
    
    s_k = s_k_minus + K @ (y_k - y_k_minus)
    
    P_k = P_k_minus - K @ P_yy @ K.T
    
    return s_k, y_k, P_k

def measurement_model(state_vector,height=750E-3,width=500E-3):
    '''
    Takes a state (or a row vector of states) and computs the corresponding 
    sensor measurement.
    
    Inputs:
        state_vector:   [x,y,theta] robot state vector
        height:         height of arena
        width:          width or arena
    
    Outputs:
        output:         output measurement made by sensors
        
    '''
    N_vectors = np.shape(state_vector)[1]
    output = np.zeros([3,N_vectors])
    for n in range(N_vectors):
        
        state = state_vector[:,n]
        x = state[0]
        y = state[1]
        theta = state[2]
        
        # Angles for each body fixed axis
        theta_x = theta
        theta_y = theta_x + np.pi/2
        
        # Measurement for laser range finder along x and y body axes
        dist_x_body = get_intercept(x,y,theta_x,height,width)
        dist_y_body = get_intercept(x,y,theta_y,height,width)

        
        output[:,n] = np.array([dist_x_body, dist_y_body, theta_x])
    
    return output

def get_intercept(x,y,theta,h,w,thresh=1000):
    '''
    Find the 2-dimensional spatial intersection coordinates of the line from
    the tip of a body axis vector to a wall.
    
    Inputs:
        x:      current robot x position
        y:      current robot y position
        theta:  current robot heading
        h:      height of arena
        w:      width of arena
        thresh: threshold to determine if tangent function encounters division by zero
    
    Outputs:
        dist:   distance from robot to wall measured along a body axis
    '''
    # Dimensions of the 4 quadrants    #_________
    a = h/2 + y                        #|   |   | b  |
    b = h/2 - y                        #|---|---|    h
    c = w/2 + x                        #|   |   | a  |
    d = w/2 - x                        #¯¯¯¯¯¯¯¯¯
                                       #  c   d
                                       #----w----
    while theta < 0 :
        theta += 2*np.pi
    
    while theta > 2*np.pi:
        theta -= 2*np.pi
    
    msg = ''.join([str(theta)])
    assert 0 <= theta <= 2*np.pi, msg
    
    theta_wall_R = [np.arctan2(-a,d) + 2*np.pi,  np.arctan2(b,d)]
    theta_wall_T = [np.arctan2(b,d),             np.arctan2(b,-c)]
    theta_wall_L = [np.arctan2(b,-c),            np.arctan2(-a,-c) + 2*np.pi]
    theta_wall_B = [np.arctan2(-a,-c) + 2*np.pi, np.arctan2(-a,d) + 2*np.pi]
    
    g_r = [w/2,np.inf]
    g_t = [np.inf,h/2]
    g_l = [-w/2,np.inf]
    g_b = [np.inf,-h/2]
    
    # Only tangent explosion must be worried about for left and right walls 
    # because the 0 case generalizes and still intercepts the Y axis.
    
    # Both the explosion and 0 cases must be considered for top and bottom due
    # to those regions' containing the y axis and the special case when the axis
    # is almost parallel to the x axis and does not intercept it.
    
    tan_theta = np.tan(theta)
    if tan_theta < 0:
        tan_theta += 2*np.pi
    if theta_wall_R[0] < theta or theta <= theta_wall_R[1]:
        
        X = g_r[0]          # Constrain X because we know it is along the wall.
        
        # If axis points along y axis, Y = y so (Y-y)^2 = 0
        if tan_theta > thresh:
            Y = y
        else:
            Y = X*tan_theta - x*tan_theta + y
        
    elif theta_wall_T[0] < theta <= theta_wall_T[1]:
        
        Y = g_t[1]          # Constrain Y because we know it is along the wall.
        
        # If axis points along y axis, X = x so (X-x)^2 = 0
        if tan_theta > thresh:
            X = x
        
        # If tangent is almost zero, Y = y and X is assumed to be at either L/R
        # wall depending on theta
        elif tan_theta < 1/thresh:
            Y = y
            if theta < np.pi/2:
                X = d
            else:
                X = c
        else:
            X = (1/tan_theta) * (Y + x*tan_theta - y)
    
    elif theta_wall_L[0] < theta <= theta_wall_L[1]:
        
        X = g_l[0]
        if tan_theta > thresh:
            Y = y
        else:
            Y = X*tan_theta - x*tan_theta + y
    
    elif theta_wall_B[0] < theta <= theta_wall_B[1]:
        Y = g_b[1]
        
        # If axis points along y axis, X = x so (X-x)^2 = 0
        if tan_theta > thresh:
            X = x
        
        # If tangent is almost zero, Y = y and X is assumed to be at either L/R
        # wall depending on theta
        elif tan_theta < 1/thresh:
            Y = y
            if theta < 3*np.pi/2:
                X = c
            else:
                X = d
        else:
            X = (1/tan_theta) * (Y + x*tan_theta - y)
    
    dist = np.linalg.norm([(X-x),(Y-y)])
    return dist

def F(A,s,B,u,w):
    '''
    State equation
    '''
    return A @ s + B @ u + w

def H(C,s,v):
    '''
    Measurement equation
    '''
    return C @ s + v

def plot_center_of_rotation():
    '''
    Plot center of rotation as a function of wheel speed ratio
    '''
    xvalues = np.arange(0,100,0.1)
    y = 2*delta*(1 - 1/(xvalues+1))
    mpl.pyplot.plot([xvalues[0],xvalues[-1]],[2*delta,2*delta],color='grey',linestyle='--')
    mpl.pyplot.plot(xvalues,y)
    mpl.pyplot.xlabel('Force ratio Fl/Fr')
    mpl.pyplot.ylabel('Center of Rotation Location')
    mpl.pyplot.title('Center of Rotation/Moment Position Relative to Left Wheel')
    mpl.pyplot.legend(['2 * delta'])
    mpl.pyplot.savefig('CenterOfRotation.png')

def compose_w(Q):
    samples = np.random.multivariate_normal([0,0,0],Q,size=1)
    
    return samples.T

def compose_v(V):
    samples = compose_w(V)
    return samples.T

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

# Generate control inputs
d_omega = 1/441
vals = np.arange(0,8,d_omega)
phi_dot_l = np.array([i for i in vals])
phi_dot_r = np.array([4.75 for i in range(len(vals))])

# Initialize histories
u_hist = np.array([phi_dot_l,phi_dot_r])
s_hist = np.zeros([3,np.shape(u_hist)[1]+1])
y_hist = np.zeros([3,np.shape(u_hist)[1]+1])
s_hist_ukf = np.zeros([3,np.shape(u_hist)[1]+1])
y_hist_ukf = np.zeros([3,np.shape(u_hist)[1]+1])

  # rad/sec/volt # motor control gain

x_start,y_start,theta_start = 0,0,np.pi/4
s0 = np.array([[x_start,y_start,theta_start]]).T
y0 = 1*s0
u0 = np.array([[u_hist[0][0],u_hist[1][0]]]).T

A = compose_A()

B = compose_B(s0,u0)

k0 = 1
k1 = 1
k2 = 1
C = compose_C(k0,k1,k2)

# Insert 0th entry in histories
u_hist[:,0:1] = u0
s_hist[:,0:1] = s0
y_hist[:,0:1] = y0
s_hist_ukf[:,0:1] = s0
y_hist_ukf[:,0:1] = y0

# Covariance parameters for sensors and wheel slippage
sig_x = 0.075**2    # 3-12% specified in datasheet
sig_y = 0.075**2    # 3-12% specified in datasheet
sig_slip = 0.05**2  # 5% slippage specified in pset
sig_imu = 0.001**2  # 0.1 RMS specified in datasheet

a = 1
mean_state = np.array([[x_start],[y_start],[theta_start]])
cov_state = a*cov_mat(s00=0,s11=0,s22=0)

b = 0.0001
mean_environ = np.array([[0],[0],[0]])
cov_environ = b*cov_mat(s00=sig_slip,s11=sig_slip,s22=0)

c = 0.000001
mean_sensor = np.array([[0],[0],[0]])
cov_sensor = c*cov_mat(s00=sig_x,s11=sig_y,s22=sig_imu)

distr_state = [mean_state,cov_state]
distr_environ = [mean_environ,cov_environ]
distr_sensor = [mean_sensor,cov_sensor]

covariance_hist = [np.linalg.det(cov_state)]

sx_j_pred,sw_j_pred,sv_j_pred,Px,Pw,Pv = ukf_init2(distr_state,distr_environ,distr_sensor)
for n in range(np.shape(u_hist)[1]):
    u_k = u_hist[:,n:(n+1)]
    
    Xx_j,Xw_j,Xv_j,Wm,Wc = ukf_calc_sigma_n_weights(sx_j_pred,sw_j_pred,sv_j_pred,Px,Pw,Pv,alpha=1E-3,k=0,beta=2)

    # Variables at time j go in; variables at time k come out.
    # y_k_minus is state covariance matrix prior to measrement update
    # y_k_minus is output vector prior to measurement update
    P_k_minus,s_k_minus,y_k_minus,Y_k_given_j,Xx_k_given_j = ukf_time_update(F,H,A,B,u_k,Xx_j,Xw_j,Xv_j,Wm,Wc)
    
    # s_k is updated state estimate after measurement
    # P_k is updated output covariance estimate after measurement
    s_k,y_k,P_k = ukf_msmt_update(H,C,Wc,s_k_minus,y_k_minus,P_k_minus,Y_k_given_j,Xx_k_given_j,sv_j_pred)
    
    covariance_hist.append(np.linalg.det(P_k))
    
    s_hist_ukf[:,(n+1):(n+2)] = s_k
    y_hist_ukf[:,(n+1):(n+2)] = y_k
    
    # Get variables for next iteration
    sx_j_pred = s_k
    sw_j_pred = compose_w(cov_environ)
    sv_j_pred = compose_w(cov_sensor)
    
    B = compose_B(sx_j_pred,u_k)
    
# Run scenario without noise and Kalman filter for truth data
for n in range(np.shape(u_hist)[1]):
    s = s_hist[:,n:(n+1)]
    u = u_hist[:,n:(n+1)]
    
    s = F(A,s,B,u,np.zeros([3,1]))
    y = measurement_model(s) + np.zeros([3,1])
    
    # Store s in state history
    s_hist[:,(n+1):(n+2)] = s
    y_hist[:,(n+1):(n+2)] = y
    
    # Update u
    u = u_hist[:,n:(n+1)]
    B = compose_B(s,u)

# Plot results
plot_results_ukf(s_hist_ukf,y_hist_ukf,u_hist,s_hist,title='Unscented Kalman Filter')