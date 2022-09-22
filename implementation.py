#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np 
import warnings
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt



dataset = pd.read_csv('EEG_Eye_State.csv',header = None)
X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13]].values[:-1]
y = dataset.iloc[:, [14]].values[:-1]
reg = LinearRegression().fit(X, y)
Beta = np.transpose(reg.coef_)
Intercept = reg.intercept_

p = 2
n = 30
Ip = 0.05
eigens = np.linalg.eig(np.dot(np.transpose(X),X))
k = np.arange(0,1.1,0.2)
sigma_cap_square = np.arange(0.25,0.45,0.05)



def design_matrix(X):
	'''S = X_transpose * X '''
	return np.dot(np.transpose(X),X)
S = design_matrix(X)


def ordinary_least_square_estimator(S,X,y):
	'''OLS'''
	return np.linalg.multi_dot([np.linalg.inv(S),np.transpose(X),y])


# OLS = ordinary_least_square_estimator(S,X,y)


def mean_squared_error_estimator(n,p,y,Beta,X):
	'''sigma_cap^2 = MSE(X*Beta + Error)'''
	return (np.dot(np.transpose(y),y) - np.linalg.multi_dot([np.transpose(Beta),np.transpose(X),y]))/(n-p)

sigma_cap_square = mean_squared_error_estimator(n,p,y,Beta,X)




'''----------------------------------------Ridge Regression Estimator-------------------------------------'''
	


def ortogonal_matrix(X,p,eigens):
	lambda_diagonal_values = []
	for i in range(p):
		lambda_diagonal_values.append(np.diag(eigens[i]))
	return lambda_diagonal_values

def alpha(ortogonal_matrix,Beta):
	return np.transpose(ortogonal_matrix) * Beta


def alpha_sum(p,alpha):
	alpha_sum = 0
	for i in range(p):
		alpha_sum += np.square(alpha[i])
	return alpha_sum



def beta_estimator(S,X,y):
	'''beta_cap = inv(S) * X_transpose * y'''

	return np.linalg.multi_dot([S,np.transpose(X),y])


def ridge_biasing_parameter_harmonic_mean(n,p,Beta,X,y,alpha_sum):
	'''k_harmonic_mean =  p*sigma_cap^2/Sum(alpha(i-th)^2)'''
	return (p*mean_squared_error_estimator(n,p,y,Beta,X))/(alpha_sum)

alpha = alpha(ortogonal_matrix(X,p,eigens)[0],Beta)
alpha_sum = alpha_sum(p,alpha)
k_harmonic_mean =  ridge_biasing_parameter_harmonic_mean(n,p,Beta,X,y,alpha_sum)

def w_variable(k,Ip,S):
	w = k * np.linalg.inv(S)
	return np.linalg.inv(w)

 


beta_cap = beta_estimator(S,X,y)
W = w_variable(k_harmonic_mean[0][0],Ip,S)

# W = w_variable(k_harmonic_mean[0][0],Ip,S)


def ridge_regression_estimator(W,beta_cap):
	'''RR(k_harmonic_mean) = W(k_harmonic_mean)*beta_cap'''

	return np.dot(W,beta_cap)

'''--------------------------------------------------------------------------------------------------------'''





'''----------------------------------------Liu Estimator---------------------------------------------------'''

def f_variable(S,Ip,d):
	'''F(d) = inv([S + Ip ]) * [S + d*Ip ]'''
	F = (np.linalg.inv(S + Ip)) * (S + d*Ip)
	return F

def alpha_cap_square(ortogonal_matrix,beta_cap):
	'''alpha_cap = QTranspose * Beta_cap'''
	return np.square(np.transpose(ortogonal_matrix) * beta_cap)

alpha_cap_square = alpha_cap_square(ortogonal_matrix(X,p,eigens)[0],beta_cap)



def liu_biasing_parameter(alpha_cap_square,sigma_cap_square,eigens):
	'''alpha_cap_square / (alpha_cap_square + mean_squared_error_estimator/eigens) '''
	'''d_alt'''
	return (alpha_cap_square/(sigma_cap_square/eigens + alpha_cap_square))



F = f_variable(S,Ip,liu_biasing_parameter(alpha_cap_square,sigma_cap_square,eigens[0]))
def liu_estimator(F,beta_cap):
	'''LE(d_alt) = F(d_alt) * beta_cap'''
	return np.dot(F,beta_cap)
'''--------------------------------------------------------------------------------------------------------'''






'''----------------------------------------One Parameter Estimator---------------------------------------------------'''


def m_variable(Ip,S,k):
	'''[Ip - kS -1]'''
	return Ip - k*S - 1
M = m_variable(Ip,S,k_harmonic_mean[0][0])
def new_one_paramter_estimator(W,M,beta_cap):
	'''The Kibria_lukan estimator'''
	'''Bkl = W(k) * M(K) * beta_cap'''
	return np.linalg.multi_dot([W,M,beta_cap])


'''--------------------------------------------------------------------------------------------------------'''

