import numpy as np
import sys
import os
sys.path.append(os.path.abspath('wethebestOLS'))
import ols
import matplotlib.pyplot as plt


nsim = 1000
nObs = 1000
nParams = 3
const = np.ones((nObs,1))
XX = np.random.random((nObs, nParams))
XX[:,0] = 1

beta_list = []
sigma_list = []
mu_list = []
proportion_list = []

for s in range(nsim):

    mu_draw    = 0
    sigma_draw = 1
    beta_draw  = np.random.random((nParams,1))

    #Generate different Ys
    E    = np.random.normal(mu_draw, sigma_draw, nObs).reshape(nObs,1)
    Y = (XX@beta_draw).reshape(nObs,1) + E

    #Estimate betas and sigmas:
    beta, se, vcv = ols.ols(Y, XX)

    beta_hat_draw  = np.random.multivariate_normal(beta.reshape(nParams), vcv, (nsim,1))

    proportion = np.mean(beta_draw.reshape(1,nParams)>beta_hat_draw.reshape(nsim,nParams),0)

    proportion_list.append(proportion)

stuff = np.array(proportion_list)



plt.hist(stuff[:,0])
