import numpy as np
import sys
import os
sys.path.append(os.path.abspath('wethebestOLS'))
import ols
import matplotlib.pyplot as plt
from linearmodels.iv import IV2SLS

beta_draw  = np.random.exponential(4)
nMarkets = 100

def demand_side(P, α, β , Ed):

    Qd = α + P*β + Ed

    return Qd

def supply_side(P, γ, ψ, Es):

    Qs = γ + P*ψ + Es

    return Qs

def instrument(Es, η):

    Z = Es + η

    return Z


#Run optimization:

nsim = 1000

for s in range(nsim):

    P = np.random.random((nMarkets, 1))
    mu = 0
    sigma = 1
    β  = -np.random.exponential(2)
    ψ  = np.random.exponential(2)
    α  = 0.3
    γ  = 0.5

    Ed  = np.random.normal(mu, sigma, nMarkets).reshape(nMarkets,1)
    Es  = np.random.normal(mu, sigma, nMarkets).reshape(nMarkets,1)

    diff = 5

    while diff > 0.00001:

        Qd = demand_side(P, 0.3, β , Ed)
        Qs = supply_side(P, 0.5, ψ , Es)

        diff = np.sum(abs(Qd-Qs))
        P = P + 0.1*(Qd - Qs)


    XX = np.ones((nMarkets,2))
    XX[:,1] = P
    beta, se, vcv = ols.ols(Qd, XX)




plt.scatter(P, Ed) #Relationship between prices and demand side shocks
plt.scatter(P, Es) #Relationship between prices and suply side shocks
