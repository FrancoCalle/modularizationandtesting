import unittest
import numpy as np
import sys
import os
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
sys.path.append(os.path.abspath('..'))
from logit import optimization
from logit import objectiveFunction

class test_beta_logit( unittest.TestCase ) :

    def test_beta_logit(self):

        # Generate fake data
        nObs = 10000
        nVars = 4

        bbeta = np.array([1, 3, 5, 7])
        X = np.random.random((nObs,4))
        Y = (np.random.uniform(0,1,nObs) < 0.5).astype(int).reshape(nObs,1)
        clf = LogisticRegression(random_state=0).fit(X, Y)
        beta_1 = clf.coef_

        parameters = optimization(objectiveFunction, Y, X, np.array([1,2,3,4]))

        beta_2 = parameters['x']

        # Find test beta_hat is close to beta
        tol = .01
        abs_diff = np.all(abs(beta_1 - beta_2) < .05)
        self.assertTrue(abs_diff)
