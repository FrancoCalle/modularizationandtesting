import unittest
import sys
import os

sys.path.append(os.path.abspath('.'))

import numpy as np

from ols import ols
import logit



class TestOLS(unittest.TestCase):
    def test_zero(self):
        y = np.zeros([2, 2])
        X = np.eye(2)
        b, se = ols(y, X)
        self.assertEqual(np.linalg.norm(b), 0)
        
    def test_known_output(self):
        y = np.array([[.3488717, .2668857, .1366463, .0285569, .8689333, .3508549, .0711051, .323368, .5551032, .875991, .2047095, .8927587, .5844658, .3697791, .8506309, .3913819, .1196613, .7542434, .6950234, .6866152, .9319346, .4548882, .0674011, .3379889, .9748848, .7264384, .0454151, .7459667, .4961259, .7167162, .859742, .1340756, .4884419, .8712187, .7664683, .2512555, .1663648, .7437958, .9805113, .7295772, .9011049, .2643649, .8856509, .882112, .748933, .9196262, .6934533, .2154026, .8285888, .0442154, .8630378, .3526046, .7720399, .5861199, .3227766, .1729307, .8053644, .3060019, .2190997, .724731, .6964867, .9119344, .6795634, .3549416, .73897, .1874017, .3146128, .1375693, .6537739, .2701319, .8998394, .5734232, .1114704, .4145227, .0030522, .6659978, .3462876, .0780235, .1275814, .2297006, .3295547, .4144089, .0360847, .0843811, .0098762, .3200437, .005197, .2275435, .851468, .9820066, .0324792, .9874847, .894106, .9684734, .2392203, .6927336, .4884359, .4376452, .5858005, .3787092]]).T
        X = np.array([[.6880603, .9794578, .6701937, .5948808, .7970893, .7835853, .6546342, .0968891, .6885059, .872496, .5296353, .8302209, .9339853, .1749891, .5536171, .5346152, .7767794, .1288747, .2775184, .4242016, .1359006, .3325624, .4675523, .5160881, .066943, .0722964, .6817465, .0880495, .1327082, .8745816, .2468877, .043255, .3764437, .7677861, .7551366, .4476188, .4087105, .2977743, .6794177, .7124024, .5662265, .1778325, .113999, .5955869, .6251604, .634899, .9944572, .7497677, .1736788, .6107705, .5754215, .3678161, .3005246, .007538, .6701369, .4241406, .9537622, .0867478, .8949648, .5890286, .4005832, .6654902, .4198386, .7472054, .7190143, .8464647, .7908313, .1900222, .3869604, .2387134, .3447002, .7795682, .7484396, .2303784, .1677032, .9180508, .3138996, .9019141, .0774052, .6341382, .8147295, .8788922, .0259935, .17993, .5778896, .4081415, .6155495, .174577, .3617646, .1338996, .0013631, .2571, .6517417, .9252081, .8233367, .9229402, .7480426, .5214148, .4022151, .8681989],
                      [.2726605, .7239472, .7955464, .8925074, .7078791, .365269, .9310499, .6216809, .8004354, .4798372, .142948, .8343448, .3431251, .5906867, .32953, .6996295, .8142969, .3429726, .1079681, .9671743, .1285523, .7578536, .25003, .9269138, .7118431, .1796715, .4506519, .1946068, .7135741, .2453114, .7672456, .3653557, .2706914, .9911318, .6851298, .5027668, .6903818, .8636012, .0404633, .1842219, .4198807, .6475499, .9103145, .6809221, .8568827, .0642018, .8390664, .6208202, .4041756, .9786366, .3627681, .3138244, .6380712, .1869937, .5053477, .5276305, .7853414, .4717338, .2299842, .7976828, .1649395, .932945, .3999315, .9881987, .9287856, .6640378, .0368038, .3336498, .7824295, .0170049, .2278204, .5782465, .7533595, .8570072, .9322746, .324447, .1637711, .958201, .6008608, .9733476, .2363827, .6764786, .1459103, .2966402, .8219558, .3213928, .4164997, .0236964, .3125404, .9322619, .0502046, .6221892, .6189114, .9028944, .3830579, .3513705, .6978495, .7828125, .8312564, .8498105],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).T
        b, se = ols(y, X)
        
        b0, b1, b2 = b.T[0]
        self.assertAlmostEqual(b0, 0.03674418, 6)
        self.assertAlmostEqual(b1, 0.00053208, 6)
        self.assertAlmostEqual(b2, 0.47830007, 6)
        
        se0, se1, se2 = se
        self.assertAlmostEqual(se0, 0.11348108, 6)
        self.assertAlmostEqual(se1, 0.11106197, 6)
        self.assertAlmostEqual(se2, 0.07989867, 6)

if __name__ == '__main__':
    unittest.main()