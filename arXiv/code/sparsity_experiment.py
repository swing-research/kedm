#!/usr/bin/env python
# coding=utf-8
import kedm
import ktools
import numpy as np
trj_output = []
kedm_output = []
###########################################################################
class parameters:
    def __init__(self):
        self.N = 6
        self.d = 2
        self.P = 0
        self.omega = 2*np.pi
        self.mode = 1
        self.T_trn = np.array([-1, 1])
        self.T_tst = np.array([-1, 1])
        self.N_trn = ktools.K_base(self)
        self.K = ktools.K_base(self)
        self.N_tst= 500
        self.Nr = 5
        self.n_del = 0
        self.sampling = 1
        self.delta = 0.99
        self.std = 0
        self.maxIter = 100
        self.n_del_init = 0
        self.bipartite = False
        self.N0 = 3
        self.Pr = 0.9
        self.path = '../../../results/kedm/python3/'
param = parameters()
###########################################################################
param, S = kedm.FindMaxSprs(param)
#np.save(param.path+'S',S)
print('The maximum sparisty level is ', S)
###########################################################################
#kedm.Save(param, kedm_output, trj_output, '100')
###########################################################################
