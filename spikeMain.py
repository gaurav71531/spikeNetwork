import numpy as np
import pickle as pk
import os
from soln import *


def startSim():

    C = 6
    K = 300
    expInd = 0
    dataDir = 'data'
    fName = 'neuronSpikeSim_wUU_C_%d_K_%d_exp_%d.p'%(C, K, expInd)
    dataRaw = pk.load(open(os.path.join(dataDir, fName), 'rb'))
    pos_beta_orig = dataRaw['pos_beta']

    net = spikeNetEst(Q=50, R = 5, I = 2, M = 5, tau=0.05)

    net.initialize(dataRaw['dN'])
    alpha, epsi, beta, llh = net.solve('woUU')

    
    return 1

if __name__ == '__main__':
    startSim()