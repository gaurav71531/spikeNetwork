import numpy as np
import pickle as pk
import os

def generateSpike(params, expInd = 0):

    alpha = np.random.uniform((1, params['C'])) + 2.5
    epsi = np.zeros((C, Q)) # % intrinsic paramter
    beta = np.zeros((params['C', params['C'], params['R'])) # extrinsic paramter
    gammaUU = np.zeros((params['C'], params['I'], params['M'])) # unknown paramter

    #  intrinsic para
    x = np.arange(1,params['Q'])
    z_epsi = np.random.uniform((1, params['C']))/2 + 1.5
    for c in range(params['C']):
        epsi[c,:] = -np.sin(x / z_epsi[c]) / (x / z_epsi[c])
    
    # extrinsic para

    pos_beta = 2*(np.random.uniform((params['C'],params['C']))>0.5)-1;
    np.fill_diagonal(pos_beta, 0)
    z_beta = np.random.uniform((params['C'],params['C']))/2 + 0.5
    for c in range(params['C']):
        for c1 in range(params['C']):
            beta[c,c1,:] = pos_beta[c,c1]*np.exp((-z_beta[c,c1])*np.arange(1, params['R']))
    

    return {'1':1}


if __name__ == '__main__':
    params = {'C':6, 'Q':50, 'R':5, 'M':5, 'I':3,
                 'K':300, 'tau':0.05}
    dataDir = 'data'
    if not os.path.exists(dataDir):
        os.makedirs(dataDir, exist_ok=True)

    numExperiments = 5
    for expInd in range(numExperiments):
        out = generateSpike(params, expInd)
        saveStr = 'neuronSpikeSim_wUU_C_%d_K_%d_exp_%d.p'%(params['C'], params['K'], expInd)
        pk.dump(out, open(os.path.join(dataDir, saveStr), 'wb'))


