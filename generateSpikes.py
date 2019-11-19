import numpy as np
import pickle as pk
import os

def generateSpike(params, expInd = 0):

	alpha = np.random.uniform(0, 1, (params['C'],)) + 2.5
	epsi = np.zeros((params['C'], params['Q'])) # % intrinsic paramter
	beta = np.zeros((params['C'], params['C'], params['R'])) # extrinsic paramter
	gammaUU = np.zeros((params['C'], params['I'], params['M'])) # unknown paramter

	#  intrinsic para
	x = np.arange(1,params['Q']+1)
	z_epsi = np.random.uniform(1.5, 2, (params['C'],))
	for c in range(params['C']):
		epsi[c,:] = -np.sin(x / z_epsi[c]) / (x / z_epsi[c])

	# extrinsic para

	pos_beta = 2*(np.random.uniform(0, 1, (params['C'],params['C']))>0.5)-1
	np.fill_diagonal(pos_beta, 0)
	z_beta = np.random.uniform(0.5, 1, (params['C'],params['C']))
	for c in range(params['C']):
		for c1 in range(params['C']):
			beta[c,c1,:] = pos_beta[c,c1]*np.exp((-z_beta[c,c1])*np.arange(1, params['R']+1))

	dN = np.zeros((params['C'], params['K']))
	dN[:,0] = np.random.uniform(0, 1, (params['C'],))>0.5

	lambda_ = np.zeros((params['C'], params['K']))
	for k in range(1, params['K']):
		for c in range(params['C']):
			in_ = np.dot(epsi[c, :np.min([k-1, params['Q']])+1], 
					np.flip(dN[c, k-np.min([k, params['Q']]):k], axis=0)
					)
			ex_ = np.sum(np.squeeze(beta[:,c,:np.min([k-1,params['R']])+1]) * 
					np.fliplr(dN[:, k-np.min([k, params['R']]):k])
					)
			un_ = 0
			lambda_[c, k] = np.exp(alpha[c] + in_ + ex_ + un_)

		for i in range(params['C']):
			u = np.random.uniform(0, 1)
			if u <= lambda_[i, k] * params['tau']:
				dN[i,k] = 1

	return {'dN':dN, 'alpha':alpha, 'epsi':epsi,
			'beta':beta, 'pos_beta':pos_beta, 'params':params}


if __name__ == '__main__':
    params = {'C':6, 'Q':50, 'R':5, 'M':5, 'I':3,
                 'K':1500, 'tau':0.05}
    dataDir = 'data'
    if not os.path.exists(dataDir):
        os.makedirs(dataDir, exist_ok=True)

    numExperiments = 5
    for expInd in range(numExperiments):
        out = generateSpike(params, expInd)
        saveStr = 'neuronSpikeSim_wUU_C_%d_K_%d_exp_%d.p'%(params['C'], params['K'], expInd)
        pk.dump(out, open(os.path.join(dataDir, saveStr), 'wb'))