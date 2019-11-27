import numpy as np
from scipy.special import gamma
import pickle as pk
import os
import argparse

def metropolisHastingsSymmetric(p, nsamples):
	x = 0
	out = np.zeros((nsamples,))
	for i in range(nsamples):
		xHat = x + np.random.normal()
		if np.random.uniform() < p(xHat) / p(x):
			x = xHat
		out[i] = x
	return out

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

	#  unknowns 
	loggamma_a = 1
	loggamma_b = 50
	shiftLogGamma = -3.9
	flg = lambda  x: np.exp(loggamma_b * (x - shiftLogGamma)) \
					* np.exp(-np.exp(x - shiftLogGamma)/loggamma_a) \
					/((loggamma_a**loggamma_b)*gamma(loggamma_b))

	nsamples = params['K']*params['I']
	# sample from log-gamma distribution with mean centered
	dU = metropolisHastingsSymmetric(flg, nsamples)
	dU = np.reshape(dU, (params['I'], params['K']))

	dN = np.zeros((params['C'], params['K']))
	dN[:,0] = np.random.uniform(0, 1, (params['C'],))>0.5

	lambda_ = np.zeros((params['C'], params['K']))
	for k in range(1, params['K']):
		for c in range(params['C']):
			in_ = np.dot(epsi[c, :min([k-1, params['Q']])+1], 
					np.flip(dN[c, k-min([k, params['Q']]):k], axis=0)
					)
			ex_ = np.sum(np.squeeze(beta[:,c,:min([k-1,params['R']])+1]) * 
					np.fliplr(dN[:, k-min([k, params['R']]):k])
					)
			if params['flag'] == 'wUU':
				if params['I'] == 1:
					un_ = np.sum(np.squeeze(gammaUU[c,:,:min([k-1,params['M']])]) * 
							np.flip(dU[:, k - min([k, params['M']]):k], axis=0)
							)
				else:
					un_ = np.sum(np.squeeze(gammaUU[c,:,:min([k-1,params['M']])+1]) * 
							np.fliplr(dU[:, k - min([k, params['M']]):k])
						)
			else:
				un_ = 0
			lambda_[c, k] = np.exp(alpha[c] + in_ + ex_ + un_)

		for i in range(params['C']):
			u = np.random.uniform(0, 1)
			if u <= lambda_[i, k] * params['tau']:
				dN[i,k] = 1

	return {'dN':dN, 'alpha':alpha, 'epsi':epsi,
			'beta':beta, 'pos_beta':pos_beta, 'gammaUU':gammaUU,
			'params':params}

flagTypes = ['woUU', 'wUU']
parser = argparse.ArgumentParser(description="artificial spikes generation")
parser.add_argument('-C', '--num_nodes',help='number of nodes', default=6,
					type=int)
parser.add_argument('-K', '--length',  help='length of spiking events', default=1500,
						type=int)
parser.add_argument('-Q', '--intr_len', default=50,
						help='intrinsic memory length', type=int)
parser.add_argument('-R', '--extr_len', default=5,
						help='extrinsic memory length', type=int)
parser.add_argument('-M', '--unknown_len', default=5,
					help = 'unknown activity memory length', type=int)
parser.add_argument('-I', '--num_unk', default=2,
					help = 'Number of unknowns', type=int)
parser.add_argument('-tau', '--tau', default=0.05,
					help = 'spiking interval length', type=float)
parser.add_argument('-f', '--flag', default='woUU', choices=flagTypes,
					help = 'type of spike samples, with/without unknowns', type=str)


if __name__ == '__main__':
	# flag = woUU: for generating spikes without unknowns contribution
	# 		 wUU : for generating spikes with unknowns contribution
	args = parser.parse_args()
	params = {'C':args.num_nodes, 'Q':args.intr_len, 'R':args.extr_len,
				'M':args.unknown_len, 'I':args.num_unk,
					'K':args.length, 'tau':args.tau, 'flag':args.flag} 
	dataDir = 'data'
	if not os.path.exists(dataDir):
		os.makedirs(dataDir, exist_ok=True)

	numExperiments = 5
	for expInd in range(numExperiments):
		out = generateSpike(params, expInd)
		saveStr = 'neuronSpikeSim_%s_C_%d_K_%d_exp_%d.p'%(params['flag'], params['C'], params['K'], expInd)
		pk.dump(out, open(os.path.join(dataDir, saveStr), 'wb'))