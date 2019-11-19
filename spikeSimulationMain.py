import numpy as np
import pickle as pk
from scipy.io import loadmat, savemat
import os
from soln import *
from utilities import *
import argparse

parser = argparse.ArgumentParser(description="network reconstruction from spikes")
parser.add_argument('-C', '--num_nodes',help='number of nodes', default=6,
					type=int)
parser.add_argument('-K', '--length',  help='length of spiking events', default=500,
						type=int)
parser.add_argument('-Q', '--intr_len', default=150,
						help='intrinsic memory length', type=int)
parser.add_argument('-R', '--extr_len', default=5,
						help='extrinsic memory length', type=int)
parser.add_argument('-M', '--unknown_len', default=5,
					help = 'unknown activity memory length', type=int)
parser.add_argument('-I', '--num_unk', default=2,
					help = 'Number of unknowns', type=int)
parser.add_argument('-tau', '--tau', default=0.05,
					help = 'spiking interval length', type=float)


def startSim(args):

    C = 6
    K = 1500
    net = spikeNetEst(args.intr_len, args.extr_len, 
                        args.num_unk, args.unknown_len, args.tau)
    dataDir = 'data'
    expList = np.arange(5)
    acc_List = np.zeros((np.size(expList),))
    for i, expInd in enumerate(expList):
        fName = 'neuronSpikeSim_wUU_C_%d_K_%d_exp_%d.p'%(C, K, expInd)
        dataRaw = pk.load(open(os.path.join(dataDir, fName), 'rb'))
        pos_beta_orig = dataRaw['pos_beta']

        net.initialize(dataRaw['dN'])
        _, _, beta, _ = net.solve('woUU')

        pos_beta_rec = get_directionalInfluence(beta)
        numCorrections = np.sum(pos_beta_orig == pos_beta_rec) - C
        per_acc = numCorrections / (C**2-C) * 100
        print('percentage accuracy for experiment:%d = %f%%'%(expInd, per_acc))
        acc_List[i] = per_acc
    print('mean accuracy = %f%%, std dev = %f%%'%(np.mean(acc_List), np.std(acc_List)))
    
    return 1

if __name__ == '__main__':
    args = parser.parse_args()
    startSim(args)