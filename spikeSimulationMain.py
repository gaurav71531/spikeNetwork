import numpy as np
import pickle as pk
from scipy.io import loadmat, savemat
import os
from soln import *
from utilities import *
import argparse

flagTypes = ['woUU', 'wUU']
parser = argparse.ArgumentParser(description="network reconstruction from spikes")
parser.add_argument('-C', '--num_nodes',help='number of nodes', default=6,
					type=int)
parser.add_argument('-K', '--length',  help='length of spiking events', default=500,
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


def startSim(args, simType):

    C = args.num_nodes
    K = args.length
    net = spikeNetEst(args.intr_len, args.extr_len, 
                        args.num_unk, args.unknown_len, tau = args.tau)
    dataDir = 'data'
    expList = np.arange(5)
    acc_List = np.zeros((np.size(expList),))
    for i, expInd in enumerate(expList):
        fName = 'neuronSpikeSim_%s_C_%d_K_%d_exp_%d.p'%(simType, C, K, expInd)
        dataRaw = pk.load(open(os.path.join(dataDir, fName), 'rb'))
        pos_beta_orig = dataRaw['pos_beta']

        net.initialize(dataRaw['dN'], gammaUU = dataRaw['gammaUU'])
        _, _, beta, _ = net.solve(flag=args.flag, verbose=1)

        pos_beta_rec = get_directionalInfluence(beta)
        numCorrections = np.sum(pos_beta_orig == pos_beta_rec) - args.num_nodes
        per_acc = numCorrections / (C**2-C) * 100
        print('percentage accuracy for experiment:%d = %f%%'%(expInd, per_acc))
        acc_List[i] = per_acc
    print('mean accuracy = %f%%, std dev = %f%%'%(np.mean(acc_List), np.std(acc_List)))
    
    return 1

if __name__ == '__main__':
    # for taking simulation spikes without unknowns set simType to woUU, 
    # and for simulating with unknowns set simType to wUU
    simType = 'wUU'  # possibilities woUU, wUU

    # for using the network reconstruction algorithm without unknowns, set -f to woUU,
    # and for using the algorthm with unknowns set -f to wUU

    # For example:
    # to simulate a 6 node network with 1500 spikes, set the following:
    # simType = 'woUU'
    
    # $python spikeSimulationMain.py -f woUU -K 1500 -C 6

    # to simulate a 6 node network with 1500 spikes and 3 unknowns, set the following:
    # simType = 'wUU'

    # $python spikeSimulationMain.py -f woUU -K 1500 -C 6 -I 3, for using the algorithm without unknowns
    # and
    # $python spikeSimulationMain.py -f wUU -K 1500 -C 6 -I 3, for using the algorithm with unknowns

    args = parser.parse_args()
    startSim(args, simType)