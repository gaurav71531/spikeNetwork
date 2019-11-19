import numpy as np
from numba import prange

class spikeNetEst(object):

    def __init__(self, Q = 50, R = 5, I = 2, M = 5, tau = 0.05):
        self.Q = Q
        self.R = R
        self.M = 5
        self.I = 0
        self.tau = tau

    def initialize(self, dN, iterMaxM = 100,
                    iterMaxE = 300, iterMaxEM = 50):
        C, K = np.shape(dN)
        dimPara = 1 + self.Q + C*self.R
        self.iterMaxM = iterMaxM
        self.iterMaxE = iterMaxE
        self.iterMaxEM = iterMaxEM
        self.dimPara = dimPara
        MLEpara0 = np.ones((C, dimPara))*np.exp(1)

        for c in range(C):
            MLEpara0[c,0] = np.exp(np.mean(dN[c,:])/0.05)
            MLEpara0[c,1+self.Q+c*self.R:1+self.Q+(c+1)*self.R]=1
    
        self.Y = self.getYVec(dN)
        self.MLEpara0 = MLEpara0
        self.dN = dN
        self.C = C
        self.K = K


    def solve(self, flag):

        llh = np.zeros((self.iterMaxEM,))
        MLEdU0 = np.zeros((self.I, self.K))
        gammaUU = np.zeros((self.C, self.I, self.M))
        # initialize EM params
        dUHat = MLEdU0
        M_para = self.MLEpara0
        lambdaUU = self.getLambdaU(dUHat, gammaUU)
        M_para, lambda_, llh[0] = self.fixPointIter_M(M_para, lambdaUU, self.iterMaxM*self.iterMaxEM)

        llh = np.ones((self.iterMaxEM,))*llh[0]

        M_para = np.log(M_para)
        alpha = M_para[:,0]
        epsi = M_para[:,1:1+self.Q]
        beta = M_para[:,1+self.Q:]
        for c in range(self.C):
            for c1 in range(c):
                temp = np.array(beta[c, c1*self.R:(c1+1)*self.R])
                beta[c,c1*self.R:(c1+1)*self.R] = beta[c1, c*self.R:(c+1)*self.R]
                beta[c1, c*self.R:(c+1)*self.R] = temp

        return alpha, epsi, beta, llh

    def fixPointIter_M(self, gammaPrev, lambdaU, niter):
        gammaOut = np.zeros(np.shape(gammaPrev))

        likelihood = np.zeros((self.C, niter))
        lambda_ = np.zeros((self.C, self.K))

        for c in range(self.C):
            YUseForC = np.reshape(self.Y[c,:,:], (self.K, self.dimPara))

            sumY = np.sum(YUseForC, axis=1)
            G_num = self.dN[c,:]@YUseForC
            betaDen = np.sum(YUseForC *
                         (np.reshape(sumY*self.dN[c,:], (self.K,1))@np.ones((1, self.dimPara))), axis=0)
            beta = G_num / betaDen

            for iterInd in range(niter):
                lambdaMat = np.power((np.ones((self.K,1)) @ np.reshape(gammaPrev[c,:], (1, self.dimPara))), YUseForC)
                lambdaUse = np.prod(lambdaMat, axis=1)

                G_den = (lambdaUse.T * lambdaU[c,:]) @ YUseForC * self.tau
                gammaOut[c,:] = gammaPrev[c,:] * np.power((G_num/G_den), beta)
#               for c' = c, set gamma = 1
        #       to exclude from the summation
                gammaOutTemp = gammaOut[c,:]
                gammaOutTemp[1+self.Q+c*self.R:1+self.Q+(c+1)*self.R] = 1
                gammaOut[c,:] = gammaOutTemp

                likelihood[c,iterInd] = np.sum(self.dN[c,1:-1] * np.log(lambdaU[c,1:-1] * lambdaUse[1:-1])
                                            -self.tau * lambdaU[c,1:-1] * lambdaUse[1:-1]
                                        )
                gammaPrev[c,:] = gammaOut[c,:]
            lambdaMat = np.power(np.ones((self.K, 1))@np.reshape(gammaOut[c,:], (1, self.dimPara)), YUseForC)
            lambdaUse = np.prod(lambdaMat, axis=1)
            lambda_[c,:] = lambdaUse
        llh = np.sum(likelihood[:,-1])

        return gammaOut, lambda_, llh

    def getYVec(self, dN):
        C, K = np.shape(dN)
        Y = np.zeros((C, K, self.dimPara))
        for c in range(C):
            # Y(c,k,:)=[1 ... dN(c,k-q) ... dN(c1,k-r)...dU(i,k-m)] D-length
            Y[c,0,0] = 1
            for k in range(1, K):
                epsi_len = np.min([k, self.Q])
                beta_len = np.min([k, self.R])
                Y[c,k,0] = 1
                Y[c,k,1:epsi_len+1] = np.flip(dN[c,k-epsi_len:k], axis=0)
                for c1 in range(C):
                    Y[c,k,1+self.Q+c1*self.R:1+self.Q+c1*self.R+beta_len] = np.flip(dN[c1,k-beta_len:k], axis=0)
        return Y


    def getLambdaU(self, dU, gammaUU):

        lambdaU = np.zeros((self.C, self.K))
        for c in range(self.C):
            convol = np.zeros((self.K,))
            for ii in range(self.I):
                gVec = np.array((0, np.reshape(gammaUU[c,ii,:], (self.M,))))
                cv1 = np.convolve(gVec, dU[ii,:])
                convol += cv1[:self.K]
            lambdaU[c,:] = np.exp(convol)
        return lambdaU
            

