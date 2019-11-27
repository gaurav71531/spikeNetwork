import numpy as np
from numba import prange

class spikeNetEst(object):

    def __init__(self, Q = 50, R = 5, I = 2, M = 5, 
                loggamma_a = 1, loggamma_b = 2,
                tau = 0.05):
        self.Q = Q
        self.R = R
        self.M = M
        self.I = I
        self._loggamma_a = loggamma_a
        self._loggamma_b = loggamma_b
        self.tau = tau


    def initialize(self, dN, gammaUU = [], iterMaxM = 100,
                    iterMaxE = 300, iterMaxEM = 50):
        try:
            C, K = np.shape(dN)
            if C < 2:
                raise ValueError('insuff_nodes')
            if K < max([self.Q, self.R, self.M]):
                raise NameError('dimErr')
        except NameError as inst:
            if inst.args[0] == 'dimErr':
                print('Insufficient number of samples, try with more spike trains...')
            raise
        except ValueError as inst:
            print('Not enough number of nodes, need atleast 2')
            raise

        dimPara = 1 + self.Q + C*self.R
        self.iterMaxM = iterMaxM
        self.iterMaxE = iterMaxE
        self.iterMaxEM = iterMaxEM
        self.dimPara = dimPara
        MLEpara0 = np.ones((C, dimPara))*np.exp(1)

        for c in range(C):
            MLEpara0[c,0] = np.exp(np.mean(dN[c,:])/0.05)
            MLEpara0[c,1+self.Q+c*self.R:1+self.Q+(c+1)*self.R]=1
    
        self._Y = self._getYVec(dN)
        self._MLEpara0 = MLEpara0
        self.dN = dN
        self.C = C
        self.K = K
        self._MLEdU0 = np.random.uniform(-1,1,(self.I, self.K))

        if len(gammaUU) == 0:
            self.gammaUU = np.zeros((self.C, self.I, self.M))
        else:
            try:
                if (self.C, self.I, self.M) != np.shape(gammaUU):
                    raise ValueError('wrong_inp')
            except ValueError as inst:
                print('The shape of gammaUU should be num_nodes (C), num_inputs (I), len_inputs (M)')
                print('Current shape: %s, required shape: %s'%(np.shape(gammaUU), (self.C, self.I, self.M)))
                raise
            self.gammaUU = gammaUU


    def solve(self, flag = 'woUU', verbose = 0):

        llh = np.zeros((self.iterMaxEM,))
        # MLEdU0 = np.zeros((self.I, self.K))
        # gammaUU = np.zeros((self.C, self.I, self.M))
        # initialize EM params
        dUHat = self._MLEdU0
        M_para = self._MLEpara0
        lambdaUU = self._getLambdaU(dUHat, self.gammaUU)
        if flag == 'woUU':
            M_para, _, llh[0] = self._fixPointIter_M(M_para, lambdaUU, self.iterMaxM*self.iterMaxEM)
            if verbose > 0:
                print('Solution, likelihood = %f'%(llh[0]))
            llh = np.ones((self.iterMaxEM,))*llh[0]
        else:
            M_para, lambda_, llh[0] = self._fixPointIter_M(M_para, lambdaUU, self.iterMaxM)
            if verbose > 0:
                print('Intialization, likelihood = %f'%(llh[0]))
            for iterInd in range(1, self.iterMaxEM):
                # E-step
                dUHat, lambdaUU = self._fixPointIter_E(dUHat, lambda_, self.gammaUU, lambdaUU, self.iterMaxE)
                # M-step
                M_para, lambda_, llh[iterInd] = self._fixPointIter_M(M_para, lambdaUU, self.iterMaxM)
                if verbose > 0:
                    if iterInd % 10 == 0:
                        print('Iterations complete = %d, likelihood = %f'%(iterInd, llh[iterInd]))

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


    def _fixPointIter_E(self, dUHat, lambda_, gammaUU, lambdaUU, niter):

        muPrev = np.exp(dUHat)
        D = self.K
        muCurrent = np.array(muPrev)

        G_num = np.zeros((self.I, D-1))
        G_den = np.zeros((self.I, D-1))

        tConst = 0.1

        for i in range(self.I):
            convol = np.zeros((D-1,))
            for c in range(self.C):
                c1 = np.convolve(np.flip(np.concatenate((np.array([0]), np.squeeze(gammaUU[c,i,:]))), axis=0),
                                self.dN[c,:]
                    )
                convol += c1[self.M:-1]
            G_num[i,:] = self._loggamma_b + convol
        
        t_den, t = self._getTExp(D, gammaUU, tConst, G_num)
        F = np.zeros((niter,))
        for iterInd in range(niter):
            cvMatTemp = lambda_ * lambdaUU
            for i in range(self.I):
                convol = np.zeros((D-1,))
                for c in range(self.C):
                    c1 = np.convolve(np.concatenate((np.array([0]), np.squeeze(gammaUU[c,i,:]))),
                                     cvMatTemp[c,:])
                G_den[i,:] = self.tau * convol + muPrev[i,:-1]/self._loggamma_a
                muCurrent[i,1:-1] = muPrev[i,1:-1] * np.power(G_num[i,1:]/G_den[i,1:], t[i,1:])
                t_reject_index = np.where(t_den[i,:]<1)
                muCurrent[i,t_reject_index] = 1
                # muCurrent[i,t_den[i,:]<1] = 1
            
            dUHat = np.log(muCurrent)
            lambdaUU = self._getLambdaU(dUHat, gammaUU)
            F[iterInd] = np.sum(self.dN[:,1:-1]*(np.log(lambdaUU[:,1:-1]) + np.log(lambda_[:,1:-1]))
                                - self.tau * lambdaUU[:,1:-1] * lambda_[:,1:-1]
                            ) \
                        + np.sum(self._loggamma_b * np.log(muCurrent[:,1:-1]) 
                                - muCurrent[:,1:-1]/self._loggamma_a
                            )
            muPrev = np.array(muCurrent)
        return dUHat, lambdaUU


    def _fixPointIter_M(self, gammaPrev, lambdaU, niter):
        gammaOut = np.zeros(np.shape(gammaPrev))

        likelihood = np.zeros((self.C, niter))
        lambda_ = np.zeros((self.C, self.K))

        for c in range(self.C):
            YUseForC = np.reshape(self._Y[c,:,:], (self.K, self.dimPara))

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


    def _getTExp(self, D, gammaUU, tConst, G_num):

        np.seterr(divide='ignore') # setting due to possible 0's in t_den
        # we will ignore those values later in the code
        t_den = np.zeros((self.I, D-1))
        t = np.zeros((self.I, D-1))
        for i in range(self.I):
            for q in range(D-1):
                for c in range(self.C):
                    for p in range(max([q-self.M, 0]), min([q+self.M, self.K])):
                        for k in range(max([p+1, q+1]), min([p+self.M+1, q+self.M+1, self.K])):
                            t_den[i, q] += gammaUU[c,i,k-q-1]*gammaUU[c,i,k-p-1]*self.dN[c,k]
                            # print(gammaUU[c,i,k-q-1]*gammaUU[c,i,k-p-1]*self.dN[c,k])
            t[i,:] = tConst * G_num[i,:] / t_den[i,:]
        
        return t_den, t
    

    def _getYVec(self, dN):
        C, K = np.shape(dN)
        Y = np.zeros((C, K, self.dimPara))
        for c in range(C):
            # Y(c,k,:)=[1 ... dN(c,k-q) ... dN(c1,k-r)...dU(i,k-m)] D-length
            Y[c,0,0] = 1
            for k in range(1, K):
                epsi_len = min([k, self.Q])
                beta_len = min([k, self.R])
                Y[c,k,0] = 1
                Y[c,k,1:epsi_len+1] = np.flip(dN[c,k-epsi_len:k], axis=0)
                for c1 in range(C):
                    Y[c,k,1+self.Q+c1*self.R:1+self.Q+c1*self.R+beta_len] = np.flip(dN[c1,k-beta_len:k], axis=0)
        return Y


    def _getLambdaU(self, dU, gammaUU):

        lambdaU = np.zeros((self.C, self.K))
        for c in range(self.C):
            convol = np.zeros((self.K,))
            for ii in range(self.I):
                gVec = np.concatenate((np.array([0]), np.reshape(gammaUU[c,ii,:], (self.M,))))
                cv1 = np.convolve(gVec, dU[ii,:])
                convol += cv1[:self.K]
            lambdaU[c,:] = np.exp(convol)
        return lambdaU