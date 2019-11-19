import numpy as np

def detect_signInfluence(y, typeFn):

    out = 0
    
    if typeFn == 1:
        out = np.sign(np.sum(np.abs(y)*np.sign(y)))
    elif typeFn == 2:
        out = np.sign(y[0])
    elif typeFn == 3:
        out = np.sign(np.sum(np.sign(y)))
    if out == 0:out = -1
    return out



def get_directionalInfluence(beta, R=5, typeFn=2):

    C, _ = np.shape(beta)
    pos_beta_rec = np.zeros((C, C))

    for i in range(C):
        for j in range(C):
            if i == j:continue
            beta_rec = beta[i, j*R:(j+1)*R]
            sign_beta = detect_signInfluence(beta_rec, typeFn)
            pos_beta_rec[i, j] = sign_beta
    return pos_beta_rec