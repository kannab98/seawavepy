import numpy as np

def cross_section(theta, cov) : 
    theta = theta[np.newaxis]
    # Коэффициент Френеля
    F = 0.8

    if len(cov.shape) <= 2:
        cov = np.array([cov])

    K = np.zeros(cov.shape[0])
    for i in range(K.size):
        K[i] = np.linalg.det(cov[i])

    sigma =  F**2/( 2*np.cos(theta.T)**4 * np.sqrt(K) )
    sigma *= np.exp( - np.tan(theta.T)**2 * cov[:, 1, 1]/(2*K))
    return sigma