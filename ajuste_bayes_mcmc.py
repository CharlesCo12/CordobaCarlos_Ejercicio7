import matplotlib.pylab as plt
import numpy as np
def model(x,betas,b):
    return np.sum(x*betas,axis=1) + b

# Numericamente es mas estable trabajar con el logaritmo
def loglikelihood(x_obs, y_obs, sigma_y_obs, m, b):
    d = y_obs -  model(x_obs, m, b)
    d = d/sigma_y_obs
    d = -0.5 * np.sum(d**2)
    return d

# Numericamente es mas estable trabajar con el logaritmo
def logprior(m, b):
    return 0
datos=np.loadtxt('notas_andes.dat')
y_obs=datos[:,4]
x_obs=datos[:,0:4]
sigma=0.1
N = 20000
lista_betas = [np.random.random(size=4)]
lista_b = [np.random.random()]
logposterior = [loglikelihood(x_obs, y_obs, sigma, lista_betas[0], lista_b[0]) + logprior(lista_betas[0], lista_b[0])]

sigma_delta_m = 0.015
sigma_delta_b = 0.01

for i in range(1,N):
    propuesta_betas = lista_betas[i-1] + np.random.normal(size=4,loc=0.0, scale=sigma_delta_m)
    propuesta_b  = lista_b[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_b)

    logposterior_viejo = loglikelihood(x_obs, y_obs, sigma, lista_betas[i-1], lista_b[i-1]) + logprior(lista_betas[i-1], lista_b[i-1])
    logposterior_nuevo = loglikelihood(x_obs, y_obs, sigma, propuesta_betas, propuesta_b) + logprior(propuesta_betas, propuesta_b)

    r = min(1,np.exp(logposterior_nuevo-logposterior_viejo))
    alpha = np.random.random()
    if(alpha<r):
        lista_betas.append(propuesta_betas)
        lista_b.append(propuesta_b)
        logposterior.append(logposterior_nuevo)
    else:
        lista_betas.append(lista_betas[i-1])
        lista_b.append(lista_b[i-1])
        logposterior.append(logposterior_viejo)
        
lista_betas = np.array(lista_betas)
lista_b = np.array(lista_b)
logposterior = np.array(logposterior)
B=np.mean(lista_betas[10000:,:],axis=0)
B0=np.mean(lista_b[10000:])
B_STD=np.std(lista_betas[10000:,:],axis=0)
B0_STD=np.std(lista_b[10000:])
plt.figure(figsize=(12,8))
plt.subplot(2,3,1)
_=plt.hist(lista_b[10000:],density=True,bins=40)
plt.xlabel('b0')
plt.ylabel('P(b0|datos)')
plt.title(r'$\beta_0$ = {:.2f} $\pm$ {:.2f}'.format(B0,B0_STD))
for i in range(4):
    plt.subplot(2,3,i+2)
    _=plt.hist(lista_b[10000:],density=True,bins=40)    
    plt.title(r'$\beta$ = {:.2f} $\pm$ {:.2f}'.format(B[i],B_STD[i]))
plt.savefig('ajuste_bayes_mcmc.png')
