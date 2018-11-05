import cvxpy as cvx
import numpy as np
import ktools
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


###########################################################################
class parameters:
    def __init__(self):
        self.N = 6
        self.d = 2
        self.P = 3
        self.omega = 2*np.pi
        self.mode = 1
        self.T_trn = np.array([-1, 1])
        self.T_tst = np.array([-1, 1])
        self.N_trn = ktools.K_base(self)
        self.K = ktools.K_base(self)
        self.N_tst= 500
        self.Nr = 5
        self.n_del = 0
        self.sampling = 1
        self.delta = 0.99
        self.std = 1
        self.maxIter = 5
        self.n_del_init = 0
        self.bipartite = False
        self.N0 = 3
        self.Pr = 0.9
        self.path = '../../../results/kedm/python3/'
param = parameters()

N = param.N
d = param.d
if d > 3 or d < 2:
    print('There is something wrong, dude!')
M = 300
t = np.linspace(param.T_tst[0],param.T_tst[1],M)
colors = np.random.rand(3,N)
fig_name = 'kedm_ambiguity.pdf'
fSize = 18
A=ktools.randomAs(param)

B0 = np.random.randn(d,d)
B1 = np.random.randn(d,d)
B2 = np.random.randn(d,d)
B3 = np.random.randn(d,d)
J = np.eye(N) - np.ones((N,N))/N
X = np.zeros((M,d,N))
Y = np.zeros((M,d,N))
for m in range(M):
    X[m,:,:] = ktools.X_t(A,t[m],param)
    Y[m,:,:] = np.matmul(X[m,:,:],J)
    f = B0 + t[m]/10 * B1 + math.cos(t[m])/ 20 * B2 + math.sin(t[m])*B3/20
    ff = np.matmul(f,f.T)
    u, s, vh = np.linalg.svd(ff, full_matrices=True)
    Y[m,:,:] = np.matmul(u,Y[m,:,:])

if d==2:
    fig, axes = plt.subplots(1, 2)
    for n in range(N):
        axes[0].plot(X[:,0,n], X[:,1,n],c = colors[:,n],linewidth=2)
    for n in range(N):
        axes[1].plot(Y[:,0,n], Y[:,1,n],c = colors[:,n],linewidth=2)
elif d==3:
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    for n in range(N):
        ax.plot(X[:,0,n], X[:,1,n],X[:,2,n],c = colors[:,n],linewidth=2)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    for n in range(N):
        ax.plot(Y[:,0,n], Y[:,1,n],Y[:,2,n],c = colors[:,n],linewidth=2)


plt.savefig(param.path+fig_name)
plt.show()
