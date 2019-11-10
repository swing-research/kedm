import cvxpy as cvx
import os, sys
import numpy as np
import ktools
import math
###########################################################################
class KEDM_OUT:
    def __init__(self):
        self.eo         = None
        self.ei         = None
        self.G          = None
        self.trn_list   = None
        self.tst_list   = None
        self.tau_list   = None
        self.A          = None
        self.status     = None
class TRJ_OUT:
    def __init__(self):
        self.A_         = None
        self.eX         = None
###########################################################################
def KEDM(param, A):
    N = param.N
    d = param.d
    K = ktools.K_base(param)
    ###########################################################################
    MaxIter = 500
    verbosity = True
    ###########################################################################
    output = KEDM_OUT()
    
    tau_list = ktools.generalsampler(param, 'basis')
    trn_list = ktools.generalsampler(param, 'trn_sample')
    tst_list = ktools.generalsampler(param, 'tst_sample')
    log_list = ktools.generalsampler(param, 'log_sample')
    
    D, W, ei = ktools.generateData(param, trn_list, A)
    
    G = []
    con = []
    for k in range(K):
        G.append(cvx.Variable((N, N), PSD=True))
        con.append(cvx.sum(G[k],axis = 0) == 0)
    for t in log_list:
        weights = ktools.W(t, tau_list, param)
        weights = weights/np.linalg.norm(weights)
        G_tot = ktools.G_t(G, weights, True)
        con.append(G_tot>>0)
    cost = 0
    for i_t, t in enumerate(trn_list):
        weights = ktools.W(t, tau_list, param)
        G_tot = ktools.G_t(G, weights, True)
        con.append(G_tot >> 0)
        D_G = ktools.gram2edm(G_tot, N, True)
        W_vec = np.diag(W[i_t].flatten())
        alpha = (np.linalg.norm( np.matmul(W_vec, D[i_t].flatten()) ) )**(-2)
        cost += alpha*cvx.norm( cvx.matmul(W_vec, cvx.vec(D[i_t]-D_G) ) )**2
    
    
    obj = cvx.Minimize(cost)
    prob = cvx.Problem(obj,con)

try:
    prob.solve(solver=cvx.CVXOPT,verbose=False,normalize = True)
    except Exception as message:
        print(message)

output.status = str(prob.status)
if str(prob.status) == 'optimal':
    G = ktools.rankProj(G, param)
    eo = ktools.testError(param, tau_list, tst_list, G, A)
    output.G        = G
        output.tau_list = tau_list
        output.trn_list = trn_list
        output.tst_list = tst_list
        output.ei       = ei
        output.eo       = eo
        output.A        = A
    return output
###########################################################################
def FindMaxSprs(param):
    n_del_init = param.n_del_init
    maxIter = param.maxIter
    Pr = param.Pr
    N = param.N
    delta = param.delta
    
    if n_del_init >= ktools.edgeCnt(param):
        print('Fix n_del_init!')
        return param
    cnt_threshold = maxIter - maxIter*Pr
    cnt_threshold = math.ceil(cnt_threshold)
    param.n_del = n_del_init
    while param.n_del < ktools.edgeCnt(param):
        cnt = 0
        cnt_wrong = 0
        while cnt < maxIter:
            A = ktools.randomAs(param)
            output = KEDM(param, A)
            cvx_status = output.status
            if cvx_status == 'optimal':
                error_out = output.eo
                cnt += 1
                if ktools.mean(error_out) > 1-delta:
                    cnt_wrong = cnt_wrong + 1
                if maxIter - cnt < cnt_threshold - cnt_wrong:
                    break
                if cnt_wrong > cnt_threshold:
                    param.n_del -= 1
                    S = param.n_del/(N*(N-1)/2)
                    return param, S
        print('n_del is ', param.n_del)
        param.n_del += 1
    param.n_del -= 1
    S = param.n_del/(N*(N-1)/2)
    return param, S
###########################################################################
def TrjRtrv(param, kedm_output):
    N = param.N
    d = param.d
    P = param.P
    N_trn = param.N_trn
    mode = param.mode
    omega = param.omega
    K = param.K
    
    anchor_idx = ktools.randomAnchor(N_trn, N, d+1)
    
    tau_list = kedm_output.tau_list
    trn_list = kedm_output.trn_list
    tst_list = kedm_output.tst_list
    
    A = kedm_output.A
    G = kedm_output.G
    N_tst = param.N_tst
    output = TRJ_OUT()
    
    M = np.shape(anchor_idx)[0]
    if M != len(trn_list):
        print('Fix this!')
        return output
    
    X_ = []
    T = []
    cnt_wrong = 0
    cnt = 0
    for m in range(M):
        list_m = np.nonzero(anchor_idx[m,:])[0]
        if len(list_m) < d:
            cnt_wrong += 1
            continue
        t = trn_list[m]
        Xt = ktools.X_t(A,t,param)
        Xtm = Xt[:,list_m]
        weights = ktools.W(t, tau_list, param)
        Gt = ktools.G_t(G, weights, False)
        Xt_ = ktools.gram2x(Gt, d)
        Xt_m = Xt_[:,list_m]
        R = ktools.rotationXY(Xtm, Xt_m)
        Nt = Xtm.shape[1]
        Xt_ = np.matmul(R,Xt_ - np.matmul(Xt_m,np.ones((Nt,N))/Nt) ) + np.matmul(Xtm,np.ones((Nt,N))/Nt)
        Tt = ktools.T_t(t, param)
        if cnt == 0:
            X_ = Xt_
            T = Tt
        else:
            X_ = np.concatenate((X_, Xt_), axis=0)
            T = np.concatenate((T, Tt), axis=0)
        cnt += 1
    if cnt_wrong == M:
        print('nonsense')
        return output
    T_inv = np.linalg.pinv(T)
    A_ = np.matmul(T_inv,X_)
    A_ = ktools.deconcat(A_, param)


if mode == 2:
    A_ = ktools.sine2fourier(A_)
    
    eX = np.zeros(N_tst)
    for i_t, t in enumerate(tst_list):
        X_ = ktools.X_t(A_,t,param)
        X = ktools.X_t(A,t,param)
        eX[i_t] = np.linalg.norm(X-X_,'fro')/np.linalg.norm(X,'fro')

    output.A_ = A_
    output.eX = eX

return output
###########################################################################
def PlotX(A, param, mode, xlim, ylim, name , color):
    
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    import math
    
    N = param.N
    d = param.d
    print(d)
    if d > 3 or d < 2:
        print('There is something wrong, dude!')
        return
    M = 1000
    J = np.eye(N) - np.ones((N,N))/N
    if mode == 'test':
        T_tst = param.T_tst
        T_trn = param.T_trn
        T1 = T_trn[0]- T_tst[0]
        T2 = T_trn[1] - T_trn[0]
        T3 = T_tst[1] - T_trn[1]
        T = T1 + T2 + T3
        int(np.floor(N/2))
        M1 = int(np.floor(T1/T * M))
        M3 = int(np.floor(T3/T * M))
        M2 = M - M1 - M3
        t = np.linspace(T_tst[0],T_tst[1],M)
        dash=[8,3]
    else:
        T_tst = param.T_tst
        t = np.linspace(T_tst[0],T_tst[1],M)
    
    X = np.zeros((M,d,N))
    ax = plt.gca()
    ax.rc('text', usetex=True)
    ax.rc('font', family='serif')
    print(d)
    if d == 2:
        ax.xlabel(r'\textit{x}_1',fontsize=18)
        ax.ylabel(r'\textit{x}_2',fontsize=18)
    else:
        ax = ax.axes(projection='3d')
        ax.set_xlabel(r'\textit{x}_1',fontsize=fSize)
        ax.set_ylabel(r'\textit{x}_2',fontsize=fSize)
        ax.set_zlabel(r'\textit{x}_3',fontsize=fSize);
    
    for m in range(M):
        X[m,:,:] = ktools.X_t(A,t[m],param)
    for n in range(N):
        if mode == 'test':
            idx1 = list(range(0,M1,1))
            idx2 = list(range(M1,M1+M2,1))
            idx3 = list(range(M1+M2,M,1))
            if d==2:
                ax.plot(X[idx2,0,n], X[idx2,1,n],c = color[:,n],linewidth=2)
                ax.plot(X[idx1,0,n], X[idx1,1,n],c = np.zeros((3)),dashes=dash,linewidth=0.5)
                ax.plot(X[idx3,0,n], X[idx3,1,n],c = np.zeros((3)),dashes=dash,linewidth=0.5)
            elif d==3:
                ax.plot3D(X[idx2,0,n], X[idx2,1,n],c = color[:,n],linewidth=2)
                ax.plot3D(X[idx1,0,n], X[idx1,1,n],c = np.zeros((3)),dashes=dash,linewidth=1)
                ax.plot3D(X[idx3,0,n], X[idx3,1,n],c = np.zeros((3)),dashes=dash,linewidth=1)
        else:
            if d==2:
                plt.plot(X[:,0,n], X[:,1,n],c = color[:,n],linewidth=2)
            elif d==3:
                ax.plot3D(X[:,0,n], X[:,1,n],c = colors[:,n],linewidth=2)
    if d==2:
        plt.ylim(ylim)
        plt.xlim(xlim)
    plt.savefig(name+'.eps')
    plt.show()


###########################################################################
def PlotErrors(kedm_output,file_name):
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    
    
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')
    #plt.xlabel(r'$t$',fontsize=18)
    #plt.ylabel(r'\textrm{Relative Error}',fontsize=15)
    
    cvx_status = kedm_output.status
    eo = kedm_output.eo
    ei = kedm_output.ei
    ax = plt.gca()
    if cvx_status == 'optimal':
        ax.plot(kedm_output.tst_list, eo,linewidth=2,color="blue")
        ax.scatter(kedm_output.trn_list, ei,linewidth=1,color="red")
        plt.savefig(file_name+'2',format='eps', dpi=1000)
        plt.ylim([0,0.05])
        plt.xlim([-0.01,1])
    #plt.legend(['$\hat{e}_D(t)$','$e_D(t)$'],fontsize=17)
    plt.savefig(file_name,format='eps', dpi=1000)
    plt.show()








###########################################################################
def SketchX(param, A,colors, figName):
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    
    N = param.N
    d = param.d
    if d > 3 or d < 2:
        print('There is something wrong, dude!')
        return
    M = 300
    maxIter = param.maxIter
    T_tst = param.T_tst
    t = np.linspace(T_tst[0],T_tst[1],M)
    
    
    fSize = 18
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    if d == 2:
        plt.xlabel(r'\textit{x}_1',fontsize=18)
        plt.ylabel(r'\textit{x}_2',fontsize=18)
    else:
        ax = plt.axes(projection='3d')
        ax.set_xlabel(r'\textit{x}_1',fontsize=fSize)
        ax.set_ylabel(r'\textit{x}_2',fontsize=fSize)
        ax.set_zlabel(r'\textit{x}_3',fontsize=fSize);
    
    i = 0
    eX = 0
    eDo = 0
    eDi = 0
    while i < maxIter:
        kedm_output = KEDM(param, A)
        cvx_status = kedm_output.status
        if cvx_status == 'optimal':
            i += 1
            print(i,'out of', maxIter)
            trj_output = TrjRtrv(param, kedm_output)
            eX = eX + ktools.mean(trj_output.eX)/maxIter
            eDo = eDo + ktools.mean(kedm_output.eo)/maxIter
            eDi = eDi + ktools.mean(kedm_output.ei)/maxIter
            A_ = trj_output.A_
            X = np.zeros((M,d,N))
            for m in range(M):
                X[m,:,:] = ktools.X_t(A_,t[m],param)
            for n in range(N):
                if d == 2:
                    plt.plot(X[:,0,n], X[:,1,n],c = colors[:,n],linewidth=1/(maxIter**0.5))
                else:
                    ax.plot3D(X[:,0,n], X[:,1,n],X[:,2,n],c = colors[:,n],linewidth=1/(maxIter**0.5))
                plt.pause(0.01)

plt.savefig(param.path+figName)
plt.close()
return eDi, eDo, eX
###########################################################################
def Save(param, kedm_output, trj_output, mode):
    import os
    import datetime
    now = str(datetime.datetime.now())
    path = param.path+now+'/'
    if not os.path.exists(path):
        os.makedirs(path)
    if mode[0] == '1':
        np.save(path+'param',param)
    if mode[1] == '1':
        np.save(path+'kedm_output',kedm_output)
    if mode[2] == '1':
        np.save(path+'trj_output',trj_output)
###########################################################################
