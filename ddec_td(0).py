import numpy as np;
import matplotlib.pyplot as plt
import myGraph
import os
gamma=0.8;
p=10;
ns=100;
ls=20;
M=30
global A

alg = 'dec'
# alg = 'ddec'

np.random.seed(3)
A=np.random.random_sample((p,ls));
#s=np.random.random_sample((ls,1))
def phi(s):
    aa=np.zeros((p,))
    aa=A@s;
    pphi=np.zeros((p,1));
    for i in range(0,p):
        pphi[i,0]=np.cos(aa[i])
    return pphi





S=10*np.random.random_sample((ls,ns));
R=10*np.random.random_sample((M,ns));
#R[0:int(M/3),:]=10*R[0:int(M/3),:]
P=np.zeros((ns,ns));
for i in range(0,ns):
    pp=np.random.random_sample((ns,))
    pp=pp/pp.sum()
    P[i,:]=pp

PP = np.zeros((ns, ns));
for i in range(0,ns):
    if i==0:
        PP[:,i]=P[:,i]
    if i>=1:
        PP[:,i]=PP[:,i-1]+P[:,i]


# W=np.eye((M))*0.3
if alg == 'dec':
    Nm=int(M/5)
else:
    Nm=int(M/5)
# for m in range(0,M):
#     # for _ in range(Nm):
#     #     while True:
#     #         mp=np.random.randint(0,M)
#     #         if mp == m:
#     #             continue
#     #         else:
#     #             break
#     #     W[m,mp]=np.random.random()
#     mp=np.random.randint(0,M)
#     W[m,mp]=np.random.random()


# #W=np.random.random_sample((M,M))


# u, s, vh = np.linalg.svd(W, full_matrices=False)
# W = u @ vh
# W=np.power(W,2)
# W=(W+W.T)/2
# mode = 0
# while True:
#     if mode % 2 == 0:
#         W = W / np.sum(W, axis=0, keepdims=True)
#     else:
#         W = W / np.sum(W, axis=1, keepdims=True)
#     mode = (mode+1) % 2
#     if min(np.sum(W, 0)) > 1.0 -1e-7 and min(np.sum(W, 1)) > 1.0 -1e-7:
#         break

W = myGraph.random_W(M, Nm)

if alg == 'ddec':
    sto_type = 0
    W = myGraph.directed_W(M, sto_type, Nm)

sk=0;

#Theta=np.zeros((M,p))
#for m in range(0,M):
#    Theta[m,:]=5*m*np.random.random_sample((1,p))
Theta=10*np.random.random_sample((M,p));

iter=2000;TTheta=np.zeros((M,p,iter))
atheta=np.zeros((p,iter))
natheta=np.zeros((iter))
alpha=0.01;
normt=np.zeros((M,iter))
Pt=P

Zk = np.zeros((M, p))
Xk  = np.copy(Theta)
Bk = np.ones(M)
Ak = np.eye(M)

for k in range(0,iter):
    Pt=Pt@P

    st=np.random.random();
    i=0
    while (i<ns):
        if st>PP[sk,i]:
            i=i+1
        if st<=PP[sk,i]:
            break;

    skp=i;
    r=np.zeros((M,1))
    for m in range(0,M):
        r[m,0]=R[m,skp]
    Hk=phi(S[:,sk])@( gamma*phi(S[:,skp]).T- phi(S[:,sk]).T)
    bk=r@phi(S[:,sk]).T;
    
    
    Gk=Theta@Hk.T+r@phi(S[:,sk]).T
    
    Theta=W@Theta+alpha*Gk;
    
    ####### with gradient push v2 ######
    if alg == 'ddec':
        Xk_temp = Xk + alpha * Gk
        Xk = W@Xk_temp
        Bk = W@Bk
        Theta = (Xk.T/Bk).T
    ####################################
    
    sk=skp
    TTheta[:,:,k]=Theta;
    a=1/M*Theta.T@np.ones((M,1))
    atheta[:, k]=a[:,0]
    natheta[k]=atheta[:,k].T@atheta[:,k]
    for m in range(0,M):
        normt[m,k]=Theta[m,:]@Theta[m,:].T




base_dir = './directed'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

plt.figure();
plt.rc('xtick',labelsize=20);plt.rc('ytick',labelsize=20)
plt.plot(normt[0,:],'b',lw=2.5);plt.plot(normt[1,:], 'r',linestyle='--',lw=2.5);
plt.plot(normt[2,:],'g',linestyle='-.',lw=2);plt.plot(normt[3,:],'m',linestyle=':',lw=2.5);


plt.savefig(os.path.join(base_dir, '{}_localave.eps'.format(alg)));
plt.savefig(os.path.join(base_dir, '{}_localave.pdf'.format(alg)));
plt.show();
# np.savetxt(os.path.join(base_dir, '{}_localave'.format(alg)), normt, fmt='%.8f')
# normt.tofile(os.path.join(base_dir, '{}_localave'.format(alg)), format='%.8f')
np.save(os.path.join(base_dir, '{}_localave.npy'.format(alg)), normt)


plt.figure();plt.plot(abs(TTheta[0,1,:]),'b',lw=2.5, label='$|\\theta_{1,1}|$');
plt.plot(abs(TTheta[1,1,:]),'r',linestyle='--',lw=2.5, label='$|\\theta_{2,1}|$');
plt.plot(abs(TTheta[2,1,:]),'g',linestyle='-.',lw=2, label='$|\\theta_{3,1}|$');
plt.plot(abs(TTheta[3,1,:]),'m',linestyle=':',lw=2.5, label='$|\\theta_{4,1}|$');
plt.savefig(os.path.join(base_dir, '{}_localele.eps'.format(alg)));
plt.savefig(os.path.join(base_dir, '{}_localele.pdf'.format(alg)))
plt.legend()
plt.show();
# np.savetxt(os.path.join(base_dir, '{}_localele'.format(alg)), TTtheta, fmt='%.8f')
# TTheta.tofile(os.path.join(base_dir, '{}_localele'.format(alg)), format='%.8f')
np.save(os.path.join(base_dir, '{}_localele.npy'.format(alg)), TTheta)

plt.figure();plt.plot(natheta,'r',lw=2.5);
plt.savefig(os.path.join(base_dir, '{}_ave.eps'.format(alg)));
plt.savefig(os.path.join(base_dir, '{}_ave.pdf'.format(alg)))
plt.show()
# np.savetxt(os.path.join(base_dir, '{}_ave'.format(alg)), natheta, fmt='%.8f')
# natheta.tofile(os.path.join(base_dir, '{}_ave'.format(alg)), format='%.8f')
np.save(os.path.join(base_dir, '{}_ave.npy'.format(alg)), natheta)

