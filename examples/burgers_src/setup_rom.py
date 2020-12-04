
make_plots  = 0

# imports
import numpy as np 
import matplotlib.pyplot as pl
import matplotlib.cm as plcm
import os,sys
import dinterp
import time

time0 = time.time()

n = 14
N = 2**14
print("FOM dof = 2 ** {:02d} = {:d}".format(n,N))

mu_fname = "_output/mu_snapshots.npy".format(n)
snapshot_fname = "_output/sol_snapshots.npy".format(n)
    
S = np.load(snapshot_fname)
mu = np.load(mu_fname)

dip0 = dinterp.DIP()
dip0.set_data(S)
dip0.compute_pieces()
dip0.compute_transport()
dip0.compute_lr_model(tol=0.9e-3)

# collect local snapshots
ind_list = []
for j in range(3):
    ind_list.append(np.arange(j,S.shape[1],15))
local_ind = np.vstack(ind_list).T.flatten()

Z = S[:,local_ind]
N = S.shape[0]

if make_plots:
    # PLOT: singvals for local snapshots
    f = pl.figure(figsize=(10,3))
    ax = f.add_subplot(1,1,1)
    ax.semilogy(s/s[0],'-rs')
    ax.semilogy(s[j0]/s[0],'-bs')
    f.show()

    # PLOT: DEIM basis and ipts
    f = pl.figure(figsize=(10,3))
    ax = f.add_subplot(1,1,1)
    ax.plot(U0)
    ii = np.arange(U0.shape[0])
    for j in range(P0.shape[1]):
        ii0 = ii[P0[:,j]]
        ax.plot([ii0,ii0],[0.0,U0[P0[:,j],j]],"-r.")
    f.show()

if make_plots:
    # PLOT: check taking derivatives
    def fd1(v):
        dv = np.zeros(v.shape[0]+1)
        dv[1:-1] = np.diff(v)
        return dv

    k0 = 0
    cp0 = np.linalg.solve(PTU,PTDpU[:,k0])
    cm0 = np.linalg.solve(PTU,PTDmU[:,k0])
    dvp0 = np.dot(U0,cp0)
    dvm0 = np.dot(U0,cm0)

    f = pl.figure(figsize=(10,5))
    ax0 = f.add_subplot(2,1,1)
    ax0.plot(xhi,dvp0)
    ax0.plot(xhi,dvm0)
    ax0.plot(xi,fd1(U0[:,k0]**2),"k--")
    ax0.set_title("7")
    
    ax1 = f.add_subplot(2,1,2,sharex=ax)
    ax1.plot(xhi,U0[:,k0]**2)
    ax1.set_title(r"$u^2$")
    f.tight_layout()
    f.show()

# squares (for flux)
Z1 = Z**2

Psi = np.zeros((N,3*mu.shape[0]))
l = 0
for k in range(mu.shape[0]):
    # mu1,mu2 : varnames used in clawpack setprob
    # mu1 : reaction coeff, mu2 : transition width
    mu1 = mu[k,0]
    mu2 = mu[k,1]
    for j in range(3):
        Zl = Z[:,l]
        Psi[:,l] = mu1*(Zl - 1.0)*(Zl - mu2)*Zl
        l += 1

if make_plots:
    # PLOT: source terms
    f = pl.figure(figsize=(10,6))
    ax0 = f.add_subplot(2,1,1)
    ax0.plot(Z)
    ax0.set_title(r"local snapshots $u$")
    ax1 = f.add_subplot(2,1,2,sharex=ax0)
    ax1.plot(Psi)
    ax1.set_title(r"source term $\psi(u)$")
    f.tight_layout()
    f.show()

Z = np.hstack((Z,Z1,Psi))

# translates (for derivatives)
ip = range(1,N) + [N-1]
im = [0] + range(0,N-1) 
Z = np.hstack((Z,Z[ip,:],Z[im,:],\
                 Z[ip,:][ip,:],Z[im,:][im,:],\
                 Z[ip,:][ip,:][ip,:],Z[im,:][im,:][im,:],\
                 Z[ip,:][ip,:][ip,:][ip,:],Z[im,:][im,:][im,:][im,:],\
                 Z[ip,:][ip,:][ip,:][ip,:][ip,:],Z[im,:][im,:][im,:][im,:]))

# compute POD basis 
U,s,V = np.linalg.svd(Z,full_matrices=False)
# pick basis by thresholding
j0 = (s/s[0] > 1.0e-4)
U0 = U[:,j0]
V0 = V[j0,:]

if make_plots:
    # .PLOT. local snapshots
    f = pl.figure(figsize=(10,3))
    ax = f.add_subplot(1,1,1)
    ax.plot(Z)
    f.show()
    
    # .PLOT. singvals for local
    f = pl.figure(figsize=(10,3))
    ax = f.add_subplot(1,1,1)
    ax.semilogy(s/s[0],'-rs')
    ax.semilogy(s[j0]/s[0],'-bs')
    f.show()

# compute deim ipts
_,P0 = dinterp.deim(U0)

if make_plots:
    # PLOT: DEIM basis and ipts
    cm0 = plcm.get_cmap("plasma")
    f = pl.figure(figsize=(10,3))
    ax = f.add_subplot(1,1,1)
    ax.plot(U0)
    ii = np.arange(U0.shape[0])
    for j in range(P0.shape[1]):
        color0 = cm0(float(j)/P0.shape[1])
        ii0 = ii[P0[:,j]]
        ax.plot([ii0,ii0],[0.0,U0[P0[:,j],j]],"-r.",color=color0)
    f.show()

    ax1 = f.add_subplot(2,1,2,sharex=ax)
    ax1.plot(xhi,u0**2)
    ax1.set_title(r"$u^2$")
    f.tight_layout()
    f.show()

# set up domain
xhi = np.linspace(0.5,N-0.5,N)
xi  = np.linspace(0.0,N,N+1)

# set up derivatives
PTU = np.vstack([U0[P0[:,j],:] for j in range(P0.shape[1])])
PTU2 = np.vstack([U0[P0[:,j],:]**2 for j in range(P0.shape[1])])

ip = range(1,N) + [N-1]
sU0 = U0[ip,:]
Pp1TU = np.vstack([sU0[P0[:,j],:] for j in range(P0.shape[1])])
im = [0] + range(0,N-1) 
sU0 = U0[im,:]
Pm1TU = np.vstack([sU0[P0[:,j],:] for j in range(P0.shape[1])])

PTDpU = (Pp1TU - PTU)
PTDmU = (PTU - Pm1TU)

# store DEIM data
dip0._deim_modes = U0
dip0._deim_coords = V0
dip0._deim_ipts = P0

# compute dinterp vectors at DEIM ipts
dip0.compute_dinterp_vectors()
dip0.save()


if make_plots:
    # .PLOT. computing source terms using DEIM

    # mu1 [1.0,5.0]
    # mu2 [0.1,0.9]

    def psi(u,mu1=1.0,mu2=0.5):
        return -mu1*u*(u - mu2)*(1.0 - u)

    mu1 = 1.0
    mu2 = 0.7

    k0 = 0
    c0 = np.linalg.solve(PTU,psi(PTU[:,k0],mu1=mu1,mu2=mu2))


    f = pl.figure(figsize=(10,5))
    def fd(v):
        dv = np.zeros(v.shape[0]+1)
        dv[1:-1] = np.diff(v)
        return dv

    ax0 = f.add_subplot(2,1,1)
    ax0.plot(xhi,np.dot(U0,c0))
    ax0.plot(xhi,psi(U0[:,k0],mu1=mu1,mu2=mu2),"k--")
    ax0.set_title(r"$\psi(u)$")
    
    cp0 = np.linalg.solve(PTU,np.dot(PTDpU,coeff1))
    cm0 = np.linalg.solve(PTU,np.dot(PTDmU,coeff1))
    dvp0 = np.dot(U0,cp0)
    dvm0 = np.dot(U0,cm0)

    f = pl.figure(figsize=(10,5))
    ax0 = f.add_subplot(2,1,1)
    ax0.plot(xhi,dvp0)
    ax0.plot(xhi,dvm0)
    ax0.plot(xi,fd(u0**2),"k--")
    ax0.set_title(r"$(u^2)_x$")
    
    ax1 = f.add_subplot(2,1,2,sharex=ax)
    ax1.plot(xhi,u0**2)
    ax1.set_title(r"$u^2$")
    f.tight_layout()
    f.show()

if make_plots:
    # mu1 [1.0,5.0]
    # mu2 [0.1,0.9]

    def psi(u,mu1=100.0,mu2=0.5):
        return -mu1*u*(u - mu2)*(1.0 - u)

    mu1 = 1.0
    mu2 = 0.7

    k0 = 0
    c0 = np.linalg.solve(PTU,psi(PTU[:,k0],mu1=mu1,mu2=mu2))

    if make_plots:
        # .PLOT. computing source terms
        f = pl.figure(figsize=(10,5))
        ax0 = f.add_subplot(2,1,1)
        ax0.plot(xhi,np.dot(U0,c0))
        ax0.plot(xhi,psi(U0[:,k0],mu1=mu1,mu2=mu2),"k--")
        ax0.set_title(r"$\psi(u)$")
        
        ax1 = f.add_subplot(2,1,2,sharex=ax)
        ax1.plot(xhi,U0[:,k0])
        ax1.set_title(r"$u$")
        f.tight_layout()
        f.show()

time1 = time.time()
offline_time = np.zeros(1)
offline_time[0] = time1 - time0

np.savetxt('offline_time.txt', offline_time)
