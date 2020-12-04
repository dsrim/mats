make_plots = 0

import numpy as np 
import os,sys
import matplotlib.pyplot as pl
import matplotlib.cm as plcm
import dinterp
import time


if make_plots:
    pl.close("all")

time0 = time.time() 

n = 14
print("FOM dof = 2 ** {:02d}".format(n))
snapshot_fname = "_output/sol_snapshots.npy"
mu_fname = "_output/mu_snapshots.npy"


S = np.load(snapshot_fname)
mu = np.load(mu_fname)

N = S.shape[0]

dip0 = dinterp.DIP()
dip0.set_data(S)
dip0.compute_pieces()
dip0.compute_transport()
dip0.compute_lr_model(tol=5.0e-3)       # compute SVD

# collect local snapshots
ind_list = []
mu_ind_list = []
for j in range(5):
    ind_list.append(np.arange(j,S.shape[1],15))
    mu_ind_list.append(np.arange(mu.shape[1]))

local_ind = np.vstack(ind_list).T.flatten()
mu_ind = np.vstack(mu_ind_list).T.flatten()

xhi = np.linspace(0.5,N-0.5,N)
xh = (xhi / N)*2.0

def cs(mu,x=xh):
    mu1 = mu[0]
    mu2 = mu[1]
    mu3 = mu[2]
    
    cval = 1.5 + mu1*np.sin(mu2*x) + 0.5*np.cos(mu3*x)
    return cval

Z_list = []
for k in range(len(dip0._signature[0])):
    if (dip0._signature[0][k] != 0):
        Z = []
        for l,j in enumerate(local_ind):
            df = dip0._get_snapshot_dpiece(j,k,nz_index=False)
            Z.append(np.cumsum(df)[1:])

        Z = np.array(Z).T
        Z_list.append(Z)


# include some modifications to the mix
if make_plots:
    # .PLOT. the source terms
    f = pl.figure(figsize=(10,6))
    ax0 = f.add_subplot(2,1,1)
    ax0.plot(Z)
    ax0.set_title(r"local snapshots $u$")
    ax1 = f.add_subplot(2,1,2,sharex=ax0)
    ax1.set_title(r"source term $\psi(u)$")
    f.tight_layout()
    f.show()

# translates (for derivatives)
ip = range(1,N) + [N-1]
im = [0] + range(0,N-1) 
for k,Z in enumerate(Z_list):
    shifts_list = []
    for l in range(Z_list[k].shape[1]): 
        cx = cs(mu[:,mu_ind[l]]) 
        Zl = Z_list[k][:,l]
        shifts_list.append((\
        cx*Zl,cx*Zl,\
        cx*Zl[ip],cx*Zl[im],\
        cx*Zl[ip][ip],cx*Zl[im][im],\
        cx*Zl[ip][ip][ip],cx*Zl[im][im][im],\
        cx*Zl[ip][ip][ip][ip],cx*Zl[im][im][im][im],\
        cx*Zl[ip][ip][ip][ip][ip],cx*Zl[im][im][im][im]))

        
U_list = []
V_list = []
for k,Z in enumerate(Z_list):
    U,s,V = np.linalg.svd(Z,full_matrices=False)
    if 0:
        # pick basis by thresholding
        j0 = (s/s[0] > 1.0e-4)
    if 1:
        # pick fixed no. of basis
        j0 = np.arange(7)
    U0 = U[:,j0]
    V0 = V[j0,:]
    U_list.append(U0)
    V_list.append(V0)

U0 = np.hstack(U_list)
V0 = np.hstack(V_list)

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

_,P0 = dinterp.deim(U0)

if make_plots:
    # .PLOT. DEIM basis and ipts
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

def fd1(v):
    dv = np.zeros(v.shape[0]+1)
    dv[1:-1] = np.diff(v)
    return dv

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

if make_plots:
    # .PLOT. taking derivatives
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

if make_plots:
    # .PLOT. taking derivatives (IC)
    k0 = 0
    u0 = Z[:,0]
    uval = np.dot(P0.T,u0)
    coeff0 = np.linalg.solve(PTU,uval)
    coeff1 = np.linalg.solve(PTU,uval**2)

    cp0 = np.linalg.solve(PTU,np.dot(PTDpU,coeff1))
    cm0 = np.linalg.solve(PTU,np.dot(PTDmU,coeff1))
    dvp0 = np.dot(U0,cp0)
    dvm0 = np.dot(U0,cm0)

    f = pl.figure(figsize=(10,5))
    ax0 = f.add_subplot(2,1,1)
    ax0.plot(xhi,dvp0)
    ax0.plot(xhi,dvm0)
    ax0.plot(xi,fd1(u0**2),"k--")
    ax0.set_title(r"$(u^2)_x$")
    
    ax1 = f.add_subplot(2,1,2,sharex=ax)
    ax1.plot(xhi,u0**2)
    ax1.set_title(r"$u^2$")
    f.tight_layout()
    f.show()

dip0._deim_modes = U0
dip0._deim_coords = V0
dip0._deim_ipts = P0
dip0.compute_dinterp_vectors()
dip0.save()

time1 = time.time() 
offline_time = np.zeros(1)
offline_time[0] = time1 - time0

np.savetxt('offline_time.txt', offline_time)
