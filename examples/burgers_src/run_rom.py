# solve u_t + u*u_x = mu1*u*(1-u)*(u-mu2) using mats

import numpy as np
import time

norm = np.linalg.norm

vij = np.load("_output/dip0_lr_v.npy")
wgtij = np.load("_output/dip0_lr_w.npy")
U0 = np.load("_output/dip0_deim_modes.npy")
P0 = np.load("_output/dip0_deim_ipts.npy")

if 0:
    # preset basis
    Nb = 5      # no of DEIM basis <= 7
    Nt = 4      # no of transport modes <= 6

U0 = U0[:,:Nb]
P0 = P0[:,:Nb]

vij = vij[:Nb,:Nt]
wgtij = wgtij[:Nb,:Nt]

N = U0.shape[0]

vij = vij[:,:Nt]
wgtij = wgtij[:,:Nt]

def update_coeff(dxipt,PTDpU,PTDmU,PTU,coeffp0,h):
    r"""
    update DEIM coefficients coeffp0 

    """

    dxipt_p = np.abs(dxipt*(dxipt >= 0.))
    dxipt_m = np.abs(dxipt*(dxipt <  0.))
    
    dup = np.dot(PTDpU,coeffp0)/h
    dum = np.dot(PTDmU,coeffp0)/h 
        
    u0_ipt = np.dot(PTU,coeffp0)
    u0_ipt += (dxipt_p*dup.T).T
    u0_ipt -= (dxipt_m*dum.T).T

    coeffp = np.linalg.solve(PTU,u0_ipt)
    
    return coeffp

def psi(u,mu1=100.0,mu2=0.5):
    return -mu1*u*(u - mu2)*(1.0 - u)

np.random.seed(15)
expl = 2
l = 0

mu_all = np.zeros((10,2))
ts_all = np.zeros(10)

failflag = 0
for l in range(10):

    nuf = 1
    nu = 0.5/nuf    # courant no.
    nu0 = nu
    
    if (expl == 0):
        mu1 = 1.12
        mu2 = 0.32
    elif (expl == 1):
        mu1 = 2.93
        mu2 = 0.71
    elif (expl == 2):
        mu1 = 110.0 + 30.0*np.random.rand(1) 
        mu2 = np.random.rand(1)*0.8 + 0.1
        print(" mu1, mu2 = {:1.4f}, {:1.4f} ".format(float(mu1),float(mu2)))
        mu_all[l,0] = mu1
        mu_all[l,1] = mu2
    elif (expl == 3):
        # error-check
        l = 0
        mu_check = np.loadtxt("mu_check.txt")
        mu1 = mu_check[l,0]
        mu2 = mu_check[l,1]
    if (expl == 4):
        mu1 = 110.00
        mu2 = 0.7
    
    T = 10.0/mu1
    dt = 0.5*(10.0/2**14)
    J = int(np.round(T/dt))
    J *= nuf

    xhi = np.linspace(0.5,N-0.5,N)
    xl,xr = (-5.0,5.0)
    xh = (xhi/N)*(xr-xl) + xl
    h = xh[1]-xh[0]                 # uniform grid-width
    n = int(np.log2(N))

    PTU = np.vstack([U0[P0[:,j],:] for j in range(P0.shape[1])])

    # set the initial condition
    snapshot_fname = "_output/sol_snapshots.npy".format(n)
    S = np.load(snapshot_fname)
    u0 = S[:,0]
    u0i = np.dot(P0.T,u0)
    coeff0 = np.linalg.solve(PTU,u0i)

    # No. of DEIM dim
    M = P0.shape[1]

    # compute DEIM basis values at interp pts
    xipto = np.array([xh[P0[:,j]] for j in range(M)])    # orig. interp
    xipt0 = np.array([xh[P0[:,j]] for j in range(M)])    # orig. interp
    ipt0 = np.array([xhi[P0[:,j]] for j in range(M)])    # orig. interp

    # set up derivatives
    PTU2 = np.vstack([U0[P0[:,j],:]**2 for j in range(P0.shape[1])])

    ip = range(1,N) + [N-1]
    sU0 = U0[ip,:]
    Pp1TU = np.vstack([sU0[P0[:,j],:] for j in range(P0.shape[1])])
    im = [0] + range(0,N-1) 
    sU0 = U0[im,:]
    Pm1TU = np.vstack([sU0[P0[:,j],:] for j in range(P0.shape[1])])

    PTDpU = (Pp1TU - PTU)
    PTDmU = (PTU - Pm1TU)

    ioi = np.argsort(xipto.flatten()).flatten()

    coeffp = np.zeros(M)
    coeffn = np.zeros(M)
    coeffp = coeff0

    # store all time-steps
    coeff_all  = np.zeros((M,J))
    dcoeff_all = np.zeros((M,J))
    coeff_all[:,0] = coeff0.flatten()

    xipt_all = np.zeros((M,J))
    xipt_all[:,0] = xipt0.flatten()
    wgts_all = np.zeros((M,J))
    tch_all  = np.zeros((Nt,J))
    dtch_all = np.zeros((Nt,J))
    dtch0 = np.zeros(Nt)

    xipt = xipt0.copy()

    tch = np.zeros(Nt)
    tn = 0.0
    Tp0_all = np.zeros((N,J))
    Tp1_all = np.zeros((N,J))

    r = np.ones((M,1))

    lind = np.argsort(np.abs(np.dot(PTDmU,coeff0)))[::-1][:Nt]
    A = vij[lind,:]
    Q,R = np.linalg.qr(A)
    iwgts = np.ones(len(lind))
    time0 = time.time()
    for j in range(1,J):

        wPTDmU  = PTDmU *r         # compute derivatives
        
        u2val = 0.5*np.dot(PTU,coeffp)**2
        u2cof = np.linalg.solve(PTU,u2val)
        duval = np.dot(wPTDmU,u2cof)
        nflux = np.linalg.solve(PTU,duval)  # numerical flux in DEIM coeff

        # time-step
        coeffm = coeffp - nu*nflux 
        src = np.linalg.solve(PTU,psi(np.dot(PTU,coeffp),mu1=mu1,mu2=mu2))
        coeffn = coeffm - nu*h*src
        
        dcoeff = -nflux
        dcoeff_all[:,j] = dcoeff.flatten()

        # estimate transport mode update
        T0i = -np.dot(PTU, coeffn-coeffp)
        T1i =  np.dot(wPTDmU,coeffp)
        #lam = 0.1
        #T1i =  np.dot(wPTDmU,lam*coeffn + (1.0-lam)*coeffp)
        
        rati = T0i[lind]/T1i[lind]
        dtch  = np.linalg.solve(R*N, np.dot(Q.T,rati)).flatten()*N

        ipt_new = ipt0.flatten() + np.dot(vij,tch+dtch)
        dxipt = ipt_new - ipt0.flatten()
        iwgts = 1.0 + np.dot(wgtij,tch+dtch)
        r = iwgts
        r = 1.0/r.reshape((-1,1))

        # update coefficients
        coeffp = coeffn.copy()

        dtch_all[:,j] = dtch.copy()
        tn += nu
        tch += dtch     

        xipt1 = ipt0.flatten() + np.dot(vij,tch)
        r = 1.0 + np.dot(wgtij,tch)
        xipt1 = xipt1/N*10.0 - 5.0
        #if xipt1.shape[0] > 2:
        #    xipt1[2] = xipt0[2]
        r = 1.0/r.reshape((-1,1))

        if ((not (np.diff(xipt1[ioi]) > 0.0).all()) and (not failflag)):
            print("-------- crossing.. j = {:d}/{:d}".format(j,J))
            msg = "Nb{:02d}_Nt{:02d}_l{:02d}_j{:02d}\n".format(Nb,Nt,l,j)
            with open("fails.txt", mode="a") as outfile:
                outfile.write(msg)
            failflag = 1
            #break

        # update coordinates to the transported pts
        dxipt = xipt1.flatten() - xipt0.flatten() 
        coeffp0 = coeffp.copy()

        np.abs(xipt1.reshape(-1,1) - xipt0.reshape(1,-1)) < h

        wPTDmU = PTDmU*r       # compute derivatives
        wPTDpU = PTDpU*r       # compute derivatives
        coeffp = update_coeff(dxipt,wPTDpU,wPTDmU,PTU,coeffp0,h)
        
        coeff_all[:,j] = coeffp.flatten()
        wgts_all[:,j] = r.flatten()

        
        tch_all[:,j] = tch.copy()
        xipt_all[:,j] = xipt0.flatten().copy()

        xipt0 = xipt1.copy()

        dist_xi = np.abs(xipt0.reshape(1,-1) - xipt0.reshape(-1,1))
        min_dist_xi = np.min(dist_xi[dist_xi != 0.0])
        
        if ((min_dist_xi < 0.1*h) and (not failflag)):
            msg = "Nb{:02d}_Nt{:02d}_l{:02d}_j{:02d}\n".format(Nb,Nt,l,j)
            with open("fails.txt", mode="a") as outfile:
                outfile.write(msg)
            failflag = 1
        
    coeff1 = coeffn
    print('                                 = done.')
    
    time1 = time.time()
    dtime = time1 - time0
    print("dtime = {:1.4f}, time-steps = {:08d}".format(dtime,j))

    ts_all[l] = j
    J1 = j

    coeff_all = coeff_all[:,:(J1+1)]
    tch_all = tch_all[:,:(J1+1)]
    xipt_all = xipt_all[:,:(J1+1)]

    np.savetxt("_output/dtime_{:02d}.txt".format(l),np.array([dtime]))
    np.savetxt("_output/coeff_{:02d}.txt".format(l),coeff_all)
    np.savetxt("_output/tch_{:02d}.txt".format(l),tch_all)
    np.savetxt("_output/xipt_{:02d}.txt".format(l),xipt_all)

np.savetxt("_output/mu_test.txt", mu_all)
np.savetxt("_output/ts_test.txt", ts_all)

