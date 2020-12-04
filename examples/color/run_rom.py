# solve u_t + (c(x) u)_x = 0 using mats

import numpy as np
import time

n = 14
make_plots = 0
save_output = 1
expl = 1

if make_plots:
    import matplotlib.pyplot as plt
    plt.close("all")

vij = np.load("_output/dip0_lr_v.npy")
wgtij = np.load("_output/dip0_lr_w.npy")
U0 = np.load("_output/dip0_deim_modes.npy")
P0 = np.load("_output/dip0_deim_ipts.npy")

print("FOM dof = 2 ** {:02d}".format(n))
mu = np.load("_output/mu_snapshots.npy")
Nt = vij.shape[1]

if 1:
    Nh = int(U0.shape[1]/2)
    N00 = Nh + a
    N01 = Nh + b
    Nt = Nt + c 
    
    U00 = U0[:,:Nh]
    U01 = U0[:,Nh:]
    
    P00 = P0[:,:Nh]
    P01 = P0[:,Nh:]

    vij0 = vij[:Nh,:Nt]
    vij1 = vij[Nh:,:Nt]
    
    wgtij0 = wgtij[:Nh,:Nt]
    wgtij1 = wgtij[Nh:,:Nt]

    U00 = U00[:,:N00]
    U01 = U01[:,:N01]

    P00 = P00[:,:N00]
    P01 = P01[:,:N01]

    wgtij0 = wgtij0[:N00,:Nt]
    wgtij1 = wgtij1[:N01,:Nt]

    vij0 = vij0[:N00,:Nt]
    vij1 = vij1[:N01,:Nt]
    
    U0 = np.hstack((U00,U01))
    P0 = np.hstack((P00,P01))

    vij = np.vstack((vij0,vij1))
    wgtij = np.vstack((wgtij0,wgtij1))

N = U0.shape[0]
Nb = N00 + N01


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


# set the initial condition
if (expl == 0):
    L = 25
    mu_fname = "_output/mu_{:02d}.npy".format(n)
    mu = np.load(mu_fname)

elif (expl == 1):

    np.random.seed(10001)
    L = 10
    mu = np.random.rand(4,L)
    
    mu[0,:] = (1.0 + mu[0,:])*0.25
    mu[1,:] = (2.0 + 4.0*mu[1,:])*np.pi     # fast period
    mu[2,:] = (1.0 + 0.1*mu[2,:])*np.pi     # slow period

    musave_fname = "_output/mu_test.npy"
    np.savetxt(musave_fname,mu[:3,:].T)

PTU = np.vstack([U0[P0[:,j],:] for j in range(P0.shape[1])])

snapshot_fname = "_output/sol_snapshots.npy".format(n)
S = np.load(snapshot_fname)
s0 = S[:,0]
u0 = s0
u0i = np.dot(P0.T,u0)
coeff0 = np.linalg.solve(PTU,u0i)

for l in range(L):

    nu = 0.5    # courant no.
    nu0 = nu
    J = 24*100 # max. no time-steps 

    mu_c = mu[:4,l]
    xhi = np.linspace(0.5,N-0.5,N)
    xl,xr = (0.0,2.0)
    xh = (xhi/N)*(xr-xl) + xl
    h = xh[1]-xh[0]                 # uniform grid-width
    n = int(np.log2(N))

    def cs(mu,x=xh):
        mu1 = mu[0]
        mu2 = mu[1]
        mu3 = mu[2]
        
        cval = 1.5 + mu1*np.sin(mu2*x) + 0.1*np.cos(mu3*x)
        return cval

    
    # No. of DEIM dim
    M = P0.shape[1]

    # compute DEIM basis values at interp pts
    xipto = np.array([xh[P0[:,j]] for j in range(M)])    # orig. interp
    xipt0 = np.array([xh[P0[:,j]] for j in range(M)])    # orig. interp
    ipt0 = np.array([xhi[P0[:,j]] for j in range(M)])    # orig. interp

    # set up derivatives
    PTU2 = np.vstack([U0[P0[:,j],:]**2 for j in range(P0.shape[1])])

    ip = [i for i in range(1,N)] + [N-1]
    sU0 = U0[ip,:]
    Pp1TU = np.vstack([sU0[P0[:,j],:] for j in range(P0.shape[1])])
    im = [0] + [i for i in range(0,N-1)]
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

    xipt = xipt0.copy()

    tch = np.zeros(Nt)
    tn = 0.0
    Tp0_all = np.zeros((N,J))
    Tp1_all = np.zeros((N,J))

    r = np.ones((M,1))

    lind = np.argsort(np.abs(np.dot(PTDmU,coeff0)))[-Nt:]
    A = vij[lind,:]
    Q,R = np.linalg.qr(A)
    iwgts = np.ones(len(lind))

    if make_plots:
        import matplotlib.pyplot as plt
        plt.close("all")

        fig,ax = plt.subplots(ncols=1,nrows=1)
        ax.plot(xipt0,xipt0*0.0,"r.")
        ua0 = np.dot(U0,coeff0)
        ax.plot(xh,ua0)
        ax.plot(xh,s0,"k--")
        ax.set_title("initial condition")
        fig.show()
        
        plt.ion()
        fig,ax = plt.subplots(ncols=1,nrows=2,figsize=(8,5))
        ax[1].plot(xh,cs(mu_c))
    
    time0 = time.time()
    for j in range(1,J):

        wPTDmU  = PTDmU*r         # compute derivatives
        wPTDpU  = PTDpU*r         # compute derivatives
        
        #duval = np.dot(0.5*(wPTDpU + wPTDmU),coeffp)    # Lax-Friedrichs
        duval = np.dot(wPTDmU,coeffp)    # Lax-Friedrichs
        cx = cs(mu_c,x=xipt0.flatten())
        nflux = np.linalg.solve(PTU,cx*duval)  # numerical flux in DEIM coeff

        # time-step
        coeffn = coeffp - nu*nflux 
        
        dcoeff = -nflux
        dcoeff_all[:,j] = dcoeff.flatten()
        
        # estimate transport mode update
        T0i = -np.dot(PTU,coeffn-coeffp)
        T1i =  np.dot(wPTDmU,coeffp)
        T2i = np.dot(PTU,coeffp)
        rati = (T0i[lind]/T1i[lind]/nu)
        rati *= (nu)
        c = np.linalg.solve(R, np.dot(Q.T,rati)).flatten() 
        
        dtch = c
        ipt_new = ipt0.flatten() + np.dot(vij,tch+dtch)
        dxipt = ipt_new - ipt0.flatten()
        iwgts = 1.0 + np.dot(wgtij,tch+dtch)
        r = iwgts
        r = 1.0/r.reshape((-1,1))

        # update coefficients
        coeffp = coeffn.copy()

        dtch_all[:,j] = dtch.copy()
        tn += nu*h
        tch += dtch     

        xipt1 = ipt0.flatten() + np.dot(vij,tch)
        r = 1.0 + np.dot(wgtij,tch)
        xipt1 = xipt1/N*2.0
        r = 1.0/r.reshape((-1,1))

        if (make_plots and (j%500 == 0)):
            print("plotting")
            u1 = np.dot(U0,coeffp)
            ax[0].cla()
            ax[0].plot(xh,u1)
            ax[0].plot(xh,s0,"k--")
            ptu1 = np.dot(PTU,coeffp)
            ax[0].plot(xipt,xipt*0.0,"b.")
            ax[0].plot([xipt1, xipt1],[xipt1*0.0, ptu1],"-r.")
            ax[0].set_xlim([0.0,2.0])
            ax[0].set_title("j = {:04d}, t = {:1.4f}".format(j,tn))
            
            plt.pause(0.1)
            fig.canvas.draw()
            fig.canvas.flush_events()

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
        
        if (min_dist_xi < 0.1*h):
            break
        
    coeff1 = coeffn
    print('= done.')
    
    time1 = time.time()
    dtime = time1 - time0
    print("dtime = {:1.4f}, time-steps = {:08d}, per time-step = {:1.6f}".format(dtime,j,dtime/float(j)))

    J1 = j

    coeff_all = coeff_all[:,:(J1+1)]
    tch_all = tch_all[:,:(J1+1)]
    xipt_all = xipt_all[:,:(J1+1)]

    if save_output:
        np.savetxt("_output/dtime_{:02d}.txt".format(l),np.array([dtime]))
        np.savetxt("_output/coeff_{:02d}.txt".format(l),coeff_all)
        np.savetxt("_output/tch_{:02d}.txt".format(l),tch_all)
        np.savetxt("_output/xipt_{:02d}.txt".format(l),xipt_all)

