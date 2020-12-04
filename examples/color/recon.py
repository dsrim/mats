# reconstruction of ROM solutions
# run after run_rom.py

remote = 0
make_plots = 0
save_output = 1
make_recon = 1

### remove
expl = 2

import numpy as np
import sys
import dinterp

if make_plots:
    import matplotlib.pyplot as plt
    plt.ion()
    from matplotlib import cm
    plt.close("all")
    fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(6.5,2))


# total no. of solutions
step = 100
L = 10

xl = 0.0
xr = 2.0

# load data
dip0 = dinterp.DIP()
dip0.load()

if make_recon:
    pts = np.linspace(0.5,N-0.5,N)
    for l in range(L):
        coeff_all = np.loadtxt("_output/coeff_{:02d}.txt".format(l))
        tch_all = np.loadtxt("_output/tch_{:02d}.txt".format(l))
        xipt_all = np.loadtxt("_output/xipt_{:02d}.txt".format(l))

        K = coeff_all.shape[1]
        for k in range(0,K,step):
            sys.stdout.write("\r l = {:04d} | k = {:06d} ".format(l,k))
            sys.stdout.flush()

            coeff1 = coeff_all[:,k]
            if tch_all.ndim == 1:
                tch_all = tch_all.reshape(1,-1)
            tch1 = tch_all[:,k]
            xipt1 = xipt_all[:,k]

            tch2 = np.zeros(4)
            tch2[:Nt] = tch1
            xh1 = dip0.dinterp_lr_pts1(pts,tch2)
            xh1 = xh1/float(N)*(xr-xl) + xl
            
            u1 = np.dot(U0,coeff1)
            xhu1 = np.vstack((xh1.flatten(),u1.flatten()))
            if save_output:
                np.save("_output/rom_test_{:02d}_{:06d}.npy".format(l,k),xhu1)


# color eqn
xl = 0.0
xr = 2.0
l0 = 200

def fom_fname(l):
    fname = expl_dir + "true_sol.npy"
    return fname

PTU = np.dot(P0.T,U0)
K = 24*step     # max time-step for checking error

relerr_all = np.zeros(L)

snapshot_samples = np.load("_output/sol_snapshots.npy")
for l in range(L):
    if make_plots:
        coeff_all = np.loadtxt("_output/coeff_{:02d}.txt".format(l))
        xipt_all = np.loadtxt("_output/xipt_{:02d}.txt".format(l))

    # PLOT for expl (color eqn)
    k_list = [0]
    error_list = []
    xx = np.linspace(xl,xr,N+1)
    xh = 0.5*(xx[1:] + xx[:-1])

    if make_plots:
        cm0 = cm.get_cmap("Blues")

    fom_fname = "_output/fom_test_{:02d}.npy".format(l)
    fom = np.load(fom_fname)
    for k0,k in enumerate(range(0,K-1,step)):
        
        xhu_fname = "_output/rom_test_{:02d}_{:06d}.npy".format(l,k)
        sys.stdout.write("\r" + xhu_fname)
        sys.stdout.flush()
        
        xhu = np.load(xhu_fname)
        color0 = 0.5 + 0.5*float(k0)/float(K)
        
        # plot ROM sol.
        xhu1 = xhu[0,:]
        
        fom_grid = fom[:,k0+1]
        
        
        # plot FOM sol.
        rom_grid = np.interp(xh,xhu1,xhu[1,:],left=0.0,right=0.0)
        

        l1error = np.sum(np.abs(rom_grid - fom_grid))\
                  /np.sum(np.abs(fom_grid))
        error_list.append(l1error)
        
        k_list.append(k0*100)

        if make_plots:
            plt.cla()
            coeff_all[:,k+1]
            xipt1 = xipt_all[:,k+1]
            u1 = np.dot(P0.T,np.dot(U0,coeff1))
            
            line0, = ax.plot(xhu1,xhu[1,:],color="r")
            line1, = ax.plot(xh,fom_grid,"k--")
            line2, = ax.plot(xipt1,u1,"b.")
            ax.set_xlim([0.0,2.0])
            ax.legend((line0,line1),("ROM","FOM","ipts"))
            ax.set_title("time = {:1.4f}".format(k*h*nu))
            
            fig.canvas.flush_events()
            ax.set_ylabel(r"$u$")
            ax.set_xlabel(r"$x$")
            ax.legend((line0,line1,line2),("ROM","FOM","ipts"))
            fig.tight_layout()
            fig.savefig("_plots/rom_plot_{:06d}.png".format(k),dpi=300)
            plt.pause(0.01)

        del(xhu)
        del(xhu1)

    if make_plots:
        # plot for the param-dep speed
        mu0 = mu[:,l]
        fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(6.5,2))
        cxh = 1.5 + mu0[0]*np.cos(mu0[1]*xh) + np.cos(mu0[2]*xh)
        ax.plot(xh,cxh)
        ax.set_title("speed $c(x)$")
        
        fig.canvas.flush_events()
        ax.set_ylabel(r"$c$")
        ax.set_xlabel(r"$x$")
        ax.set_xlim([0.0,2.0])
        
        fig.savefig("_plots/rom_speed_{:02d}.png".format(l),dpi=300)
        plt.pause(0.01)

        # PLOT trajectory
        fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(6.5,2))

        m = coeff_all.shape[0]
        for m in range(m):
            sys.stdout.write("\r l = {:02d}, m = {:04d}".format(l,m))
            sys.stdout.flush()

            xipt1 = xipt_all[m,:(k-1)]
            ax.plot(xipt1,np.arange(k-1))

        ax.set_ylabel("time-step")
        ax.set_xlabel("$x$")
        ax.set_xlim([xl,xr])
        fig.tight_layout()
        fig.show()
        fig.savefig("_plots/ipts_traj_{:02d}.png".format(l),dpi=300)


    relerr_all[l] = np.mean(error_list)
    print("  avg error = " + str(np.mean(error_list)))

np.savetxt("_output/l1rel_error_{:02d}_{:02d}.txt".format(Nb,Nt), relerr_all)

