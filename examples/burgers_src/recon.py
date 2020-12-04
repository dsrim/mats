# reconstruction of ROM solutions

remote = 0
make_plots = 0

### remove
expl = 2

import numpy as np
import os,sys
dinterp_path="/scratch/dr1653/dinterp-dev/build/lib.linux-x86_64-2.7"
if remote:
    import matplotlib
    matplotlib.use("Agg")
sys.path.insert(0,dinterp_path)
import dinterp
if make_plots:
    import matplotlib.pyplot as plt
    plt.ion()
    from matplotlib import cm
    plt.close("all")
    cm0 = cm.get_cmap("Blues")

snapshot_samples = np.load("_output/sol_snapshots.npy")

xl = -5.0
xr =  5.0
nuf = 1
step = nuf
N = 2**14   # resolution

## reconstruction 

L = 10      # no of test params

# load data
dip0 = dinterp.DIP()
dip0.load()
U0 = dip0._deim_modes

def fom_fname(l):
    return "_output/fom_test_{:02}.npy".format(l)

U0 = np.load("_output/dip0_deim_modes.npy")
P0 = np.load("_output/dip0_deim_ipts.npy")

U0 = U0[:,:Nb]
P0 = P0[:,:Nb]
PTU = np.dot(P0.T,U0)

os.system("rm _output/rom_test_??_??????.npy")
pts = np.linspace(0.5,N-0.5,N)
for l in range(L):
    
    coeff_all = np.loadtxt("_output/coeff_{:02d}.txt".format(l))
    tch_all = np.loadtxt("_output/tch_{:02d}.txt".format(l))
    xipt_all = np.loadtxt("_output/xipt_{:02d}.txt".format(l))

    K = coeff_all.shape[1]
    for k in range(0,K,step*10):
        sys.stdout.write("\r l = {:04d} | k = {:04d} ".format(l,k))
        sys.stdout.flush()

        coeff1 = coeff_all[:,k]
        if tch_all.ndim == 1:
            tch_all = tch_all.reshape(1,-1)
        tch1 = tch_all[:,k]
        xipt1 = xipt_all[:,k]

        tch2 = np.zeros(16)
        tch2[:Nt] = tch1
        xh1 = dip0.dinterp_lr_pts1(pts,tch2)
        xh1 = xh1/float(N)*(xr-xl) + xl
        
        u1 = np.dot(U0,coeff1)
        xhu1 = np.vstack((xh1.flatten(),u1.flatten()))

        np.save("_output/rom_test_{:02d}_{:06d}.npy".format(l,k),xhu1)



mu_test = np.loadtxt("_output/mu_test.txt")
l1rel_err = np.zeros(L)
for l in range(L):
    k_list = [0]
    error_list = []

    # PLOT overlayed sol
    xx = np.linspace(xl,xr,N+1)
    xh = 0.5*(xx[1:] + xx[:-1])
    
    xxr = np.linspace(xl,xr,N+1)
    xhr = 0.5*(xxr[1:] + xxr[:-1])

    # no of time-steps to take

    mu1 = mu_test[l,0]
    T = 10.0/mu1
    dt = 0.5*(10.0/2**14)
    K = int(np.round(T/dt))
    K *= nuf
    if make_plots:
        plt.ion()
        fig,ax = plt.subplots(ncols=1,nrows=1,figsize=(6.5,2))
        fig.show()

    # do some corrections for expl 2
    fom = np.load(fom_fname(l))
    fom0 = fom[:,1]
    
    for k0,k in enumerate(range(0,K-1,step)):
        if (not k0%10):
            fom = np.load(fom_fname(l))
            xhu_fname = "_output/rom_test_{:02d}_{:06d}.npy".format(l,k)
            
            sys.stdout.write("\r" + xhu_fname)
            sys.stdout.flush()

            tol = 1e-4
            
            xhu = np.load(xhu_fname)
            ii = (xhu[1,:] > 0.0 + tol)*(xhu[1,:] < 1.0 - tol)
            xhu1 = np.hstack(([xl],xhu[0,:][ii],[xr]))
            f1 = np.hstack(([1.0],xhu[1,:][ii],[0.0]))

            rom_gridr = np.interp(xhr,xhu1,f1,left=1.0,right=0.0)
            if (k0 > 2):
                fom_grid = fom[:,k0]
            else:
                fom_grid = fom0
            
            fom_gridr = fom_grid

            l1error = np.sum(np.abs(rom_gridr - fom_gridr))\
                     /np.sum(np.abs(fom_gridr))
            error_list.append(l1error)
            
            k_list.append(k0)
            
            if make_plots:
                #line0,=ax.plot(xhu1,f1,color=cm0(0.5 + 0.5*float(k0)/float(K)))
                line0,=ax.plot(xhu[0,:],xhu[1,:],\
                               color=cm0(0.5 + 0.5*float(k0)/float(K)))
                line1, = ax.plot(xh,fom_grid,"k--")
                ax.set_xlim([-5.0,5.0])
                fig.canvas.flush_events()
                ax.legend((line0,line1),("ROM","FOM"))
                plt.pause(0.001)

    coeff_all = np.loadtxt("_output/coeff_{:02d}.txt".format(l))
    tch_all = np.loadtxt("_output/tch_{:02d}.txt".format(l))
    xipt_all = np.loadtxt("_output/xipt_{:02d}.txt".format(l))
    
    PTUc = np.dot(PTU,coeff_all)
    
    if make_plots:
        for m in range(Nt):
            sys.stdout.write("\r l = {:02d}, m = {:04d}".format(l,m))
            sys.stdout.flush()

            xipt1 = xipt_all[m,k_list]
            PTU1 = PTUc[m,k_list]
            ax.plot(xipt1,PTU1,marker=".")
        ax.set_xlim([xl,xr])
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$u$")
        fig.tight_layout()
        fig.savefig("_plots/romvfom_{:02}.png".format(l),dpi=300)
        fig.show()

        plt.close('all')

    mean_err = np.mean(error_list)
    l1rel_err[l] = mean_err

    print("avg error = {:1.4e} / {:1.4e}".format(mean_err,mu_test[l,0]))

    np.savetxt("_output/l1rel_error_{:02d}_{:02d}.txt".format(Nb,Nt),\
               l1rel_err)

