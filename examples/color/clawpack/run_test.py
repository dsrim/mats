# FOM sol for test params

make_plots = 0

import numpy as np
import os
import setrun

rundata = setrun.setrun()

os.system('make .exe')

mu_test = np.loadtxt("../_output/mu_test.npy")
L = mu_test.shape[0]

xr = 2.0
xl = 0.0 

n = 14
print("running clawpack, num_cells = 2**{:d} = {:d}".format(n,2**n))
rundata.clawdata.num_cells = 2**n

dx = (xr - xl)/float(2**n)
nu0 = 0.5       # uniform CFL number for ROM
dt = nu0*dx
step = 100*dt   # output every 100 time-steps

rundata.clawdata.output_times = np.arange(0.0,200*step,step)
K = len(rundata.clawdata.output_times)

for l in range(L):
    data_array = np.zeros((2**n,K))
    
    mu = mu_test[l,:]

    rundata.probdata.mu1 = mu[0]
    rundata.probdata.mu2 = mu[1]
    rundata.probdata.mu3 = mu[2]
    rundata.probdata.mu4 = 0.2

    print(" {:2d} | mu1 = {:5.4f}, mu2 = {:5.4f}, mu3 = {:5.4f}".format(l,mu[0],mu[1],mu[2]))

    rundata.clawdata.output_style = 2

    # make .exe before running
    os.system('rm -f .data')
    rundata.write()
    os.system('make output >> output.log 2>> output.log')
    if make_plots:
        os.system('make plots >> plot.log 2>> output.log')
        os.system('mv _plots _plots_all/_plots_{:04d}'.format(l))
    
    # read in snapshots
    for k in np.arange(0,K):
        data_array[:,k] = \
            np.loadtxt('_output/fort.q{:04}'.format(k),skiprows=6)

    shot_fname = '../_output/fom_test_{:02}.npy'.format(l)
    np.save(shot_fname,data_array)

