# expl: burgers with src term
#
# run clawpack and collect snapshots 
# make .exe before running

make_plots = 0      # plots for each snapshot, save in _plots_all
save_shots = 1      # collect all snapshots, save in ../_output/

import numpy as np
import os,sys
import setrun
import matplotlib.pyplot as plt

rundata = setrun.setrun()

# problem size (number of cells)
n = 14
N = 2**n
rundata.clawdata.num_cells[0] = N

# uniformly space samples for snapshots
np.random.seed(24680)
L = 25
mu_all = np.zeros((L,2))
for l in range(L):
    mu1 = 100.0 + 50.0*np.random.rand(1) 
    mu2 = np.random.rand(1)*0.8 + 0.1
    print(" mu1, mu2 = {:1.4f}, {:1.4f} ".format(float(mu1),float(mu2)))
    mu_all[l,0] = mu1
    mu_all[l,1] = mu2

data_all = np.empty((N,0))

os.system("make .exe")

print("running clawpack..")
for l in range(L):

    # setrun parameters
    mu1 = mu_all[l,0]
    mu2 = mu_all[l,1]            
    
    msg = "\r {:04d} | mu1 = {:1.4f}, mu2 = {:1.4f}".format(l+1,mu1,mu2)
    sys.stdout.write(msg)
    sys.stdout.flush()
    
    rundata.probdata.mu1 = mu1
    rundata.probdata.mu2 = mu2
    
    # setrun times
    rundata.clawdata.output_style = 2
    t1 = 0.05
    ltimes = np.linspace(0.0,t1/mu1,6)[:5]      # local times
    gtimes = np.linspace(t1/mu1,50.0/mu1,10)    # global times
    times = np.hstack((ltimes,gtimes))
    rundata.clawdata.output_times = times

    # run clawpack
    os.system('rm -f .data')
    rundata.write()
    os.system('make output >> output.log 2>> output.log')
    
    if make_plots:
        os.system('make plots >> plot.log 2>> output.log')
        os.system('mv _plots _plots_all/_plots_{:04d}'.format(l))
    
    # read in clawpack output
    M = len(rundata.clawdata.output_times)
    for k in np.arange(0,M):
        data = np.loadtxt('_output/fort.q{:04}'.format(k),skiprows=6)
        data_all = np.hstack((data_all,data.reshape(-1,1)))

if save_shots:
    msg = "\r saving output... "
    sys.stdout.write(msg)
    sys.stdout.flush()
    np.save('../_output/sol_snapshots.npy'.format(n),data_all)
    np.save('../_output/mu_snapshots.npy'.format(n),mu_all)


msg = "\rdone = " + 60*" " + "\n"
sys.stdout.write(msg)
sys.stdout.flush()

# check signature condition
import dinterp

dip0 = dinterp.DIP()
dip0.set_data(data_all)
dip0.compute_pieces()
sgn_cond = dip0.check_signature()
print("signature condition: " + str(sgn_cond))

if make_plots:
    # :PLOT: show local snapshot variations
    f,ax = plt.subplots(nrows=1,ncols=1)
    for j in range(len(mu_all)):
        ax.plot(data_all[:,(j*15):(j*15 + 5)])
    f.show()
        
