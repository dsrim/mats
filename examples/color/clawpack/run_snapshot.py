# collect snapshots
import numpy as np
import os
import setrun
import matplotlib.pyplot as plt

make_solplots = 0   # plots for each snapshot, save in _plots_all
make_plots = 0      # plot collected snapshots
save_shots = 1      # collect all snapshots, save in ../_output/

np.random.seed(12345)
rundata = setrun.setrun()

# set times
t1 = 0.005
rundata.clawdata.output_times = \
                    np.hstack((np.linspace(0.0,t1,6)[:5],\
                               np.linspace(t1,1.0,10)))

os.system("make .exe")
if make_solplots:
    os.system("rm -rf _plots_all/*")

K = len(rundata.clawdata.output_times)
J = 25
n = 14
print("running clawpack, num_cells = 2**{:d} = {:d}".format(n,2**n))

data_all = np.zeros((2**n,J*K))
Mu = np.zeros((4,J))
rundata.clawdata.num_cells = 2**n

for j in range(J):
    mu = np.random.rand(4)

    mu[0] = (1.0 + mu[0])*0.25
    mu[1] = (2.0 + 4.0*mu[1])*np.pi     # fast osc
    mu[2] = (1.0 + 0.1*mu[2])*np.pi     # slow osc

    rundata.probdata.mu1 = mu[0]
    rundata.probdata.mu2 = mu[1]
    rundata.probdata.mu3 = mu[2]
    rundata.probdata.mu4 = mu[3]        # dummy var (unused)

    Mu[:,j] = mu
    print("j = {:2d} | mu1 = {:5.4f}, mu2 = {:5.4f}, mu3 = {:5.4f}".format(\
          j,mu[0],mu[1],mu[2]))

    rundata.clawdata.output_style = 2

    # make .exe before running
    os.system('rm -f .data')
    rundata.write()
    os.system('make output >> output.log 2>> output.log')
    if make_solplots:
        os.system('make plots >> plot.log 2>> output.log')
        os.system('mv _plots _plots_all/_plots_{:04d}'.format(j))
    
    # read in snapshots
    for k in np.arange(0,K):
        data = np.loadtxt('_output/fort.q{:04}'.format(k),skiprows=6)
        data_all[:,j*K + k] = data

if save_shots:
    shot_fname = '../_output/sol_snapshots.npy'
    mu_fname = '../_output/mu_snapshots.npy'
    np.save(shot_fname, data_all)
    np.save(mu_fname, Mu)

# check signature condition
import dinterp
dip0 = dinterp.DIP()
dip0.set_data(data_all)
dip0.compute_pieces()

print("signature condition: " + str(dip0.check_signature()))

if make_plots:
    # PLOT show local snapshot variations
    f,ax = plt.subplots(nrows=1,ncols=1)
    for j in range(J):
        ax.plot(data_all[:,(j*K):(j*K + 2)])
    f.show()
        
    # PLOT show local snapshot variations
    f,ax = plt.subplots(nrows=1,ncols=1)
    for j in range(J):
        ax.plot(data_all[:,(j*K):(j*K + 5)])
    f.show()

    local_data = np.hstack([data_all[:,(j*K):(j*K + 5)] for j in range(J)])
    U,s,V = np.linalg.svd(local_data)

