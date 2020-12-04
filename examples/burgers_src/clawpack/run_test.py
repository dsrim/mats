# compute FOM sol for test values of parameters mu

import numpy as np
import os,setrun

mu_test = np.loadtxt("../_output/mu_test.txt")	# parameters
ts_test = np.loadtxt("../_output/ts_test.txt") 	# no of time-steps

L = mu_test.shape[0]

r = 0
n = 14 + r
h = 10.0 / 2**n
nu = 0.5 
dt = h*nu

for l in range(0,L):

    print( "="*60 + "l={:d}".format(l))
    rundata = setrun.setrun()
    
    mu1 = mu_test[l,0] 
    mu2 = mu_test[l,1]
    nts = int(ts_test[l])       # no of time-steps
    
    rundata.clawdata.num_cells[0] = 2**n      # mx
    rundata.clawdata.order = 1
    
    rundata.clawdata.output_style = 1
    rundata.clawdata.num_output_times = nts
    rundata.clawdata.tfinal = nts*dt*(2**r)
    read_times = np.arange(1,nts)   # clawpack outputs one more file?
    
    rundata.probdata.mu1 = mu1
    rundata.probdata.mu2 = mu2

    data_list = [np.zeros(2**n)]*nts
    times_list = [0.0]*nts
    
    os.system('rm -f .data')
    rundata.write()
    os.system('touch .data')
    os.system('make output')
    for k in read_times:
        data = np.loadtxt('_output/fort.q{:04}'.format(k),skiprows=6)
        data_list[k] = data
        with open("_output/fort.t{:04}".format(k)) as infile:
            times = infile.readlines()
        time = float(times[0].split()[0])
        times_list[k] = time
    
    data_all = np.vstack(data_list).T
    np.save('../_output/fom_test_{:02}.npy'.format(l),data_all)
    np.save('../_output/fom_test_times_{:02}.npy'.format(l),times_list)

