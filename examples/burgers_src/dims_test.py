import numpy as np
import os

NT, NB = np.meshgrid(np.arange(2,7), np.arange(2,8))
Ntb = np.vstack((NT.flatten(),NB.flatten()))

no_list = []

os.system("rm fails.txt")

for j in range(Ntb.shape[1]):
    Nt = Ntb[0,j]
    Nb = Ntb[1,j]
    if (Nt <= Nb) and ((Nb,Nt) not in no_list):
        print("========= Nb = {:d}, Nt = {:d} ============".format(Nb,Nt))
        execfile("run_rom.py")
        execfile("recon.py")
