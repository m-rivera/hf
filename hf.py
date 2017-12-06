#Hartree-Fock program for HeH+ from Szabo and Ostlund
import numpy as np

def HF_calc(print_level, N, R, zeta_1, zeta_2):
    if print_level != 0:
        welcome_msg = "STO-"+str(N)+"G with Zetas {:6.4f} and {:6.4f}\n".format(zeta_1,zeta_2)
        print(welcome_msg)

    intgrl(print_level,N,R,zeta_1,zeta_2,ZA,ZB)

    return
def intgrl(print_level,N,R,zeta_1,zeta_2,ZA,ZB):
    # contraction coefficients for 1s Slater orbital with exponent 1.0 in terms
    # of normalized 1s primitive Gaussians
    c_coeff = np.mat([[1.000000,0.000000,0.000000],
                      [0.678914,0.430129,0.000000],
                      [0.444635,0.535328,0.154329]])
    # corresponding exponents
    c_exp = np.mat([[0.270950,0.000000,0.000000],
                    [0.151623,0.851819,0.000000],
                    [0.109818,0.405771,2.227660]])
    R2=R**2

    # scale with Slater exponent
    A1 = c_exp * zeta_1**2
    A2 = c_exp * zeta_2**2
# STO-NG calc
N=3
# interatomic distance
R=1.4632
# exponent for He
zeta_1 = 2.0925
# exponent for H
zeta_2 = 1.24
# atomic number of He
ZA = 2
# atomic number of He
ZB = 1

HF_calc(3, N, R, zeta_1, zeta_2)


