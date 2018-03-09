# Hartree-Fock program for HeH+ from Szabo and Ostlund
import numpy as np
import math


def HF_calc(print_level, N, R, zeta_1, zeta_2):
    if print_level != 0:
        welcome_msg = "STO-" + \
            str(N) + \
            "G with Zetas {:6.4f} and {:6.4f}\n".format(zeta_1, zeta_2)
        print(welcome_msg)
    # Calculate core Hamiltonian and 2 electron integrals
    S, H_core, V_twoe = intgrl(print_level, N, R, zeta_1, zeta_2, ZA, ZB)
    # Diagonalize the overlap matrix
    X = diago(print_level, S)
    # Perform Self Consistent Field algorithm
    scf(print_level, N, R, zeta_1, zeta_2, ZA, ZB, S, H_core, V_twoe, X)

    return


def intgrl(print_level, N, R, zeta_1, zeta_2, ZA, ZB):
    # contraction coefficients for 1s Slater orbital with exponent 1.0 in terms
    # of normalized 1s primitive Gaussians. Row N corresponds to STO-NG
    c_coeff = np.array([[1.000000, 0.000000, 0.000000],
                        [0.678914, 0.430129, 0.000000],
                        [0.444635, 0.535328, 0.154329]])
    # corresponding exponents
    c_exp = np.array([[0.270950, 0.000000, 0.000000],
                      [0.151623, 0.851819, 0.000000],
                      [0.109818, 0.405771, 2.227660]])
    R2 = R**2

    # scale with Slater exponent to give the final exponents for each
    # primitive Gaussian which add up to a contracted Gaussian function of
    # exponent width zeta

    # We select the row N of the c_exp matrix subtracting 1 because Python
    # starts at 0
    A1 = c_exp[N - 1] * zeta_1**2
    A2 = c_exp[N - 1] * zeta_2**2

    # scale contraction coefficients such that a contracted Gaussian function
    # is a sum of exponent zeta is expressed by the sum over i of
    # D[i]*primitive Gaussian of exponent A[i]
    D1 = c_coeff[N - 1] * np.power((2 * A1 / np.pi), 0.75)
    D2 = c_coeff[N - 1] * np.power((2 * A2 / np.pi), 0.75)

    print("Primitive Gaussian exponents")
    print(A1)
    print("Contracted Gaussian contraction coefficients")
    print(D1)

    print("Printing preliminary integrals")
    # overlap matrix
    S = np.zeros((2, 2))

    # The core Hamiltonian is the kinetic energy T + the nuclear attraction V
    T = np.zeros((2, 2))
    V_nuc = np.zeros((2, 2))
    # V_nuc will be = V_nucA + V_nucB
    V_nucA = np.zeros((2, 2))
    V_nucB = np.zeros((2, 2))

    # here we will store the 2 electron integrals
    V_twoe = np.zeros((2, 2, 2, 2))
    # double sum because we are going to do one-electron integrals. In this
    # case we are integrating a product of two contracted Gaussian functions
    # which are made up of N primitive Gaussians. So the product will be a sum
    # of N*N products of primitive Gaussians which can be integrated
    # individually by S_overlap
    for i in range(N):
        for j in range(N):
            # Eq 3.228
            # Summing over all pairs of primitive Gaussians, the coefficients
            # of these primitive Gaussiansm the coefficients D of these
            # primitive Gaussians also need to be multiplied
            S[0][1] += S_overlap(A1[i], A2[j], R2) * D1[i] * D2[j]
            S[1][0] = S[0][1]
            # Since the contracted Gaussian functions are normalised, the
            # integral of their square is 1. These are the diagonal elements of
            # the overlap matrix
            S[0][0] = 1
            S[1][1] = 1

            # The kinetic energy matrix is:
            T[0][0] += T_kinetic(A1[i], A1[j], 0) * D1[i] * D1[j]
            T[0][1] += T_kinetic(A1[i], A2[j], R2) * D1[i] * D2[j]
            T[1][0] = T[0][1]
            T[1][1] += T_kinetic(A2[i], A2[j], 0) * D2[i] * D2[j]

            # The nuclear attraction:

            # First we need some info on the weighted centre of the two nuclei
            # r_ap or r_bp are distances between atoms a and b and weighted
            # centre between the two p. More info at Eq. A.4
            r_ap = A2[j] * R / (A1[i] + A2[j])
            r_ap2 = r_ap**2
            r_bp2 = (R - r_ap)**2

            # Now the matrix summing over nucleus A
            V_nucA[0][0] += V_nuclear(A1[i], A1[j], 0, 0, ZA) * D1[i] * D1[j]
            V_nucA[0][1] += V_nuclear(A1[i], A2[j],
                                      R2, r_ap2, ZA) * D1[i] * D2[j]
            # NB: here the penultimate arbument is the distance between the
            # weighted center of the two orbitals involved in the off-diagnoal
            # and nucleus A
            V_nucA[1][0] = V_nucA[0][1]
            V_nucA[1][1] += V_nuclear(A2[i], A2[j], 0, R2, ZA) * D2[i] * D2[j]
            # NB: now the weighted centre of the two orbitals is simply nucleus
            # B since this element is on the diagonal bottom right. So the
            # distance is just the distance AB

            # Similarly for the sum over nucleus B:
            V_nucB[0][0] += V_nuclear(A1[i], A1[j], 0, R2, ZB) * D1[i] * D1[j]
            V_nucB[0][1] += V_nuclear(A1[i], A2[j],
                                      R2, r_bp2, ZB) * D1[i] * D2[j]
            V_nucB[1][0] = V_nucB[0][1]
            V_nucB[1][1] += V_nuclear(A2[i], A2[j], 0, 0, ZB) * D2[i] * D2[j]

    V_nuc = V_nucA + V_nucB
    H_core = T + V_nuc

    # Now for the 2 electron integrals
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    # Again we need info on weighted centres
                    r_ap = A2[i] * R / (A2[i] + A1[j])
                    r_bp = R - r_ap
                    r_aq = A2[k] * R / (A2[k] + A1[l])
                    r_bq = R - r_aq
                    r_pq = r_ap - r_aq
                    r_ap2 = r_ap**2
                    r_bp2 = r_bp**2
                    r_aq2 = r_aq**2
                    r_bq2 = r_bq**2
                    r_pq2 = r_pq**2

                    # And now we make a 2-electron 2x2x2x2 tensor V_twoe_efgh=(ef|gh):
                    # But let's write down each element explicitly
                    V_twoe[0][0][0][
                        0] += two_e(A1[i], A1[j], A1[k], A1[l], 0, 0, 0) * D1[i] * D1[j] * D1[k] * D1[l]
                    V_twoe[1][0][0][
                        0] += two_e(A2[i], A1[j], A1[k], A1[l], R2, 0, r_ap2) * D2[i] * D1[j] * D1[k] * D1[l]
                    V_twoe[1][0][1][
                        0] += two_e(A2[i], A1[j], A2[k], A1[l], R2, R2, r_pq2) * D2[i] * D1[j] * D2[k] * D1[l]
                    V_twoe[1][1][0][
                        0] += two_e(A2[i], A2[j], A1[k], A1[l], 0, 0, R2) * D2[i] * D2[j] * D1[k] * D1[l]
                    V_twoe[1][1][1][
                        0] += two_e(A2[i], A2[j], A2[k], A1[l], 0, R2, r_bq2) * D2[i] * D2[j] * D2[k] * D1[l]
                    V_twoe[1][1][1][
                        1] += two_e(A2[i], A2[j], A2[k], A2[l], 0, 0, 0) * D2[i] * D2[j] * D2[k] * D2[l]

                    # permutation relations are explained in
                    # http://vergil.chemistry.gatech.edu/notes/permsymm/permsymm.pdf
                    V_twoe[0][1][0][0] = V_twoe[0][0][1][
                        0] = V_twoe[0][0][0][1] = V_twoe[1][0][0][0]
                    V_twoe[0][1][1][0] = V_twoe[0][1][0][
                        1] = V_twoe[1][0][0][1] = V_twoe[1][0][1][0]
                    V_twoe[0][0][1][1] = V_twoe[1][1][0][0]
                    V_twoe[0][1][1][1] = V_twoe[1][0][1][
                        1] = V_twoe[1][1][0][1] = V_twoe[1][1][1][0]

    if print_level != 0:
        print("{:10}{:10}{:10}".format("R", "ZETA1", "ZETA2"))
        print("{:<10}{:<10}{:<10}\n".format(R, zeta_1, zeta_2))
        print("Overlap matrix S")
        print(S)
        print()
        print("Kinetic energy matrix T")
        print(T)
        print()
        print("Nuclear attraction energy V_nucA due to A")
        print(V_nucA)
        print("Nuclear attraction energy V_nucB due to B")
        print(V_nucB)
        print("Nuclear attraction energy V_nuc")
        print(V_nuc)
        print()
        print("Core Hamiltonian H_core = T + V_nuc")
        print(H_core)
        print()
        print("Two electron integral tensor")
        print(V_twoe)
        print()

        return S, H_core, V_twoe


def S_overlap(a, b, r_ab2):
    """
    Calculate overlaps for un-normalized primitive Gaussians

    Eq 3.136
    Eq A.9
    """
    out_overlap = ((np.pi / (a + b))**(1.5)) * np.exp(-a * b * r_ab2 / (a + b))

    return out_overlap


def T_kinetic(a, b, r_ab2):
    """
    Calculate the kinetic energy integral of un-normalized primitive Gaussians

    Eq 3.151
    Eq. A.11
    Similar to above but because it's a kinetic energy there is a double
    differential inside the bra-ket. As a result the primitive Gaussian is
    differentiated twice and a prefactor appears.

    """
    out_kin = a * b / (a + b) * (3 - 2 * a * b * r_ab2 / (a + b)) * \
        ((np.pi / (a + b))**1.5) * (np.exp(-a * b * r_ab2 / (a + b)))

    return out_kin


def V_nuclear(a, b, r_ab2, r_cp2, z_c):
    """
    Calculate the nuclear attraction integrals for the un-normalized primitive
    Gaussians for one nucleus

    Add sum this over all nuclei in the sum of Eq. 152 to get the integral for
    one pair of primitive Gaussians
    Eq. 3.152
    Eq. A.33
    This is trickier than the last two integrals because now there is a 1/|r-R|
    along with the primitive Gaussians in the product to be integrated. The
    appendix solves it via Fourier transforms.
    r_ab2 is the squared distance between the two atoms whereas r_cp2 is the
    squared distance between the weighted center of the two primitive Gaussians
    discussed Eq.A.4 and the current center of the sum over nuclei.

    """
    out_vnuc = 2 * np.pi / (a + b) * f_0((a + b) * r_cp2) * \
        np.exp(-a * b * r_ab2 / (a + b))
    out_vnuc = -out_vnuc * z_c

    return out_vnuc


def f_0(arg):
    """
    Modified error function used for calculation of nuclear attraction integrals
    and 2 electron integrals

    """
    if arg < 10e-6:
        f_out = 1 - arg / 3
    else:
        f_out = np.sqrt(np.pi / arg) * math.erf(np.sqrt(arg)) / 2

    return f_out


def two_e(a, b, c, d, r_ab2, r_cd2, r_pq2):
    """
    Calculate the two electron integrals

    This uses Eq.3.155 and Ea.A.41. Now the weighted center Q corresponds to
    A and B and the weighted centre Q corresponds to C and D

    """

    two_out = 2 * (np.pi**2.5) / ((a + b) * (c + d) * np.sqrt(a + b + c + d)) * f_0((a + b) * (c + d)
                                                                                    * r_pq2 / (a + b + c + d)) * np.exp(-a * b * r_ab2 / (a + b) - c * d * r_cd2 / (c + d))

    return two_out


def derf(arg):
    """
    Compute error function from Handbook of Mathematical functions. This gives
    the same results as math.erf

    """

    num = 0.3275911
    params = [0.254829592, -0.284496736,
              1.421413741, -1.453152027, 1.061405429]

    t = 1 / (1 + num * arg)
    tn = t
    poly = params[0] * tn
    for i in params[1:]:
        tn *= t
        poly += +i * tn
    out_val = 1 - poly * np.exp(-arg**2)

    return out_val


def diago(print_level, S_in):
    """
    This function diagonalizes the overlap matrix via Eq. 3.259-3.262
    Full discussion of canonical diagonalization at Eq. 3.169 onwards

    """
    X_out = np.zeros((2, 2))
    X_out[0][0] = 1 / np.sqrt(2 * (1 + S_in[0][1]))
    X_out[1][0] = X_out[0][0]
    X_out[0][1] = 1 / np.sqrt(2 * (1 - S_in[0][1]))
    X_out[1][1] = -X_out[0][1]

    if print_level != 0:
        print("Diagonalizing matrix")
        print(X_out)
        print()
    return X_out


def scf(print_level, N, R, zeta_1, zeta_2, ZA, ZB, S, H_core, V_twoe, X):
    # The Fock matrix
    F = np.zeros((2, 2))
    # The two electron part of the Fock matrix so that F = H_core + G Eq. 3.154
    G = np.zeros((2, 2))
    # The expansion coefficients matrix Eq. 3.133 where C_ij is the coefficient
    # of the molecular orbital i for the basis function j
    C = np.zeros((2, 2))
    # The above matrix transformed via X to get the orthogonal Roothaan equation
    # with a transformed basis Eq. 3.173 and Fock matrix Eq. 3.177 for a
    # Roothaan equation Eq. 3.178
    C_prime = np.zeros((2, 2))
    # Density matrix Eq. 3.145
    P = np.zeros((2, 2))
    # And the density matrix from the last loop
    old_P = np.zeros((2, 2))
    # Convergence criterion for the density matrix
    crit = 10e-4
    # Density matrix difference to be checked (start with a value above crit)
    delta = 1
    # Maximum number of iterations
    max_iter = 25
    # iteration number
    it = 0

    while delta > crit:
        it += 1
        # Our initial guess at the density matrix is the null matrix
        if print_level == 3:

            print("Density matrix P")
            print(P)
            print()
            print("Begin iteration " + str(it))
            print()
        # 2 electron part of the Fock matrix
        G = formg(P, V_twoe)
        if print_level == 3:
            print("G, the 2 electron part of the Fock matrix")
            print(G)
            print()
        # Fock matrix
        F = H_core + G

        hf_energy = 0
        for i in range(2):
            for j in range(2):
                hf_energy += 0.5 * P[i][j] * (H_core[i][j] + F[i][j])

        if print_level == 3:
            print("Fock matrix F")
            print(F)
            print()
            print("Electronic energy = " + str(hf_energy) + "\n")

        # Transform the Fock matrix with X with Eq. 3.266 F_prime = X.T * F * X
        F_prime = X.T.dot(F.dot(X))

        # Diagonalize the new Fock matrix
        E,C_prime = np.linalg.eig(F_prime)

        # Transform C_prime into C from Eq. 3.174
        C = X.T.dot(C_prime)

        # Save old density matrix
        old_P = P

        # Make a new density matrix
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    P[i][j] += 2 * C[i][k] * C[j][k]
        print(P)

        break  # For testing REMOVE AT THE END


def formg(P, V_twoe):
    """
    Calculate the 2 electron part of the Fock matrix from Eq. 3.154

    """
    G_out = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    G_out[i][j] += P[k][l] * \
                        (V_twoe[i][j][k][l] - 0.5 * V_twoe[i][l][k][j])

    return G_out

    # Start loop
# STO-NG calc
N = 3
# interatomic distance
R = 1.4632
# exponent for He
zeta_1 = 2.0925
# exponent for H
zeta_2 = 1.24
# atomic number of He
ZA = 2
# atomic number of H
ZB = 1

HF_calc(3, N, R, zeta_1, zeta_2)
