#cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, cdivision=True
from libc.math cimport exp, pow, tgamma, sqrt, abs
from scipy.special.cython_special cimport hyp1f1

'''
A (hopefully) gentle guide to the computer implementation of molecular integrals - [https://joshuagoings.com/assets/integrals.pdf]
[https://gqcg-res.github.io/knowdes/the-mcmurchie-davidson-integral-scheme.html]
One- and two-electron integrals over cartesian gaussian functions - [https://www.sciencedirect.com/science/article/pii/002199917890092X]
'''

import numpy as np
cimport numpy as np

cdef double pi = 3.141592653589793238462643383279

#from integral - e
cdef double cye(int ia,int ja,int type, double r, double ie, double je, int n = 0, double x = 0.0):

    cdef:
        double p = ie + je
        double q = ie*je / p

    if n == 0:
        if (type < 0) or (type > (ia + ja)):
            return 0.0
        elif (ia + ja + type) == 0:
            return exp(-q*r*r)
        elif ja == 0:
            return ((1/(2 * p)) * cye(ia-1,ja,type-1,r,ie,je) - (q*r/ie) * cye(ia-1,ja,type,r,ie,je) +
                         (type+1) * cye(ia-1,ja,type+1,r,ie,je))
        else:
            return ((1/(2 * p)) * cye(ia,ja-1,type-1,r,ie,je) + (q*r/je) * cye(ia,ja-1,type,r,ie,je) +
                         (type+1) * cye(ia,ja-1,type+1,r,ie,je))
    else:
        return cye(ia+1,ja,type,r,ie,je,n-1,x) + x * cye(ia,ja,type,r,ie,je,n-1,x)

cdef double ovlp(int ia0, int ia1, int ia2, int ja0, int ja1, int ja2, int type, \
                           double r0, double r1, double r2, double ie, double je):
    cdef double s
    s =  cye(ia0, ja0, type, r0, ie, je)
    s *= cye(ia1, ja1, type, r1, ie, je)
    s *= cye(ia2, ja2, type, r2, ie, je)

    return s * pow(pi/(ie+je),1.5)

cdef double clmb(int l, int m, int n, int bf, double p, double r0, double r1, double r2):

    cdef double t, s, nm
    nm = sqrt(r0*r0 + r1*r1 + r2*r2)
    t = p * nm * nm

    s = 0.0
    if (l+m+n)  == 0:
        s += pow(-2*p, bf) * boys(bf, t)
    elif (l+m) == 0:
        if n > 1:
            s +=(n-1) * clmb(l,m,n-2,bf+1,p,r0,r1,r2)
        s += r2 * clmb(l,m,n-1,bf+1,p,r0,r1,r2)
    elif l == 0:
        if m > 1:
            s +=(m-1) * clmb(l,m-2,n,bf+1,p,r0,r1,r2)
        s += r1 * clmb(l,m-1,n,bf+1,p,r0,r1,r2)
    else:
        if l > 1:
            s +=(l-1) * clmb(l-2,m,n,bf+1,p,r0,r1,r2)
        s += r0 * clmb(l-1,m,n,bf+1,p,r0,r1,r2)

    return s


#boys function
cdef double boys(double m,double T):
    return hyp1f1(m+0.5,m+1.5,-T)/(2.0*m+1.0)

cdef double tei(int al0, int al1, int al2, int al3, short[:,:] aa, double[:,:] an, double[:,:] ac, \
                double[:,:] ae, double[:,:] ao, int i, int j, int k, int l):

    cdef:
        double s = 0.0
        int mu, nu, vu, su, tu, psi, phi, chi, alpha, beta, gamma
        double f, p, q, t1, s1, s2
        double t2[3]

    for mu in range(al0):
        for nu in range(al1):
            for vu in range(al2):
                for su in range(al3):
                    f =  an[i,mu] * an[j,nu] * an[k,vu] * an[l,su] * ac[i,mu] * ac[j,nu] * ac[k,vu] * ac[l,su]
                    p = ae[i,mu] + ae[j,nu]
                    q = ae[k,vu] + ae[l,su]
                    t1 = p*q/(p+q)
                    for tu in range(3):
                        t2[tu] = (ae[i,mu]*ao[i,tu] + ae[j,nu]*ao[j,tu])/p - (ae[k,vu]*ao[k,tu] + ae[l,su]*ao[l,tu])/q

                    s1 = 0.0
                    for psi in range(aa[i,0]+aa[j,0]+1):
                        for phi in range(aa[i,1]+aa[j,1]+1):
                            for chi in range(aa[i,2]+aa[j,2]+1):
                                for alpha in range(aa[k,0]+aa[l,0]+1):
                                    for beta in range(aa[k,1]+aa[l,1]+1):
                                        for gamma in range(aa[k,2]+aa[l,2]+1):

                                            s2 = (cye(aa[i,0],aa[j,0],psi, ao[i,0]-ao[j,0],ae[i,mu],ae[j,nu]) *
                                                  cye(aa[i,1],aa[j,1],phi, ao[i,1]-ao[j,1],ae[i,mu],ae[j,nu]) *
                                                  cye(aa[i,2],aa[j,2],chi, ao[i,2]-ao[j,2],ae[i,mu],ae[j,nu]))
                                            s2*= (cye(aa[k,0],aa[l,0],alpha, ao[k,0]-ao[l,0],ae[k,vu],ae[l,su]) *
                                                  cye(aa[k,1],aa[l,1],beta, ao[k,1]-ao[l,1],ae[k,vu],ae[l,su])  *
                                                  cye(aa[k,2],aa[l,2],gamma, ao[k,2]-ao[l,2],ae[k,vu],ae[l,su]))
                                            s2*= pow(-1, alpha+beta+gamma) * clmb(psi+alpha, phi+beta, chi+gamma, 0, t1, t2[0],t2[1],t2[2])

                                            s1 += s2
                    s1 *= 2 * pow(pi, 2.5) / ((p*q) * sqrt(p+q))
                    s += s1 * f
    return s

#|-------------------------------------dipole helper-------------------------------------|

cdef double mu(int[3] ia, int[3] ja, double ie, double je, double[3] ir, double[3] jr, double[3] kr, int direction):
    # dipole moment
    cdef:
        double p = ie + je
        double[3] q, ijr
        int i
        double u, v, t

    for i in range(3):
        q[i] = ((ie*ir[i] + je*jr[i])/p) - kr[i]
        ijr[i] = ir[i] - jr[i]

    if direction == 1:
        u = cye(ia[0], ja[0], 1, ijr[0], ie, je) + q[0]* cye(ia[0], ja[0], 0, ijr[0], ie, je)
        v = cye(ia[1], ja[1], 0, ijr[1], ie, je)
        t = cye(ia[2], ja[2], 0, ijr[2], ie, je)
        return u * v * t * pow(pi/p, 1.5)
    if direction == 2:
        u = cye(ia[0], ja[0], 0, ijr[0], ie, je)
        v = cye(ia[1], ja[1], 1, ijr[1], ie, je) + q[1]* cye(ia[1], ja[1], 0, ijr[1], ie, je)
        t = cye(ia[2], ja[2], 0, ijr[2], ie, je)
        return u * v * t * pow(pi/p, 1.5)
    if direction == 3:
        u = cye(ia[0], ja[0], 0, ijr[0], ie, je)
        v = cye(ia[1], ja[1], 0, ijr[1], ie, je)
        t = cye(ia[2], ja[2], 1, ijr[2], ie, je) + q[2]* cye(ia[2], ja[2], 0, ijr[2], ie, je)
        return u * v * t * pow(pi/p, 1.5)

#|---------------------------------end dipole helper-------------------------------------|
#|------------------------------------momentum helper------------------------------------|
cdef double ang(int[3] ia, int[3] ja, double ie, double je, double[3] ir, double[3] jr, double[3] kr, int direction):
    # angular momentum
    cdef:
        double p = ie + je
        double[3] ijr
        int i
        double u, v, t
        double sd[3][3]

    for i in range(3):
        ijr[i] = ir[i] - jr[i]

    for i in range(3):
        sd[0][i] = cye(ia[i], ja[i], 0, ijr[i], ie, je)
        sd[1][i] = cye(ia[i], ja[i], 0, ijr[i], ie, je, 1, ir[i]-kr[i])
        sd[2][i] = (ja[i] * cye(ia[i], ja[i]-1, 0, ijr[i], ie, je)) - (2.0 * je * cye(ia[i], ja[i]+1, 0, ijr[i], ie, je))

    if direction == 1:
        return -sd[0][0] * (sd[1][1] * sd[2][2] - sd[1][2] * sd[2][1]) * pow(pi/p, 1.5)
    elif direction == 2:
        return -sd[0][1] * (sd[1][2] * sd[2][0] - sd[1][0] * sd[2][2]) * pow(pi/p, 1.5)
    elif direction == 3:
        return -sd[0][2] * (sd[1][0] * sd[2][1] - sd[1][1] * sd[2][0]) * pow(pi/p, 1.5)

#|--------------------------------end momentum helper------------------------------------|
#|---------------------------------begin nabla helper------------------------------------|
cdef double nab(int[3] ia, int[3] ja, double ie, double je, double[3] ir, double[3] jr, int direction):
    #differential operator - nabla
    cdef:
        double p = ie + je
        double[3] ijr
        int i
        double u, v, t
        double sd[3]
        double dd[3]

    for i in range(3):
        ijr[i] = ir[i] - jr[i]

    for i in range(3):
        sd[i] = cye(ia[i], ja[i], 0, ijr[i], ie, je)
        dd[i] = ja[i] * cye(ia[i], ja[i]-1, 0, ijr[i], ie, je) - 2.0 * je * cye(ia[i], ja[i]+1, 0, ijr[i], ie, je)

    return dd[direction-1] * sd[direction % 3] * sd[(direction+1) % 3]  * pow(pi/p , 1.5)

#|----------------------------------end nabla helper-------------------------------------|
#|--------------------------------begin quadrupole helper--------------------------------|
cdef double qup(int[3] ia, int[3] ja, double ie, double je, double[3] ir, double[3] jr, double[3] kr, int direction):
    #quadrupole moment direction = 1-6, xx,yy,zz,xy,yz,zx
    cdef:
        double p = ie + je
        double[3] q, ijr
        int i
        double u
        double sd[3]
        double td[3]

    for i in range(3):
        q[i] = ((ie*ir[i] + je*jr[i])/p) - kr[i]
        ijr[i] = ir[i] - jr[i]

    for i in range(3):
        sd[i] = cye(ia[i], ja[i], 0, ijr[i], ie, je)
        td[i] = cye(ia[i], ja[i], 1, ijr[i], ie, je) + q[i] * cye(ia[i], ja[i], 0, ijr[i], ie, je)


    if direction <= 2:
        i = direction
        u = (2.0 * cye(ia[i], ja[i], 2, ijr[i], ie, je) + 2.0 * q[i] * cye(ia[i], ja[i], 1, ijr[i], ie, je) +
            (q[i]*q[i] + (0.5 / p)) * cye(ia[i], ja[i], 0, ijr[i], ie, je))
        return u * pow(pi/p, 1.5) * sd[(i+1)%3] * sd[(i+2)%3]
    else:
        i = direction - 3
        return td[(i)%3] * td[(i+1)%3] * sd[(i+2)%3] * pow(pi/p, 1.5)
#|---------------------------------end quadrupole helper---------------------------------|
#|------------------------------begin electric field helper------------------------------|
cdef double ele(int[3] ia, int[3] ja, double ie, double je, double[3] ir, double[3] jr, double[3] kr, int direction):
    #electric field
    cdef:
        double p = ie + je
        double[3] q, r
        int[3] ix = [0,0,0]
        double s
        int i, j, k

    for i in range(3):
        q[i] = ((ie*ir[i] + je*jr[i])/p)
        r[i] = q[i] - kr[i]

    if direction != 3: ix[direction] = 1

    s = 0.0
    for i in range(ia[0]+ja[0]+1):
        for j in range(ia[1]+ja[1]+1):
            for k in range(ia[2]+ja[2]+1):
                s += (cye(ia[0], ja[0], i, ir[0]-jr[0], ie, je) *
                      cye(ia[1], ja[1], j, ir[1]-jr[1], ie, je) *
                      cye(ia[2], ja[2], k, ir[2]-jr[2], ie, je) *
                      clmb(i+ix[0], j+ix[1], k+ix[2], 0, p, r[0], r[1], r[2]))

    return pow(-1, sum(ix)) * s * np.pi * 2.0 / p

#get the atom and basis classes
def aello(atom, basis, mode = 'scf', density = None, gauge = None):

    cdef:
        int na = len(atom)
        int nb = len(basis)
        int ng = len(basis[0].coefficients)
        int  i, j, k, l, m, n, p, q

    #get largest primative length
    for i in range(nb):
        j = len(basis[i].coefficients)
        if j > ng:
           ng = j

    #convert atom class properties to c views
    mx = np.empty([na,3], dtype = np.double)
    mz = np.empty([na], dtype   = np.short)
    cdef:
        double[:,:] alo_x = mx
        short[:]    alo_z = mz
    for p in range(na):
        for q in range(3):
            alo_x[p,q] = atom[p].center[q]
        alo_z[p] = atom[p].number

    #convert basis class properties to c-variables
    me = np.empty([nb,ng], dtype = np.double)
    mc = np.empty([nb,ng], dtype = np.double)
    mn = np.empty([nb,ng], dtype = np.double)
    ma = np.empty([nb,3],  dtype = np.short)
    mo = np.empty([nb,3],  dtype = np.double)
    ml = np.empty([nb],    dtype = np.short)

    cdef:
        double[:,:] alo_e = me
        double[:,:] alo_c = mc
        double[:,:] alo_n = mn
        short[:,:]  alo_a = ma
        double[:,:] alo_o = mo
        short[:]    alo   = ml

    for p in range(nb):
        alo[p] = len(basis[p].coefficients)
        for q in range(len(basis[p].coefficients)):
            alo_e[p,q] = basis[p].exponents[q]
            alo_c[p,q] = basis[p].coefficients[q]
            alo_n[p,q] = basis[p].normals[q]
        for q in range(3):
            alo_a[p,q] = basis[p].momenta[q]
            alo_o[p,q] = basis[p].atom.center[q]

    if mode == 'dipole':
        return aelloDipole(alo_n, alo_c, alo_e, alo_a, alo_o, alo, alo_z, alo_x, na, nb, gauge)
    elif mode == 'angular':
        return aelloAngular(alo_n, alo_c, alo_e, alo_a, alo_o, alo, alo_z, alo_x, na, nb, gauge)
    elif mode == 'nabla':
        return   aelloNabla(alo_n, alo_c, alo_e, alo_a, alo_o, alo, alo_z, alo_x, na, nb)
    elif mode == 'quadrupole':
        return aelloQuadrupole(alo_n, alo_c, alo_e, alo_a, alo_o, alo, alo_z, alo_x, na, nb, gauge)
    elif mode == 'electric field':
        return aelloElectricField(alo_n, alo_c, alo_e, alo_a, alo_o, alo, alo_z, alo_x, na, nb, gauge)
#-------------------------------------Begin Overlap---------------------------------------|
    S = np.empty([nb,nb], dtype = np.double)
    cdef:
        double [:,:] overlap  = S
        double s, f

    for p in range(nb):
        for q in range(p, nb):

            s = 0.0
            for i in range(alo[p]):
                for j in range(alo[q]):
                    f = alo_n[p,i] * alo_n[q,j] * alo_c[p,i] * alo_c[q,j]
                    s += (ovlp(alo_a[p,0], alo_a[p,1], alo_a[p,2], alo_a[q,0], alo_a[q,1], alo_a[q,2],
                          0 ,alo_o[p,0] - alo_o[q,0], alo_o[p,1] - alo_o[q,1], alo_o[p,2] - alo_o[q,2],
                          alo_e[p,i], alo_e[q,j]) * f)

            overlap[p,q] = s
            if p != q:
                overlap[q,p] = overlap[p,q]
#----------------------------------------End Overlap----------------------------------------|

#---------------------------------------Begin Kinetic---------------------------------------|
    K = np.empty([nb,nb], dtype = np.double)
    cdef:
        double[:,:] kinetic = K
        double t1, t2, t3

    for p in range(0, nb):
        for q in range(p, nb):

            s = 0.0
            for i in range(alo[p]):
                for j in range(alo[q]):
                    f = alo_n[p,i] * alo_n[q,j] * alo_c[p,i] * alo_c[q,j]
                    t1 = (alo_e[q,j] * (2*(alo_a[q,0] + alo_a[q,1] + alo_a[q,2]) + 3) *
                          ovlp(alo_a[p,0], alo_a[p,1], alo_a[p,2], alo_a[q,0], alo_a[q,1], alo_a[q,2],
                               0 ,alo_o[p,0] - alo_o[q,0], alo_o[p,1] - alo_o[q,1], alo_o[p,2] - alo_o[q,2],
                               alo_e[p,i], alo_e[q,j]) )

                    t2 = (-2 * alo_e[q,j] * alo_e[q,j] * (
                          ovlp(alo_a[p,0], alo_a[p,1], alo_a[p,2], alo_a[q,0]+2, alo_a[q,1], alo_a[q,2],
                               0 ,alo_o[p,0] - alo_o[q,0], alo_o[p,1] - alo_o[q,1], alo_o[p,2] - alo_o[q,2],
                               alo_e[p,i], alo_e[q,j])   +
                          ovlp(alo_a[p,0], alo_a[p,1], alo_a[p,2], alo_a[q,0], alo_a[q,1]+2, alo_a[q,2],
                               0 ,alo_o[p,0] - alo_o[q,0], alo_o[p,1] - alo_o[q,1], alo_o[p,2] - alo_o[q,2],
                               alo_e[p,i], alo_e[q,j])   +
                          ovlp(alo_a[p,0], alo_a[p,1], alo_a[p,2], alo_a[q,0], alo_a[q,1], alo_a[q,2]+2,
                               0 ,alo_o[p,0] - alo_o[q,0], alo_o[p,1] - alo_o[q,1], alo_o[p,2] - alo_o[q,2],
                               alo_e[p,i], alo_e[q,j]) ) )


                    t3 = (alo_a[q,0] * (alo_a[q,0] - 1) *
                          ovlp(alo_a[p,0], alo_a[p,1], alo_a[p,2], alo_a[q,0]-2, alo_a[q,1], alo_a[q,2],
                               0 ,alo_o[p,0] - alo_o[q,0], alo_o[p,1] - alo_o[q,1], alo_o[p,2] - alo_o[q,2],
                               alo_e[p,i], alo_e[q,j]))
                    t3 += (alo_a[q,1] * (alo_a[q,1] - 1) *
                           ovlp(alo_a[p,0], alo_a[p,1], alo_a[p,2], alo_a[q,0], alo_a[q,1]-2, alo_a[q,2],
                                0 ,alo_o[p,0] - alo_o[q,0], alo_o[p,1] - alo_o[q,1], alo_o[p,2] - alo_o[q,2],
                                alo_e[p,i], alo_e[q,j]))
                    t3 += (alo_a[q,2] * (alo_a[q,2] - 1) *
                           ovlp(alo_a[p,0], alo_a[p,1], alo_a[p,2], alo_a[q,0], alo_a[q,1], alo_a[q,2]-2,
                                0 ,alo_o[p,0] - alo_o[q,0], alo_o[p,1] - alo_o[q,1], alo_o[p,2] - alo_o[q,2],
                                alo_e[p,i], alo_e[q,j]))

                    s += (t1 + t2 - 0.5*t3) * f

            kinetic[p,q] = s
            if p != q:
                kinetic[q,p] = kinetic[p,q]
#----------------------------------------End Kinetic----------------------------------------|

#---------------------------------------Begin Coulomb---------------------------------------|
    J = np.empty([nb,nb], dtype = np.double)
    cdef:
        double[:,:] coulomb = J
        double r[3]
        double cp

    for p in range(nb):
        for q in range(p, nb):

            t1 = 0.0
            for k in range(na):

                s = 0.0
                for i in range(alo[p]):
                    for j in range(alo[q]):
                        f = alo_n[p,i] * alo_n[q,j] * alo_c[p,i] * alo_c[q,j]
                        cp = alo_e[p,i] + alo_e[q,j]
                        for n in range(3):
                            r[n] = ((alo_e[p,i] * alo_o[p,n]) + (alo_e[q,j] * alo_o[q,n]))/cp - alo_x[k,n]

                        t2 = 0.0
                        for l in range(alo_a[p,0]+alo_a[q,0]+1):
                            for m in range(alo_a[p,1]+alo_a[q,1]+1):
                                for n in range(alo_a[p,2]+alo_a[q,2]+1):
                                    t2 += (cye(alo_a[p,0], alo_a[q,0], l, alo_o[p,0]- alo_o[q,0], alo_e[p,i], alo_e[q,j]) *
                                           cye(alo_a[p,1], alo_a[q,1], m, alo_o[p,1]- alo_o[q,1], alo_e[p,i], alo_e[q,j]) *
                                           cye(alo_a[p,2], alo_a[q,2], n, alo_o[p,2]- alo_o[q,2], alo_e[p,i], alo_e[q,j]) *
                                           clmb(l, m, n, 0, cp, r[0], r[1], r[2]))

                        t2 = t2 * pi * 2.0 / cp
                        s += t2 * f
                t1 -= s * alo_z[k]
            coulomb[p,q] = t1
            if p != q:
                coulomb[q,p] = coulomb[p,q]

#----------------------------------------End Coulomb----------------------------------------|

#----------------------------------Begin electron repulsion---------------------------------|
    I = np.empty([nb,nb,nb,nb], dtype=np.double)
    cdef:
        double[:,:,:,:] eri = I

    for i in range(nb):
        for j in range(i+1):
            m = i * (i+1)/2 + j
            for k in range(nb):
                for l in range(k+1):
                    n = k*(k+1)/2 + l
                    if m >= n:
                        f = tei(alo[i], alo[j], alo[k], alo[l], alo_a, alo_n, alo_c, alo_e, alo_o, i, j, k, l)
                        I[i,j,k,l]=I[k,l,i,j]=I[j,i,l,k]=I[l,k,j,i]=I[j,i,k,l]=I[l,k,i,j]=I[i,j,l,k]=I[k,l,j,i] = f

#|----------------------------------End electron repulsion----------------------------------|

    return S, K, J, I

#---------------------------------------Begin Dipole----------------------------------------|
cpdef aelloDipole(double[:,:] alo_n, double[:,:] alo_c, double[:,:] alo_e, short[:,:] alo_a, double[:,:] alo_o, \
                        short[:] alo, short[:] alo_z, double[:,:] alo_x, int na, int nb, gauge):

    D = np.empty([3,nb,nb], dtype = np.double)
    cdef:
        double[:,:,:] dipole = D
        double[3] gaugeOrigin = gauge
        int direction, p, q, i, j
        double s, f
        double[3] dipoleComponent

    for direction in range(3):
        #electronic component
        for p in range(nb):
            for q in range(p, -1, -1):

                s = 0.0
                for i in range(alo[p]):
                    for j in range(alo[q]):
                        f = alo_n[p,i] * alo_n[q,j] * alo_c[p,i] * alo_c[q,j]
                        s += (mu([alo_a[p,0], alo_a[p,1], alo_a[p,2]], [alo_a[q,0], alo_a[q,1], alo_a[q,2]],
                                  alo_e[p,i], alo_e[q,j],
                                  [alo_o[p,0], alo_o[p,1], alo_o[p,2]], [alo_o[q,0], alo_o[q,1], alo_o[q,2]],
                                  gaugeOrigin, direction+1) * f)

                dipole[direction, p, q] = s
                if p != q:
                    dipole[direction,q,p] = dipole[direction,p,q]


    return D
#|-----------------------------------------End Dipole---------------------------------------|
#|--------------------------------------Begin Quadrupole------------------------------------|
cpdef aelloQuadrupole(double[:,:] alo_n, double[:,:] alo_c, double[:,:] alo_e, short[:,:] alo_a, double[:,:] alo_o, \
                        short[:] alo, short[:] alo_z, double[:,:] alo_x, int na, int nb, gauge):

    Q = np.empty([6,nb,nb], dtype = np.double)
    cdef:
        double[:,:,:] quadrupole = Q
        double[3] gaugeOrigin = gauge
        int direction, p, q, i, j

    for direction in range(6):

        #electronic component
        for p in range(nb):
            for q in range(p, -1, -1):

                s = 0.0
                for i in range(alo[p]):
                    for j in range(alo[q]):
                        f = alo_n[p,i] * alo_n[q,j] * alo_c[p,i] * alo_c[q,j]
                        s += (qup([alo_a[p,0], alo_a[p,1], alo_a[p,2]], [alo_a[q,0], alo_a[q,1], alo_a[q,2]],
                                   alo_e[p,i], alo_e[q,j],
                                   [alo_o[p,0], alo_o[p,1], alo_o[p,2]], [alo_o[q,0], alo_o[q,1], alo_o[q,2]],
                                   gaugeOrigin, direction) * f)

                quadrupole[direction,p, q] = s
                if p != q:
                    quadrupole[direction,q,p] = quadrupole[direction,p,q]

    return Q
#|----------------------------------------Begin Angular-------------------------------------|

cpdef aelloAngular(double[:,:] alo_n, double[:,:] alo_c, double[:,:] alo_e, short[:,:] alo_a, double[:,:] alo_o, \
                        short[:] alo, short[:] alo_z, double[:,:] alo_x, int na, int nb, gauge):

    A = np.empty([3,nb,nb], dtype = np.double)
    cdef:
        double[:,:,:] angular = A
        double[3] gaugeOrigin = gauge
        int direction, p, q, i, j
        double s, f

    for direction in range(3):
        #electronic component
        for p in range(nb):
            for q in range(p+1):

                s = 0.0
                for i in range(alo[p]):
                    for j in range(alo[q]):
                        f = alo_n[p,i] * alo_n[q,j] * alo_c[p,i] * alo_c[q,j]
                        s += (ang([alo_a[p,0], alo_a[p,1], alo_a[p,2]], [alo_a[q,0], alo_a[q,1], alo_a[q,2]],
                                  alo_e[p,i], alo_e[q,j],
                                  [alo_o[p,0], alo_o[p,1], alo_o[p,2]], [alo_o[q,0], alo_o[q,1], alo_o[q,2]],
                                  gaugeOrigin, direction+1) * f)

                angular[direction, p, q] = s
                if p != q:
                    angular[direction,q,p] = -angular[direction,p,q]

    return A

#|----------------------------------------End Angular---------------------------------------|
#|----------------------------------------Begin Nabla---------------------------------------|
cpdef aelloNabla(double[:,:] alo_n, double[:,:] alo_c, double[:,:] alo_e, short[:,:] alo_a, double[:,:] alo_o, \
                        short[:] alo, short[:] alo_z, double[:,:] alo_x, int na, int nb):

    N = np.empty([3,nb,nb], dtype = np.double)
    cdef:
        double[:,:,:] nabla = N
        int direction, p, q, i, j
        double s, f

    for direction in range(3):
        #electronic component
        for p in range(nb):
            for q in range(p+1):

                s = 0.0
                for i in range(alo[p]):
                    for j in range(alo[q]):
                        f = alo_n[p,i] * alo_n[q,j] * alo_c[p,i] * alo_c[q,j]
                        s += (nab([alo_a[p,0], alo_a[p,1], alo_a[p,2]], [alo_a[q,0], alo_a[q,1], alo_a[q,2]],
                                  alo_e[p,i], alo_e[q,j],
                                  [alo_o[p,0], alo_o[p,1], alo_o[p,2]], [alo_o[q,0], alo_o[q,1], alo_o[q,2]],
                                  direction+1) * f)

                nabla[direction, p, q] = s
                if p != q:
                    nabla[direction,q,p] = -nabla[direction,p,q]

    return N
#|-----------------------------------------End Nabla----------------------------------------|
#|------------------------------------Begin Electric Field----------------------------------|
cpdef aelloElectricField(double[:,:] alo_n, double[:,:] alo_c, double[:,:] alo_e, short[:,:] alo_a, double[:,:] alo_o, \
                        short[:] alo, short[:] alo_z, double[:,:] alo_x, int na, int nb, gauge):

    E = np.empty([4,nb,nb], dtype = np.double)
    cdef:
        double[:,:,:] electric_field = E
        double[3] gaugeOrigin = gauge
        int direction, p, q, i, j
        double s, f

    for direction in range(4):
        #electronic component
        for p in range(nb):
            for q in range(p+1):

                s = 0.0
                for i in range(alo[p]):
                    for j in range(alo[q]):

                        f = alo_n[p,i] * alo_n[q,j] * alo_c[p,i] * alo_c[q,j]
                        s += (ele([alo_a[p,0], alo_a[p,1], alo_a[p,2]], [alo_a[q,0], alo_a[q,1], alo_a[q,2]],
                                   alo_e[p,i], alo_e[q,j],
                                   [alo_o[p,0], alo_o[p,1], alo_o[p,2]], [alo_o[q,0], alo_o[q,1], alo_o[q,2]],
                                   gaugeOrigin, direction) * f)

                electric_field[direction, p, q] = s
                if p != q:
                    electric_field[direction,q,p] = electric_field[direction,p,q]

    return E
#|----------------------------------------Gradient Integrals--------------------------------|
def aello_dx(object atom, object basis, density, fock):

    cdef:
        int na = len(atom)
        int nb = len(basis)
        int ng = len(basis[0].coefficients)
        int i, j, k, l, m, n, p, q, r, s

#get largest primative length
    for i in range(nb):
        j = len(basis[i].coefficients)
        if j > ng:
            ng = j

    #convert atom class properties to c views
    mx = np.empty([na,3], dtype = np.double)
    mz = np.empty([na],   dtype = np.short)
    cdef:
        double[:,:] alo_x = mx
        short[:]   alo_z = mz
    for p in range(0, na):
        for q in range(0, 3):
            alo_x[p,q] = atom[p].center[q]
        alo_z[p] = atom[p].number

    #convert basis class properties to c-variables
    me = np.empty([nb,ng], dtype = np.double)
    mc = np.empty([nb,ng], dtype = np.double)
    mn = np.empty([nb,ng], dtype = np.double)
    ma = np.empty([nb,3],  dtype = np.short)
    mo = np.empty([nb,3],  dtype = np.double)
    ml = np.empty([nb],    dtype = np.short)
    mp = np.empty([nb],    dtype = np.short)

    cdef:
        double[:,:] alo_e = me
        double[:,:] alo_c = mc
        double[:,:] alo_n = mn
        short[:,:]  alo_a = ma
        double[:,:] alo_o = mo
        short[:]    alo   = ml
        short[:]    ala   = mp

    for p in range(nb):
        alo[p] = len(basis[p].coefficients)
        ala[p] = basis[p].atom.id
        for q in range(len(basis[p].coefficients)):
            alo_e[p,q] = basis[p].exponents[q]
            alo_c[p,q] = basis[p].coefficients[q]
            alo_n[p,q] = basis[p].normals[q]
        for q in range(3):
            alo_a[p,q] = basis[p].momenta[q]
            alo_o[p,q] = basis[p].atom.center[q]

    #matrix definitions
    cdef:
        double ss, sk, sj, sh, si, sf, f, ra, rb, force
        int axis

    Sx = np.zeros([nb,nb], dtype = np.double)
    Ix = np.zeros([nb,nb,nb,nb], dtype = np.double)
    Hx = np.zeros([nb,nb], dtype = np.double)
    Fx = np.zeros([nb,nb], dtype = np.double)
    Wx = np.zeros([nb,nb], dtype = np.double)
    Ex = np.zeros([nb,nb], dtype = np.double)

    Vx = np.zeros([na,3], dtype = np.double)
    cdef:
        double[:,:]      overlap_gradient = Sx
        double[:,:,:,:]  two_electron_gradient = Ix
        double[:,:]      one_electron_gradient = Hx
        double[:,:]      fock_gradient= Fx
        double[:,:]      weighted_energy = Wx
        double[:,:]      energy = Ex
        double[:,:]      force_tensor = Vx

#----------------------------------------Begin derivatives ------------------------------------|

#---------------------------------Begin one electron gradients---------------------------------|

    for center in range(na):
        for axis in range(3):

            for p in range(nb):
                for q in range(p+1):

                    sx, kx, jx, hx = 0.0, 0.0, 0.0, 0.0

                    for i in range(alo[p]):
                        for j in range(alo[q]):

                            f = alo_n[p,i] * alo_n[q,j] * alo_c[p,i] * alo_c[q,j]
                            if ala[p] == center:
                                sx += (grad_sx(alo_a[p,0], alo_a[p,1], alo_a[p,2], alo_a[q,0], alo_a[q,1], alo_a[q,2],
                                               alo_e[p,i], alo_e[q,j], alo_o[p,0], alo_o[p,1], alo_o[p,2], alo_o[q,0], alo_o[q,1], alo_o[q,2],
                                             [0,0,0], [0,0,0], axis, 0   ) * f)

                                kx += (grad_kx([alo_a[p,0], alo_a[p,1], alo_a[p,2]], [alo_a[q,0], alo_a[q,1], alo_a[q,2]],
                                                alo_e[p,i], alo_e[q,j], [alo_o[p,0], alo_o[p,1], alo_o[p,2]], [alo_o[q,0], alo_o[q,1], alo_o[q,2]],
                                                [0,0,0], [0,0,0], axis, 0   ) * f)

                            if ala[q] == center:
                                sx += (grad_sx(alo_a[p,0], alo_a[p,1], alo_a[p,2], alo_a[q,0], alo_a[q,1], alo_a[q,2],
                                               alo_e[p,i], alo_e[q,j], alo_o[p,0], alo_o[p,1], alo_o[p,2], alo_o[q,0], alo_o[q,1], alo_o[q,2],
                                               [0,0,0], [0,0,0], axis, 1  ) * f)

                                kx += (grad_kx([alo_a[p,0], alo_a[p,1], alo_a[p,2]], [alo_a[q,0], alo_a[q,1], alo_a[q,2]],
                                                alo_e[p,i], alo_e[q,j], [alo_o[p,0], alo_o[p,1], alo_o[p,2]], [alo_o[q,0], alo_o[q,1], alo_o[q,2]],
                                                [0,0,0], [0,0,0], axis, 1  ) * f)

                            hx -= (grad_h([alo_a[p,0], alo_a[p,1], alo_a[p,2]], [alo_a[q,0], alo_a[q,1], alo_a[q,2]],
                                           alo_e[p,i], alo_e[q,j], [alo_o[p,0], alo_o[p,1], alo_o[p,2]], [alo_o[q,0], alo_o[q,1], alo_o[q,2]],
                                           alo_x[center], [0,0,0], [0,0,0], axis  ) * f * alo_z[center])

                    for r in range(na):
                        for i in range(alo[p]):
                            for j in range(alo[q]):
                                f = alo_n[p,i] * alo_n[q,j] * alo_c[p,i] * alo_c[q,j]
                                if ala[p] == center:
                                    jx -= (grad_j([alo_a[p,0], alo_a[p,1], alo_a[p,2]], [alo_a[q,0], alo_a[q,1], alo_a[q,2]],
                                                   alo_e[p,i], alo_e[q,j], [alo_o[p,0], alo_o[p,1], alo_o[p,2]], [alo_o[q,0], alo_o[q,1], alo_o[q,2]],
                                                   alo_x[r], [0,0,0], [0,0,0], axis, 0  ) * f * alo_z[r])

                                if ala[q] == center:
                                    jx -= (grad_j([alo_a[p,0], alo_a[p,1], alo_a[p,2]], [alo_a[q,0], alo_a[q,1], alo_a[q,2]],
                                                   alo_e[p,i], alo_e[q,j], [alo_o[p,0], alo_o[p,1], alo_o[p,2]], [alo_o[q,0], alo_o[q,1], alo_o[q,2]],
                                                   alo_x[r], [0,0,0], [0,0,0], axis, 1  ) * f * alo_z[r])


                    one_electron_gradient[p,q] = one_electron_gradient[q,p] = kx + jx + hx
                    overlap_gradient[p,q] = overlap_gradient[q,p] = sx

#-----------------------------------End one electron gradients---------------------------------|

#----------------------------------Begin two electron gradients--------------------------------|

            for p in range(nb):
                for q in range(p+1):

                    si = 0.0

                    i = p*(p+1)//2 + q
                    for r in range(nb):
                        for s in range(r+1):

                            j = r*(r+1)//2 + s

                            if i >= j:
                                si = 0.0
                                if ala[p] == center:
                                    si += erifx(alo , p ,q ,r ,s ,alo_n ,alo_c ,alo_e ,alo_a , alo_o, axis, 0)
                                if ala[q] == center:
                                    si += erifx(alo , p ,q ,r ,s ,alo_n ,alo_c ,alo_e ,alo_a , alo_o, axis, 1)
                                if ala[r] == center:
                                    si += erifx(alo , p ,q ,r ,s ,alo_n ,alo_c ,alo_e ,alo_a , alo_o, axis, 2)
                                if ala[s] == center:
                                    si += erifx(alo , p ,q ,r ,s ,alo_n ,alo_c ,alo_e ,alo_a , alo_o, axis, 3)

                            two_electron_gradient[p,q,r,s] = two_electron_gradient[p,q,s,r] = two_electron_gradient[q,p,s,r] = two_electron_gradient[q,p,r,s] = si
                            two_electron_gradient[r,s,p,q] = two_electron_gradient[r,s,q,p] = two_electron_gradient[s,r,q,p] = two_electron_gradient[s,r,p,q] = si

#-----------------------------------End two electron gradients---------------------------------|

#--------------------------------------build Fock gradient-------------------------------------|

            for p in range(nb):
                for q in range(nb):
                    sf = 0.0
                    for r in range(nb):
                        for s in range(nb):
                            sf += (2.0 * two_electron_gradient[p,q,r,s] - two_electron_gradient[p,s,q,r]) * density[s,r]

                    fock_gradient[p,q] = one_electron_gradient[p,q] + sf

#|----------------------------------------build energy-----------------------------------------|

            force = 0.0
            for p in range(nb):
                for q in range(nb):
                    force -= density[p,q] * (fock_gradient[q,p] + one_electron_gradient[q,p])

#|-----------------------------------density weighted energy-----------------------------------|
            for p in range(nb):
                for q in range(nb):
                    energy[p,q] = 0.0
                    for r in range(nb):
                        energy[p,q] += fock[p,r] * density[r,q]
            for p in range(nb):
                for q in range(nb):
                    weighted_energy[p,q] = 0.0
                    for r in range(nb):
                        weighted_energy[p,q] += density[p,r] * energy[r,q]

#|-------------------------------------overlap contribution------------------------------------|

            for p in range(nb):
                for q in range(nb):
                    force += 2.0 * overlap_gradient[p,q] * weighted_energy[q,p]

#|---------------------------------------nuclear repulsion-------------------------------------|

            for i in range(na):
                ra = (sqrt((alo_x[center,0] - alo_x[i,0])*(alo_x[center,0] - alo_x[i,0]) +
                           (alo_x[center,1] - alo_x[i,1])*(alo_x[center,1] - alo_x[i,1]) +
                           (alo_x[center,2] - alo_x[i,2])*(alo_x[center,2] - alo_x[i,2])))
                rb = alo_x[center,axis] - alo_x[i,axis]
                if ra != 0:
                    force += rb * alo_z[i] * alo_z[center]/(ra*ra*ra)
#|------------------------------------------final forces---------------------------------------|

            if abs(force) > 1e-12:
                force_tensor[center,axis] = force

    return Vx

#|-----------------------------------------gradient helpers------------------------------------|
cdef double efx(int ia,int ja,int type, double r, double ie, double je, int n = 0, double x = 0.0, int p = 0, int s = 0):

    if p == 1:
        return 2.0 * ie * cye(ia+1, ja, type, r, ie, je, n, x) - ia * cye(ia-1, ja, type, r, ie, je, n, x)
    elif s == 1:
        return 2.0 * je * cye(ia, ja+1, type, r, ie, je, n, x) - ja * cye(ia, ja-1, type, r, ie, je, n, x)

#|------------------------------------------overlap helper-------------------------------------|
cdef double grad_sx(int ia0, int ia1, int ia2, int ja0, int ja1, int ja2, double ie, double je, \
     double ir0, double ir1, double ir2, double jr0, double jr1, double jr2, int[3] n, double[3] origin, int x, int center):

    cdef:
        int pa = 0
        int pb
        double t = 0.0
        double r0, r1, r2

    if center == 0:
        pa = 1
    pb = (pa+1) % 2

    r0 = ir0 - jr0
    r1 = ir1 - jr1
    r2 = ir2 - jr2

    if x == 0:
        t =  efx(ia0, ja0 , 0, r0, ie, je, n[0], ir0 - origin[0],pa ,pb )
        t *= cye(ia1, ja1 , 0, r1, ie, je, n[1], ir1 - origin[1])
        t *= cye(ia2, ja2 , 0, r2, ie, je, n[2], ir2 - origin[2])
    elif x == 1:
        t =  cye(ia0, ja0 , 0, r0, ie, je, n[0], ir0 - origin[0])
        t *= efx(ia1, ja1 , 0, r1, ie, je, n[1], ir1 - origin[1],pa ,pb )
        t *= cye(ia2, ja2 , 0, r2, ie, je, n[2], ir2 - origin[2])
    elif x == 2:
        t =  cye(ia0, ja0 , 0, r0, ie, je, n[0], ir0 - origin[0])
        t *= cye(ia1, ja1 , 0, r1, ie, je, n[1], ir1 - origin[1])
        t *= efx(ia2, ja2 , 0, r2, ie, je, n[2], ir2 - origin[2],pa ,pb )

    return t * pow(pi/(ie+je), 1.5)

#|------------------------------------------kinetic helper-------------------------------------|
cdef double grad_kx(int[3] ia, int[3] ja, double ie, double je, double[3] ir, double[3] jr, \
     int[3] n, double[3] origin, int x, int center):

    #cases for center 'a' and center 'b'
    cdef:
        int pa = 0
        int pb, i
        double[3] t
        double[3] mu, nu, vu

    if center == 0:
        pa = 1
    pb = (pa+1) % 2

    for i in range(3):
        mu[i] = (2*ja[i] + 1) * je
        nu[i] = -2*pow(je,2)
        vu[i] = -0.5 * ja[i]* (ja[i]-1)
        t[i] = 0.0

    for i in range(3):

        if i == x:
            t[x] = mu[x] * efx(ia[x], ja[x] , 0, ir[x] - jr[x], ie, je, n[x], ir[x] - origin[x],pa ,pb ) +     \
                   nu[x] * efx(ia[x], ja[x] + 2 , 0, ir[x] - jr[x], ie, je, n[x], ir[x] - origin[x],pa ,pb ) + \
                   vu[x] * efx(ia[x], ja[x] - 2, 0, ir[x] - jr[x], ie, je, n[x], ir[x] - origin[x],pa ,pb )

        else:
            t[i] = mu[i] * cye(ia[i], ja[i] , 0, ir[i] - jr[i], ie, je, n[i], ir[i] - origin[i]) +    \
                   nu[i] * cye(ia[i], ja[i] + 2, 0, ir[i] - jr[i], ie, je, n[i], ir[i] - origin[i]) + \
                   vu[i] * cye(ia[i], ja[i] - 2, 0, ir[i] - jr[i], ie, je, n[i], ir[i] - origin[i])

    for i in range(3):

        if i == x:
            t[(x+1) % 3] *= efx(ia[x], ja[x] , 0, ir[x] - jr[x], ie, je, n[x], ir[x] - origin[x],pa ,pb )
            t[(x+2) % 3] *= efx(ia[x], ja[x] , 0, ir[x] - jr[x], ie, je, n[x], ir[x] - origin[x],pa ,pb )
        else:
            t[(i+1) % 3] *= cye(ia[i], ja[i], 0, ir[i] - jr[i], ie, je, n[i], ir[i] - origin[i])
            t[(i+2) % 3] *= cye(ia[i], ja[i], 0, ir[i] - jr[i], ie, je, n[i], ir[i] - origin[i])

    return (t[0] + t[1] + t[2]) * pow(pi/(ie+je), 1.5)

#|------------------------------------------overlap helper-------------------------------------|
cdef double grad_j(int[3] ia, int[3] ja, double ie, double je, double[3] ir, double[3] jr, \
     double[:] nucleus, int[3] n, double[3] origin, int x, int center):
    #generalised coulomb derivatives dV(ab^(1,0,0))/dx terms - overlap derivative forces

    cdef:
        double p = ie + je
        double[3] q, r
        int i, mu, nu, vu, pa, pb
        int tau[3]
        double sum, val
        int[3] xi

    for i in range(0, 3):
        q[i] = (ie*ir[i] + je*jr[i])/p
        tau[i] = ia[i] + ja[i] + n[i] + 1
        r[i] = q[i] - nucleus[i]

    tau[x] += 1

    pa = 0
    if center == 0:
        pa = 1
    pb = (pa+1) % 2

    sum = 0.0
    val = 1.0

    for mu in range(tau[0]):
        for nu in range(tau[1]):
            for vu in range(tau[2]):
                val = 1.0
                xi = [mu,nu,vu]
                for i in range(3):
                    if i == x:
                        val *= efx(ia[x], ja[x], xi[x], ir[x]-jr[x], ie, je, n[x], ir[x]-nucleus[x], pa, pb)
                    else:
                        val *=   cye(ia[i], ja[i], xi[i], ir[i]-jr[i], ie, je, n[i], ir[i]-nucleus[i])

                sum += val * clmb(mu,nu,vu, 0, p, r[0], r[1], r[2] )

    return sum * 2 * pi/p

cdef double grad_h(int[3] ia, int[3] ja, double ie, double je, double[3] ir, double[3] jr, \
     double[:] nucleus, int[3] n, double[3] origin, int x):
    #generalised coulomb derivatives dV(ab^(0,0,0))/dx terms - operator derivatives inc Hellman-Feynman forces

    cdef:
        double p = ie + je
        double[3] q, r
        int i, mu, nu, vu, pa, pb
        int[3] tau
        double sum, val
        int[3] xi

    for i in range(3):
        q[i] = (ie*ir[i] + je*jr[i])/p
        tau[i] = ia[i] + ja[i] + n[i] + 1
        r[i] = q[i] - nucleus[i]

    sum = 0.0
    val = 1.0

    for mu in range(tau[0]):
        for nu in range(tau[1]):
            for vu in range(tau[2]):
                val = 1.0
                xi = [mu,nu,vu]
                for i in range(3):
                    val *=   cye(ia[i], ja[i], xi[i], ir[i]-jr[i], ie, je, n[i], ir[i]-nucleus[i])

                xi[x] += 1

                sum -= val * clmb(xi[0],xi[1],xi[2], 0, p, r[0], r[1], r[2] )

    return sum * 2 * pi/p

cdef double two_electron_gradient(short[:] ia, short[:] ja, short[:] ka, short[:] la, double ie, double je, double ke, double le, \
                  double[:] ir, double[:] jr, double[:] kr, double[:] lr, int[3] ra, int[3] rb, double[3] origin, int x, int center):

    cdef:
        double p = ie + je
        double q = ke + le
        double rho = p*q/(p + q)
        double[3] P, Q, r
        int i, pa, pb, mu, nu, vu, psi, phi, chi
        int[3] xia, xib, tau, sigma
        double val = 0.0
        double term

    for i in range(0, 3):
        P[i] = (ie*ir[i] + je*jr[i])/p
        Q[i] = (ke*kr[i] + le*lr[i])/q
        r[i] = P[i] - Q[i]

        tau[i] = ia[i] + ja[i] + 1 + ra[i]
        sigma[i] = ka[i] + la[i] + 1 + rb[i]

    if (center == 0) or (center == 1):
        tau[x] += 1
    else:
        sigma[x] += 1

    pa = 0
    if (center == 0) or (center == 2):
        pa = 1
    pb = (pa+1) % 2

    for mu in range(tau[0]):
        for nu in range(tau[1]):
            for vu in range(tau[2]):
                for psi in range(sigma[0]):
                    for phi in range(sigma[1]):
                        for chi in range(sigma[2]):
                            xia = [mu, nu, vu]
                            xib = [psi, phi, chi]
                            term = 1.0
                            for i in range(0, 3):
                                if (i == x):
                                    if (center == 0 or center == 1):
                                        term *= efx(ia[x],ja[x],xia[x],ir[x]-jr[x],ie,je,ra[x],ir[x] - origin[x], pa, pb)
                                        term *= cye(ka[x],la[x],xib[x],kr[x]-lr[x],ke,le,rb[x],kr[x] - origin[x])
                                    elif (center == 2 or center == 3):
                                        term *= cye(ia[x],ja[x],xia[x],ir[x]-jr[x],ie,je,ra[x],ir[x] - origin[x])
                                        term *= efx(ka[x],la[x],xib[x],kr[x]-lr[x],ke,le,rb[x],kr[x] - origin[x], pa, pb)

                                else:
                                    term *= cye(ia[i],ja[i],xia[i],ir[i]-jr[i],ie,je,ra[i],ir[i] - origin[i])
                                    term *= cye(ka[i],la[i],xib[i],kr[i]-lr[i],ke,le,rb[i],kr[i] - origin[i])

                            term *= pow(-1, (psi+phi+chi)) * clmb(mu+psi,nu+phi,vu+chi,0, rho,r[0], r[1], r[2])
                            val += term

    return val*2*pow(pi,2.5)/(p*q*sqrt(p+q))


cdef double erifx(short[:] ng, int p, int q, int r, int s, double[:,:] im, double[:,:] ic, double[:,:] ie, short[:,:] ia, double[:,:] io, \
                  int x, int center):

    cdef:
        double sum = 0.0
        int i, j, k, l

    for i in range(ng[p]):
        for j in range(ng[q]):
            for k in range(ng[r]):
                for l in range(ng[s]):
                    sum += im[p,i]*im[q,j]*im[r,k]*im[s,l] * ic[p,i]*ic[q,j]*ic[r,k]*ic[s,l] *   \
                           two_electron_gradient(ia[p], ia[q], ia[r], ia[s], ie[p,i], ie[q,j], ie[r,k], ie[s,l], \
                                 io[p], io[q], io[r], io[s], [0,0,0], [0,0,0], [0,0,0], x, center)

    return sum

#-----------------------------------End Shared Routines -----------------------------------|
