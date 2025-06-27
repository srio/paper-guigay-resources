#
# Fig 2 in Acta Cryst. (2016). A72, 489–499 Guigay and Ferrero Dynamical focusing by bent Laue crystals
#

import numpy
from scipy.special import jv as BesselJ
import scipy.constants as codata
from crystal_data import get_crystal_data
from srxraylib.plot.gol import plot, set_qt, plot_show
import mpmath

# sgplus[x_, q_] :=
#   2*NIntegrate[
#     Hypergeometric1F1[I*kap, 1, I*acmax*(1 - (v/a)^2)]*
#      Exp[0.5*I*k*v^2*invle[q]]*Cos[k*v*(x/q - I*kiny)], {v, 0, a}];

def sgplus(x, q):
    v = numpy.linspace(0, a, 1000)
    y = numpy.zeros_like(v, dtype=complex)
    for i in range(v.size):
        y[i] = mpmath.hyp1f1(1j * kap, 1, 1j * acmax * (1 - (v[i] / a)**2)) * \
            numpy.exp(1j * k * 0.5 * v[i]**2 * invle(q)) * numpy.cos(k * v[i] * (x / q - 1j * kiny))
    return 2 * numpy.trapz(y, x=v)

# curved symmetric Laue, source at finite p
# Eq. 13 in doi:10.1107/S0108767312044601  (2013) or
# Eq. 23 in https://doi.org/10.1107/S1600577521012480 (2022)
def integral_psisym(x, q, k=0.0, Z=0.0, a=0.0, RcosTheta=0.0, p=0.0, on_x_at_maximum=0):
    v = numpy.linspace(-a, a, 2000)
    y = numpy.zeros_like(v, dtype=complex)
    pe = p * RcosTheta / (RcosTheta + p)
    qe = q * RcosTheta / (RcosTheta - q)
    lesym = pe + qe
    # pe + qe # RcosTheta * q / (RcosTheta - q)
    # fact_eta = (2 * x * qe / q / lesym +  a / raygam) # diverges at q=0
    fact_eta = (2 * x * (RcosTheta / (RcosTheta - q)) / lesym + a / RcosTheta)
    if on_x_at_maximum:
        x = 0.0
        fact_eta = 0.0

    for i in range(v.size):
        arg_bessel = Z * numpy.sqrt((a + v[i]) * (a - v[i]))
        y[i] = BesselJ(0, arg_bessel) * numpy.exp(1j * k * 0.5 * (v[i] ** 2 / lesym - v[i] * fact_eta))

    return numpy.trapz(y, x=v)


def get_max(xx, yy, verbose=1):
    i = numpy.argmax(yy)
    if verbose: print("Maximum found at x=%g, y=%g" % (xx[i],yy[i]))
    return xx[i], yy[i], i

def run_symmetric(plot_inset=1):

    #
    # inputs (in mm) ===========================================================
    #
    photon_energy_in_keV = 80.0
    thickness = 1.0 # mm


    p = 0.0 # mm

    crystal_id="Si"
    hkl=[1,1,1]

    #
    # end inputs ===========================================================
    #
    teta, chizero, chih = get_crystal_data(crystal_id, hkl=hkl, photon_energy_in_keV=photon_energy_in_keV, verbose=False)
    lambda1 = codata.h * codata.c / codata.e / (photon_energy_in_keV * 1e3) * 1e3 # in mm
    print("photon_energy_in_keV:", photon_energy_in_keV)
    print("Crystal %s %d%d%d" % (crystal_id, hkl[0], hkl[1], hkl[2]))
    print(">>>>>>>>>> teta_deg:", teta * 180 / numpy.pi)
    print(">>>>>>>>>> chizero:", chizero)
    print(">>>>>>>>>> chih:", chih)
    print(">>>>>>>>>> lambda1:", lambda1)



    k = 2 * numpy.pi / lambda1
    h = 2 * k * numpy.sin(teta)
    chihm = -1j * chih
    chih2 = chih * chihm

    print(">>>>>>>>>> chih*chihbar:")

    Zsym = k * numpy.sqrt(chih2) / numpy.sin(2 * teta)
    attsym = 1.0 # numpy.exp(-k * numpy.imag(chizero) * thickness / numpy.cos(teta))
    asym = thickness * numpy.sin(teta)
    Zasym = Zsym * asym
    qzero = asym * numpy.sin(2 * teta) / numpy.real(numpy.sqrt(chih2))

    print("attsym", attsym)
    print("asym, halfwith of reflected beam [**a in mm**]", asym)
    print("Zsym, Zasym", Zsym, Zasym)
    print("qzero, dynamical focal length [**q0 in mm]", qzero)

    R = 2000.0
    print("R", R)
    raygam = R * numpy.cos(teta)
    print("raygam", raygam)

    #
    # this part uses the source at p=finite, R finite, and calculates I(xi, qdyn) and I(focus,q) to obtain qdyn
    #

    qq = numpy.linspace(100, 5000, 1000)
    yy = numpy.zeros_like(qq)

    for j in range(qq.size):
        yy[j] = attsym / (lambda1 * (p + qq[j])) * numpy.abs(integral_psisym(0, qq[j],
                                                    k=k, Z=Zsym, a=asym, RcosTheta=raygam, p=p, on_x_at_maximum=1)) ** 2

    if p == 0.0:
        q_lensequation = 0
    else:
        q_lensequation = 1.0 / (2 / raygam - 1 / p)
    plot(qq, yy,
         [q_lensequation, q_lensequation], [0, yy.max()], legend=['Dynamical theory', 'Lens equation'],
         xtitle='q [mm]', ytitle="Intensity on axis", title="R=%g mm p=%g mm" % (R, p), show=0)

    if plot_inset: # lateral scan (I vs chi)
        qdyn, _, imax  = get_max(qq, yy)
        xi = numpy.linspace(-asym, asym, 1000)
        yy1 = numpy.zeros_like(xi)
        for j in range(xi.size):
            yy1[j] = attsym / (lambda1 * (p + qq[imax])) * \
                     numpy.abs(integral_psisym(xi[j], qq[imax], k=k, Z=Zsym, a=asym, RcosTheta=raygam, p=p)) ** 2

        yy2_ampl = numpy.zeros_like(xi, dtype=complex)
        yy2 = numpy.zeros_like(xi)
        qposition = 2466.0
        for j in range(xi.size):
            yy2_ampl[j] = numpy.sqrt(attsym / (lambda1 * (p + qposition))) * \
                          integral_psisym(xi[j], qposition, k=k, Z=Zsym, a=asym, RcosTheta=raygam, p=p)
            yy2[j] =  numpy.abs(yy2_ampl[j]) ** 2

        plot(xi, yy1, xi, yy2, legend=['q=%.1f mm' % qq[imax], 'q=%0.1f mm' % qposition],
             xtitle='xi [mm]', ytitle="Intensity",
             title="xi scan R=%g mm, p=%.1f, ReflInt = %g" % (R, p, yy1.sum() * (xi[1] - xi[0])),
             show=0)
    plot_show()

if __name__ == "__main__":
    import mpmath

    #
    # test Kummer
    #
    if False:
        # In[86] := Hypergeometric1F1[0.1, 1, 0.2 * I]
        # Out[86] = 0.998902 + 0.0199487
        out = mpmath.hyp1f1(0.1, 1, 0.2j)
        print(out)

        # In[87]:= Hypergeometric1F1[0.1 * I , 1, 0.2 * I]
        # Out[87]= 0.980144 - 0.000991696 I
        out = mpmath.hyp1f1(0.1j, 1, 0.2j)
        print(out)


    #
    # alpha=0
    #
    if 0:
        run_symmetric(plot_inset=1)

    #
    # asymmetric
    #

    #
    # inputs (in mm) ===========================================================
    #
    if 1:
        teta = 1.4161222418 * numpy.pi / 180
        lambda1 = 0.01549817566e-6
        k = 2 * numpy.pi / lambda1
        h = 2 * k * numpy.sin(teta)
        chizero = -0.150710e-6 + 1j * 0.290718e-10
        chih = numpy.conjugate(-5.69862 + 1j * 5.69574) * 1e-8
        chimh = -1j * chih;
        chih2 = chih * chimh
        u2 = 0.25 * chih2 * k**2
        thickness = 1
        p = 0
        R = 2000
        raygam = R * numpy.cos(teta)
        kp = k * numpy.sin(2 * teta)
        kp2 = k * numpy.sin(2 * teta)**2
        poisson_ratio = 0.2201

        #
        # TODO: Not working for alfa_deg=0
        #

        alfa_deg = -0.05
        alfa = alfa_deg * numpy.pi / 180
        teta1 = alfa + teta
        teta2 = alfa - teta
        fam1 = numpy.sin(teta1)
        fam2 = numpy.sin(teta2)
        gam1 = numpy.cos(teta1)
        gam2 = numpy.cos(teta2)
        t1 = thickness / gam1
        t2 = thickness / gam2
        qpoly = p * R * gam2 / (2 * p + R * gam1)
        att = numpy.exp(-k * 0.5 * (t1 + t2) * numpy.imag(chizero))
        s2max = 0.25 * t1 * t2
        u2max = u2 * s2max
        gamma = t2 / t1
        a = numpy.sin(2 * teta) * t1 * 0.5
        kin = 0.25 * (t1 - t2) * chizero / a
        kinx = numpy.real(kin)
        kiny = numpy.imag(kin)
        com = numpy.sin(alfa) * (1 + gam1 * gam2 * (1 + poisson_ratio))
        kp3 = 0.5 * k * (gamma * a)**2
        mu1 = (numpy.cos(alfa) * 2 * fam1 * gam1 + numpy.sin(alfa) * (fam1**2 + poisson_ratio * gam1**2)) / (numpy.sin(2*teta)* numpy.cos(teta))
        mu2 = (numpy.cos(alfa) * 2 * fam2 * gam2 + numpy.sin(alfa) * (fam2**2 + poisson_ratio * gam2**2)) / (numpy.sin(2*teta)* numpy.cos(teta))
        a1 = (0.5 * thickness / numpy.cos(teta)) * (numpy.cos(alfa) * numpy.sin(teta1) + poisson_ratio*numpy.sin(alfa) * numpy.cos(teta1))
        a2 = (0.5 * thickness / numpy.cos(teta)) * (numpy.cos(alfa) * numpy.sin(teta2) + poisson_ratio*numpy.sin(alfa) * numpy.cos(teta2))
        acrist = -h * numpy.sin(alfa) * (1 + gam1 * gam2 * (1 + poisson_ratio)) / R
        acmax = acrist * s2max
        g = gamma * acrist * R / kp2
        kap = u2max / acmax   # TODO acmax is zero when alfa is zero!!!!!!!!!!!!!!!!!!
        pe = p * R / (gamma**2 * (R - p * mu2) - g * p)

        # qe[q_] := q*rayon/(rayon - q*(mu1 + g));
        qe = lambda q:  q * R / (R - q * (mu1 + g))

        # be[q_] := 1/qe[q] + 1/pe;
        be = lambda q: 1 / qe(q) + 1 / pe

        # le[q_] := (pe + qe[q])/(1 + g*(pe*qe[q]));
        le = lambda q: (pe + qe(q)) / (1 + g * (pe * qe(q)))

        # invle[q_] := 1/q - mu1/rayon;
        invle = lambda q: 1 / q - mu1 / R

        # q-scan
        if True:
            qq = numpy.linspace(100, 5000, 100)
            yy = numpy.zeros_like(qq)
            for j in range(qq.size):
                yy[j] = numpy.abs(sgplus(0, qq[j]) ** 2 * att / (lambda1 * qq[j]))
            plot(qq, yy,
                 xtitle='q [mm]', ytitle="Intensity on axis", title="alfa=%g deg" % (alfa_deg),
                 show=0)
            qdyn, _, imax = get_max(qq, yy)
            qposition = qdyn
        else:
            qposition = 1680.96

        # x-scan
        if 1:
            xx = numpy.linspace(-0.0025, .0025, 200)
            yy = numpy.zeros_like(xx)
            for j in range(xx.size):
                yy[j] = numpy.abs(sgplus(xx[j], qposition) ** 2 * att / (lambda1 * qposition))
            plot(xx, yy,
                 xtitle='xi [mm]', ytitle="Intensity on axis", title="alfa=%g deg" % (alfa_deg),
                 show=0)

    plot_show()