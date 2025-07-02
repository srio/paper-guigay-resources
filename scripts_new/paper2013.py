import numpy
from scipy.special import jv as BesselJ
import scipy.constants as codata

# flat symmetric Laue, source at surface (p=0)
# Eq. in pag 92 of https://doi:10.1107/S0108767312044601  (for on-axis only) or
# Eq. 18 in https://doi.org/10.1107/S1600577521012480 (2022)
# psizero[x_,q_]:=2*NIntegrate[BesselJ[0,sym*Sqrt[(asym+v)*(asym-v)]]*Exp[I*k*0.5*(v-x)^2/q],{v,0,asym}];
def integral_pzero(x, q, k=0.0, Z=0.0, a=0.0):
    v = numpy.linspace(-a, a, 2000)
    y = numpy.zeros_like(v,dtype=complex)
    for i in range(v.size):
        arg_bessel = Z * numpy.sqrt((a + v[i]) * (a - v[i]))
        y[i] = BesselJ(0, arg_bessel) * numpy.exp(1j * k * 0.5 * (v[i] - x) ** 2 / q)
    return numpy.trapz(y, x=v)

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


def calculate_fig(fig=4, plot_inset=1):
    #
    # Attempt to simulate Fig. 4 ** Check the value of R **
    #

    if fig == 4:
        photon_energy_in_keV = 20.0
        thickness = 0.200  # mm
        qposition = 750.0
        p = 29700.0  # mm
    elif fig == 5:
        photon_energy_in_keV = 20.0
        thickness = 0.300
        qposition = 1125.0
        p = 29700.0  # mm
    elif fig == 6:
        photon_energy_in_keV = 8.3
        thickness = 0.300
        qposition = 0.001
        p = 29700.0  # mm
    else:
        raise NotImplementedError()

    #
    #
    #

    crystal_id = "Si"
    hkl = [1, 1, 1]

    if True:
        teta, chizero, chih = get_crystal_data(crystal_id, hkl=hkl, photon_energy_in_keV=photon_energy_in_keV,
                                               verbose=False)
        lambda1 = codata.h * codata.c / codata.e / (photon_energy_in_keV * 1e3) * 1e3  # in mm
        chihm = -1j * chih
        chih2 = numpy.conjugate(chih * chihm)
    else: # for figure 6
        teta = 13.72985 * (numpy.pi / 180)
        lambda1 = 0.148842e-6
        chih = (-0.552979 + 1j * 0.522674) * 1e-5
        chihm = 1j * chih
        chizero = -0.141506e-4 - 1j * 0.306695e-6
        chih2 = chih * chihm
        """
        >>>>>>>>>> chizero:      (-1.41506e-05-3.06695e-07j)
        >>>>>>>>>> chih:         (-5.52979e-06+5.2267400000000005e-06j)
        >>>>>>>>>> chihm:        (-5.2267400000000005e-06-5.52979e-06j)
        >>>>>>>>>> chih*chihbar: (5.780554916920001e-11+3.2597664164999957e-12j)
        """






    k = 2 * numpy.pi / lambda1
    h = 2 * k * numpy.sin(teta)


    Zsym = k * numpy.sqrt(chih2) / numpy.sin(2 * teta)
    attsym = 1.0  # numpy.exp(-k * numpy.imag(chizero) * thickness / numpy.cos(teta))
    asym = thickness * numpy.sin(teta)
    Zasym = Zsym * asym
    qzero = asym * numpy.sin(2 * teta) / numpy.real(numpy.sqrt(chih2))
    # raygam = rayon * numpy.cos(teta)

    print("photon_energy_in_keV:", photon_energy_in_keV)
    print("Crystal %s %d%d%d" % (crystal_id, hkl[0], hkl[1], hkl[2]))
    print(">>>>>>>>>> teta_deg:", teta * 180 / numpy.pi)
    print(">>>>>>>>>> chizero:", chizero)
    print(">>>>>>>>>> chih:", chih)
    print(">>>>>>>>>> chihm:", chihm)
    print(">>>>>>>>>> chih*chihbar:", chih2)
    print(">>>>>>>>>> lambda1 in mm:", lambda1)
    print("attsym", attsym)
    print("asym, halfwith of reflected beam [**a in mm**]", asym)
    print("Zsym, Zasym", Zsym, Zasym)
    print("qzero, dynamical focal length [**q0 in mm]", qzero)

    # print("rayon", rayon)
    # print("raygam", raygam)

    #######

    # q-scan
    if True:
        qq = numpy.linspace(100, 2000, 1000)
        yy = numpy.zeros_like(qq)

        for j in range(qq.size):
            # image at distance q , curvature adjusted for chromatic focusing ;
            # R * cos(theta) = 2 pq / (p - q)
            raygam_j = 2 * p * qq[j] / (p - qq[j])
            yy[j] = attsym / (lambda1 * (p + qq[j])) * numpy.abs(integral_psisym(0, qq[j],
                                                                                 Z=Zsym,
                                                                                 k=k,
                                                                                 a=asym,
                                                                                 RcosTheta=raygam_j,
                                                                                 p=p,
                                                                                 on_x_at_maximum=1)
                                                                 )**2

        qdyn, _, imax = get_max(qq, yy)
        plot(qq, yy,
             xtitle='q [mm]', ytitle="Intensity on axis", title="p=%g mm; max at %g" % (p, qdyn), show=0)
    else:
        qdyn = 500.0

    if plot_inset:

        xi = numpy.linspace(-asym, asym, 1000)
        yy1 = numpy.zeros_like(xi)
        raygam_j = 2 * p * qdyn / (p - qdyn)
        for j in range(xi.size):
            yy1[j] = attsym / (lambda1 * (p + qdyn)) * \
                     numpy.abs(integral_psisym(xi[j],
                                               qq[imax],
                                               k=k,
                                               Z=Zsym,
                                               a=asym,
                                               RcosTheta=raygam_j,
                                               p=p)
                               ) ** 2



        yy2_ampl = numpy.zeros_like(xi, dtype=complex)
        yy2 = numpy.zeros_like(xi)

        raygam_j = 2 * p * qposition / (p - qposition)
        for j in range(xi.size):
            yy2_ampl[j] = numpy.sqrt(attsym / (lambda1 * (p + qposition))) * \
                          integral_psisym(xi[j],
                                          qposition,
                                          k=k,
                                          Z=Zsym,
                                          a=asym,
                                          RcosTheta=raygam_j,
                                          p=p)
            yy2[j] =  numpy.abs(yy2_ampl[j]) ** 2

        plot(xi, yy1, xi, yy2, legend=['q=%.1f mm' % qdyn, 'q=%0.1f mm' % qposition],
             xtitle='xi [mm]', ytitle="Intensity",
             title="xi scan, p=%.1f, ReflInt = %g" % (p, yy1.sum() * (xi[1] - xi[0])),
             show=0)



if __name__ == "__main__":
    from crystal_data import get_crystal_data
    from srxraylib.plot.gol import plot, set_qt, plot_show
    set_qt()


    # calculate_fig(fig=4, plot_inset=1)
    # calculate_fig(fig=5, plot_inset=1)
    calculate_fig(fig=6, plot_inset=0)


    plot_show()