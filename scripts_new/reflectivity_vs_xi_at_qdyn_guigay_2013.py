import numpy
from scipy.special import jv as BesselJ
import scipy.constants as codata



# flat symmetric Laue, source at surface (p=0)
# Eq. in pag 92 of doi:10.1107/S0108767312044601  (for on-axis only) or
# Eq. 18 in https://doi.org/10.1107/S1600577521012480
# psizero[x_,q_]:=2*NIntegrate[BesselJ[0,sym*Sqrt[(asym+v)*(asym-v)]]*Exp[I*k*0.5*(v-x)^2/q],{v,0,asym}];
def integral_pzero(x, q, Z=0.0, a=0.0):
    v = numpy.linspace(0, a, 1000)
    y = numpy.zeros_like(v,dtype=complex)
    for i in range(v.size):
        arg_bessel = Z * numpy.sqrt((a + v[i]) * (a - v[i]))
        y[i] = BesselJ(0, arg_bessel) * numpy.exp(1j * k * 0.5 * (v[i] - x) ** 2 / q)
    return 2 * numpy.trapz(y,x=v)

# TODO: DELETE OR CHECK
# Curved (R)  symmetric Laue, source at infinity (p=inf)
# Eq. XXX
# psipinfini[x_,q_]:=2*NIntegrate[BesselJ[0,sym*Sqrt[(asym+v)*(asym-v)]]*Exp[I*k*0.5*v^2*(raygam-q)/raygam^2]*Cos[k*v*x/raygam],{v,0,asym}];
def integral_pinfinity(x, q, Z=0.0, a=0.0):
    v = numpy.linspace(0, asym, 1000)
    y = numpy.zeros_like(v, dtype=complex)
    for i in range(v.size):
        arg_bessel = Z * numpy.sqrt((a + v[i]) * (a - v[i]))
        y[i] = BesselJ(0, arg_bessel) * \
               numpy.exp(1j * k * 0.5 * v[i]**2 * (raygam - q) / raygam**2) * numpy.cos(k * v[i] * x / raygam)
    return 2 * numpy.trapz(y,x=v)


# curved symmetric Laue, source at finite p
# Eq. 13 in doi:10.1107/S0108767312044601  or
# Eq. 23 in https://doi.org/10.1107/S1600577521012480
def integral_psisym(x, q, Z=0.0, a=0.0, RcosTheta=0.0, p=0.0, on_x_at_maximum=0):
    v = numpy.linspace(0, a, 1000)
    y = numpy.zeros_like(v, dtype=complex)
    pe = p * RcosTheta / (RcosTheta + p)
    qe = q * RcosTheta / (RcosTheta - q)
    lesym = pe + qe
    # pe + qe # RcosTheta * q / (RcosTheta - q)
    # fact_eta = (2 * x * qe / q / lesym +  a / raygam) # diverges at q=0
    fact_eta = (2 * x * (RcosTheta / (RcosTheta - q)) / lesym + a / raygam)
    if on_x_at_maximum:
        x = 0.0
        fact_eta = 0.0

    for i in range(v.size):
        arg_bessel = Z * numpy.sqrt((a + v[i]) * (a - v[i]))
        y[i] = BesselJ(0, arg_bessel) * numpy.exp(1j * k * 0.5 * (v[i] ** 2 / lesym - v[i] * fact_eta))

    return 2 * numpy.trapz(y, x=v)


def get_max(xx, yy, verbose=1):
    i = numpy.argmax(yy)
    if verbose: print("Maximum found at x=%g, y=%g" % (xx[i],yy[i]))
    return xx[i], yy[i], i


if __name__ == "__main__":
    from crystal_data import get_crystal_data
    from srxraylib.plot.gol import plot, set_qt, plot_show
    set_qt()

    #
    # inputs (in mm) ===========================================================
    #
    photon_energy_in_keV = 8.3 # 20.0
    thickness = 0.300 # mm


    p = 29700.0# mm

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



    #
    # q = qzero / 2
    rayon = 1900.0 # 2.0 / numpy.cos(teta) / (1 / p + 1 / q)  # 1000 # mm
    print("rayon", rayon)
    raygam = rayon * numpy.cos(teta)
    print("raygam", raygam)



    #
    # this part uses the source at p=0, and calculates I(0,q), qdyn and I(xi, qdyn).
    #
    if False:
        for tmp in ["positive"]:
            if tmp == "negative":
                qq = numpy.linspace(-5000, -200, 100)
            elif tmp == "positive":
                qq = numpy.linspace(20, 5000, 100)

            yy = numpy.zeros_like(qq)
            for j in range(qq.size):
                yy[j] = attsym / (lambda1 * qq[j]) * numpy.abs(integral_pzero(0, qq[j], Z=Zsym, a=asym))**2

            qdyn, _, imax = get_max(qq, yy)
            plot(qq,yy, xtitle='q [mm]', ytitle="Intensity on axis", title="p=0", show=0)

            # xi-scan at q=qdyn
            xi = numpy.linspace(-asym, asym, 200)
            rr = numpy.zeros_like(xi)
            for j in range(xi.size):
                rr[j] = attsym / (lambda1 * qq[imax]) * numpy.abs(integral_pzero(xi[j], qq[imax], Z=Zsym, a=asym))**2
            plot(xi, rr,  xtitle='xi [mm]', ytitle="Intensity",
                 title="xi scan at p=0, q = %.1f, ReflInt = %g" % (qdyn, rr.sum() * (xi[1] - xi[0])), show=0)

    #
    # TODO: DELETE OR CHECK
    # this part uses the source at p=infinite, R finite, and calculates I(xi, qdyn) and I(focus,q) to obtain qdyn
    #
    if False:
        # intpinf=Plot[attsym*Abs[psipinfini[0,q]^2/(lambda*raygam)],{q,0,3000},AxesOrigin->{0,0},PlotRange->All,PlotStyle->{RGBColor[1,0,0]}]

        qq = numpy.linspace(0, 3000, 100)

        yy = numpy.zeros_like(qq)
        for j in range(qq.size):
            yy[j] = attsym / (lambda1 * raygam) * numpy.abs(integral_pinfinity(0, qq[j], Z=Zsym, a=asym))**2
        qdyn, _, imax = get_max(qq, yy)
        q_lensequation = raygam / 2
        plot(qq, yy,
             [q_lensequation, q_lensequation], [0,yy.max()], legend=['Dynamical theory', 'Lens equation'],
             xtitle='q [mm]', ytitle="Intensity on axis", title="p=infinity", show=0)

        xi = numpy.linspace(-asym, asym, 200)
        rr = numpy.zeros_like(xi)
        for j in range(xi.size):
            rr[j] = attsym / (lambda1 * raygam) * numpy.abs(integral_pinfinity(xi[j], qq[imax], Z=Zsym, a=asym))**2
        plot(xi, rr, xtitle='xi [mm]', ytitle="Intensity",
             title="xi scan R=%g mm, p at infinity, q = %.1f, ReflInt = %g" % (rayon, qdyn, rr.sum() * (xi[1] - xi[0])),
             show=0)

    #
    # this part uses the source at p=finite, R finite, and calculates I(xi, qdyn) and I(focus,q) to obtain qdyn
    #
    if True:
        qq = numpy.linspace(0, 2000, 1000)
        yy = numpy.zeros_like(qq)

        for j in range(qq.size):
            yy[j] = attsym / (lambda1 * (p + qq[j])) * numpy.abs(integral_psisym(0, qq[j],
                                                        Z=Zsym, a=asym, RcosTheta=raygam, p=p, on_x_at_maximum=1)) ** 2

        q_lensequation = 1.0 / (2 / raygam - 1 / p)
        plot(qq, yy,
             [q_lensequation, q_lensequation], [0, yy.max()], legend=['Dynamical theory', 'Lens equation'],
             xtitle='q [mm]', ytitle="Intensity on axis", title="R=%g mm p=%g mm" % (rayon, p), show=0)

        if 1:
            qdyn, _, imax  = get_max(qq, yy)
            xi = numpy.linspace(-asym, asym, 2000)
            yy1 = numpy.zeros_like(xi)
            for j in range(xi.size):
                yy1[j] = attsym / (lambda1 * (p + qq[imax])) * \
                         numpy.abs(integral_psisym(xi[j], qq[imax],
                                                   Z=Zsym, a=asym, RcosTheta=raygam, p=p)) ** 2

            yy2_ampl = numpy.zeros_like(xi, dtype=complex)
            yy2 = numpy.zeros_like(xi)
            qposition = 0.480e3 # 0.548e3 # 0.0
            for j in range(xi.size):
                yy2_ampl[j] = numpy.sqrt(attsym / (lambda1 * (p + qposition))) * \
                              integral_psisym(xi[j], qposition, Z=Zsym, a=asym, RcosTheta=raygam, p=p)
                yy2[j] =  numpy.abs(yy2_ampl[j]) ** 2
            print(yy2)

            plot(xi, yy1, xi, yy2, legend=['q=%.1f mm' % qq[imax], 'q=%0.1f mm' % qposition],
                 xtitle='xi [mm]', ytitle="Intensity",
                 title="xi scan R=%g mm, p=%.1f, ReflInt = %g" % (rayon, p, yy1.sum() * (xi[1] - xi[0])),
                 show=0)


    plot_show()