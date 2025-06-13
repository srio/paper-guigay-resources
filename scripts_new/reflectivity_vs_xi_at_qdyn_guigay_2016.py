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
        y[i] = mpmath.hyp1f1(I * kap, 1, I * acmax * (1 - (v[i] / a)**2)) * \
            Exp(I * k * 0.5 * v[i]**2 * invle(q)) * Cos(k * v[i] * (x / q - I * kiny))
    return 2 * numpy.trapz(y, x=v)

# curved symmetric Laue, source at finite p
# Eq. 13 in doi:10.1107/S0108767312044601  or
# Eq. 23 in https://doi.org/10.1107/S1600577521012480
def integral_psisym(x, q, Z=0.0, k=0.0, a=0.0, RcosTheta=0.0, p=0.0, on_x_at_maximum=0):
    v = numpy.linspace(0, a, 1000)
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

    return 2 * numpy.trapz(y, x=v)


def get_max(xx, yy, verbose=1):
    i = numpy.argmax(yy)
    if verbose: print("Maximum found at x=%g, y=%g" % (xx[i],yy[i]))
    return xx[i], yy[i], i

def run_symmetric():

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

    rayon = 2000.0 # 2.0 / numpy.cos(teta) / (1 / p + 1 / q)  # 1000 # mm
    print("rayon", rayon)
    raygam = rayon * numpy.cos(teta)
    print("raygam", raygam)

    #
    # this part uses the source at p=finite, R finite, and calculates I(xi, qdyn) and I(focus,q) to obtain qdyn
    #
    # Figure 2 for alpha=0
    if True:
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
             xtitle='q [mm]', ytitle="Intensity on axis", title="R=%g mm p=%g mm" % (rayon, p), show=0)

        if 1: # lateral scan (I vs chi)
            qdyn, _, imax  = get_max(qq, yy)
            xi = numpy.linspace(-0.02, 0.02, 1000)
            yy1 = numpy.zeros_like(xi)
            for j in range(xi.size):
                yy1[j] = attsym / (lambda1 * (p + qq[imax])) * \
                         numpy.abs(integral_psisym(xi[j], qq[imax],
                                                   k=k, Z=Zsym, a=asym, RcosTheta=raygam, p=p)) ** 2

            yy2_ampl = numpy.zeros_like(xi, dtype=complex)
            yy2 = numpy.zeros_like(xi)
            qposition = 0.480e3 # 0.548e3 # 0.0
            for j in range(xi.size):
                yy2_ampl[j] = numpy.sqrt(attsym / (lambda1 * (p + qposition))) * \
                              integral_psisym(xi[j], qposition, Z=Zsym, a=asym, RcosTheta=raygam, p=p)
                yy2[j] =  numpy.abs(yy2_ampl[j]) ** 2
            # print(yy2)

            plot(xi, yy1, xi, yy2, legend=['q=%.1f mm' % qq[imax], 'q=%0.1f mm' % qposition],
                 xtitle='xi [mm]', ytitle="Intensity",
                 title="xi scan R=%g mm, p=%.1f, ReflInt = %g" % (rayon, p, yy1.sum() * (xi[1] - xi[0])),
                 show=0)
    plot_show()

if __name__ == "__main__":
    import mpmath

    #
    # test Kummer
    #
    if 0:
        # In[86] := Hypergeometric1F1[0.1, 1, 0.2 * I]
        # Out[86] = 0.998902 + 0.0199487
        out = mpmath.hyp1f1(0.1, 1, 0.2j)
        print(out)

        # In[87]:= Hypergeometric1F1[0.1 * I , 1, 0.2 * I]
        # Out[87]= 0.980144 - 0.000991696 I
        out = mpmath.hyp1f1(0.1j, 1, 0.2j)
        print(out)


    # run_symmetric()

    #
    # asymmetric
    #


    #
    # inputs (in mm) ===========================================================
    #

    # photon_energy_in_keV = 80.0
    # thickness = 1.0 # mm
    #
    #
    # p = 0.0 # mm
    #
    # crystal_id="Si"
    # hkl=[1,1,1]
    #
    # #
    # # end inputs ===========================================================
    # #
    #
    #
    # teta, chizero, chih = get_crystal_data(crystal_id, hkl=hkl, photon_energy_in_keV=photon_energy_in_keV, verbose=False)
    # lambda1 = codata.h * codata.c / codata.e / (photon_energy_in_keV * 1e3) * 1e3 # in mm
    # print("photon_energy_in_keV:", photon_energy_in_keV)
    # print("Crystal %s %d%d%d" % (crystal_id, hkl[0], hkl[1], hkl[2]))
    # print(">>>>>>>>>> teta_deg:", teta * 180 / numpy.pi)
    # print(">>>>>>>>>> chizero:", chizero)
    # print(">>>>>>>>>> chih:", chih)
    # print(">>>>>>>>>> lambda1:", lambda1)
    #
    #
    #
    # k = 2 * numpy.pi / lambda1
    # h = 2 * k * numpy.sin(teta)
    # chihm = -1j * chih
    # chih2 = chih * chihm
    #
    # print(">>>>>>>>>> chih*chihbar:")

    Pi = numpy.pi
    Sin = numpy.sin
    Cos = numpy.cos
    Exp = numpy.exp
    I = 1j
    Conjugate = numpy.conjugate
    Im = numpy.imag
    Re = numpy.real
    Abs = numpy.abs

    teta = 1.4161222418*Pi/180
    lambda1 = 0.01549817566e-6
    k = 2*Pi/lambda1
    h = 2*k*Sin(teta)
    chizero = -0.150710e-6 + I*0.290718e-10
    chih = Conjugate(-5.69862 + I*5.69574)*1e-8
    chimh = -I*chih;
    chih2 = chih*chimh
    u2 = 0.25*chih2*k**2
    t = 1
    p = 0
    rayon = 2000
    raygam = rayon*Cos(teta)
    kp = k*Sin(2*teta)
    kp2 = k*Sin(2*teta)**2
    rau = 0.2201
    pha = 0.67


    alfa = 0.05*Pi/180
    teta1 = alfa + teta
    teta2 = alfa - teta
    fam1 = Sin(teta1)
    fam2 = Sin(teta2)
    gam1 = Cos(teta1)
    gam2 = Cos(teta2)
    t1 = t/gam1
    t2 = t/gam2
    qpoly = p*rayon*gam2/(2*p + rayon*gam1)
    att = Exp(-k*0.5*(t1 + t2)*Im(chizero))
    s2max = 0.25*t1*t2; u2max = u2*s2max
    gama = t2/t1
    a = Sin(2*teta) * t1 * 0.5
    kin = 0.25*(t1 - t2)*chizero/a
    kinx = Re(kin)
    kiny = Im(kin)
    com = Sin(alfa) * (1 + gam1 * gam2 * (1 + rau))
    kp3 = 0.5 * k * (gama*a)**2
    mu1 = (Cos(alfa) * 2 * fam1 * gam1 + Sin(alfa) * (fam1**2 + rau * gam1**2)) / (Sin(2*teta)* Cos(teta))
    mu2 = (Cos(alfa) * 2 * fam2 * gam2 + Sin(alfa) * (fam2**2 + rau * gam2**2)) / (Sin(2*teta)* Cos(teta))
    a1 = (0.5 * t / Cos(teta)) * (Cos(alfa) * Sin(teta1) + rau*Sin(alfa) * Cos(teta1))
    a2 = (0.5 * t / Cos(teta)) * (Cos(alfa) * Sin(teta2) + rau*Sin(alfa) * Cos(teta2))
    acrist = -h * Sin(alfa) * (1 + gam1 * gam2 * (1 + rau)) / rayon
    acmax = acrist * s2max
    g = gama * acrist * rayon / kp2
    kap = u2max / acmax
    pe = p * rayon / (gama**2 * (rayon - p * mu2) - g * p)


    # qe[q_] := q*rayon/(rayon - q*(mu1 + g));
    qe = lambda q:  q * rayon / (rayon - q * (mu1 + g))

    # be[q_] := 1/qe[q] + 1/pe;
    be = lambda q: 1 / qe(q) + 1 / pe


    # le[q_] := (pe + qe[q])/(1 + g*(pe*qe[q]));
    le = lambda q: (pe + qe(q)) / (1 + g * (pe * qe(q)))

    # invle[q_] := 1/q - mu1/rayon;
    invle = lambda q: 1 / q - mu1 / rayon

    # q-scan
    if 1:
        qq = numpy.linspace(100, 5000, 500)
        yy = numpy.zeros_like(qq)
        for j in range(qq.size):
            yy[j] = Abs(sgplus(0, qq[j]) ** 2 * att / (lambda1 * qq[j]))
            print(j, qq[j], yy[j])
        plot(qq, yy)
        qdyn, _, imax = get_max(qq, yy)

    # x-scan
    if 0:
        xx = numpy.linspace(-2.5e-3, 2.5e-3, 200)
        yy = numpy.zeros_like(xx)
        qposition = 1679.0
        for j in range(xx.size):
            yy[j] = Abs(sgplus(xx[j], qposition) ** 2 * att / (lambda1 * qposition))
            print(j, xx[j], yy[j])
        plot(xx, yy)


# intplus =
#  Plot[Abs[sgplus[0, q]^2*att/(lambda*q)], {q, 100, 5000},
#   AxesOrigin -> {2000, 0}, PlotRange -> All,
#   PlotStyle -> {RGBColor[0, 1, 0]}]
# psplus = Table[Abs[sgplus[0, q]^2*att/(lambda*q)], {q, 1600, 1800, 1}]
# focplus =
#  Plot[Abs[sgplus[xprim*0.001, 1679]^2*
#     att/(lambda*1679)], {xprim, -2.5, 2.5}, AxesOrigin -> {0, 0},
#   PlotRange -> All, PlotStyle -> {RGBColor[0, 1, 0]}]









