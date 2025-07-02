#
# Figs in Acta Cryst. (2016). A72, 489–499 Guigay and Ferrero Dynamical focusing by bent Laue crystals
#

import numpy
import mpmath
from scipy.special import jv as BesselJ
import scipy.constants as codata
from crystal_data import get_crystal_data
from srxraylib.plot.gol import plot, set_qt, plot_show

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

def run_symmetric(fig=2, plot_inset=1, use_automatic_chi=True):
    #
    # inputs (in mm) ===========================================================
    #

    if fig == 2:
        photon_energy_in_keV = 80.0
        thickness = 1.0 # mm
        p = 0.0 # mm
        qmax = 5000
        qposition = 2466.0  # extra q position
        # qposition = 1679
    elif fig == 3:
        photon_energy_in_keV = 80.0
        thickness = 1.0  # mm
        p = 20000.0  # mm
        qmax = 4000
        qposition = 912.826
    elif fig in [4, 5, 6]:
        photon_energy_in_keV = 20.0
        thickness = 0.250 # mm
        p = 29000.0 # mm
        qmax = 10000
        if use_automatic_chi:
            qposition = 3556.5 # extra q position
        else:
            qposition = 550
        # qposition = 3579 # 0.00001
    else:
        raise NotImplementedError()

    #
    # end inputs ===========================================================
    #
    if use_automatic_chi:
        teta, chizero, chih = get_crystal_data("Si", hkl=[1,1,1], photon_energy_in_keV=photon_energy_in_keV, verbose=False)
        lambda1 = codata.h * codata.c / codata.e / (photon_energy_in_keV * 1e3) * 1e3 # in mm
        chihm = -1j * chih
    else:
        if fig == 2:
            teta = numpy.radians(1.4161222418)
            lambda1 = 0.01549817556e-6
            chizero = -0.150710e-6 + 1j * 0.290718e-10
            chih = (-5.69862 + 1j * 5.69574) * 1e-8
            chihm = 1j * chih
        else:# # 20 keV
            teta = numpy.radians(5.67310)
            lambda1 = 0.619927e-7
            chizero = -0.242370e-5 + 1j * 0.918640e-8
            chih = (-0.922187 + 1j * 0.9131110) * 1e-6
            chihm = 1j * chih

    print("photon_energy_in_keV:", photon_energy_in_keV)
    print("CrystalSi 111")
    print(">>>>>>>>>> teta_deg:", teta * 180 / numpy.pi)
    print(">>>>>>>>>> chizero:", chizero)
    print(">>>>>>>>>> chih:", chih)
    print(">>>>>>>>>> lambda1:", lambda1)



    k = 2 * numpy.pi / lambda1
    h = 2 * k * numpy.sin(teta)
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

    qq = numpy.linspace(100, qmax, 1000)
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
        for j in range(xi.size):
            yy2_ampl[j] = numpy.sqrt(attsym / (lambda1 * (p + qposition))) * \
                          integral_psisym(xi[j], qposition, k=k, Z=Zsym, a=asym, RcosTheta=raygam, p=p)
            yy2[j] =  numpy.abs(yy2_ampl[j]) ** 2

        plot(xi, yy1, xi, yy2, legend=['q=%.1f mm' % qq[imax], 'q=%0.1f mm' % qposition],
             xtitle='xi [mm]', ytitle="Intensity",
             title="xi scan R=%g mm, p=%.1f, ReflInt = %g" % (R, p, yy1.sum() * (xi[1] - xi[0])),
             show=0)

if __name__ == "__main__":
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

    # run_symmetric(fig=2, plot_inset=1, use_automatic_chi=0)
    # run_symmetric(fig=2, plot_inset=1, use_automatic_chi=1)

    # note that with use_automatic_chi=0 the maximum is at the second main peak (like in the paper)
    # but for use_automatic_chi=1 the maximum is at the first peak.
    # run_symmetric(fig=4, plot_inset=1, use_automatic_chi=0)
    # run_symmetric(fig=4, plot_inset=1, use_automatic_chi=1)

    run_symmetric(fig=3, plot_inset=1, use_automatic_chi=1)

    plot_show()