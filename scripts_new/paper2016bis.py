#
# Fig 2 in Acta Cryst. (2016). A72, 489–499 Guigay and Ferrero Dynamical focusing by bent Laue crystals
#

import numpy
import mpmath
import scipy.constants as codata
from crystal_data import get_crystal_data
from srxraylib.plot.gol import plot, set_qt, plot_show


# sgplus[x_, q_] :=
#   2*NIntegrate[
#     Hypergeometric1F1[I*kap, 1, I*Aeq17*(1 - (v/a)^2)]*
#      Exp[0.5*I*k*v^2*invle[q]]*Cos[k*v*(x/q - I*kiny)], {v, 0, a}];


#
# Equation 24 in Guigay and Ferrero (2016)
#
def integrate_sgplus(x, q, k=0.0, a=0.0, mu1=0.0, R=0.0, beta=0.0, acmax=0.0, kiny=0.0,
                    gamma=0, teta=0, Aeq17=0, # NEWWWWWWWWWWWWWWW
                     ):
    v = numpy.linspace(0, a, 1000)
    yy = numpy.zeros_like(v, dtype=complex)

    invle = 1 / q - mu1 / R

    for i in range(v.size):
        # y is defined before eq 23: y = A gamma (a^2-u^2) / sin^2(2theta)
        y = Aeq17 *  gamma * (a**2-v[i]**2) / (numpy.sin(2*teta))**2
        # y0 = acmax * (1 - (v[i] / a)**2)
        # print("y: ", y0, y)

        yy[i] = mpmath.hyp1f1(1j * beta, 1, 1j * y) * \
            numpy.exp(1j * k * 0.5 * v[i]**2 * invle) * numpy.cos(k * v[i] * (x / q - 1j * kiny))
    return 2 * numpy.trapz(yy, x=v)

def get_max(xx, yy, verbose=1):
    i = numpy.argmax(yy)
    if verbose: print("Maximum found at x=%g, y=%g" % (xx[i],yy[i]))
    return xx[i], yy[i], i

def run_asymmetric(plot_qscan=1, plot_xscan=1):
    # teta = 1.4161222418 * numpy.pi / 180
    # lambda1 = 0.01549817566e-6
    # chizero = -0.150710e-6 + 1j * 0.290718e-10
    # chih = numpy.conjugate(-5.69862 + 1j * 5.69574) * 1e-8

    photon_energy_in_keV = 80.0
    teta, chizero, chih = get_crystal_data("Si", hkl=[1, 1, 1], photon_energy_in_keV=photon_energy_in_keV,
                                           verbose=False)
    lambda1 = codata.h * codata.c / codata.e / (photon_energy_in_keV * 1e3) * 1e3  # in mm
    print("photon_energy_in_keV:", photon_energy_in_keV)
    print("CrystalSi 111")
    print(">>>>>>>>>> teta_deg:", teta * 180 / numpy.pi)
    print(">>>>>>>>>> chizero:", chizero)
    print(">>>>>>>>>> chih:", chih)
    print(">>>>>>>>>> lambda1:", lambda1)


    k = 2 * numpy.pi / lambda1
    h = 2 * k * numpy.sin(teta)

    chimh = -1j * chih;
    chih2 = chih * chimh
    u2 = 0.25 * chih2 * k**2
    thickness = 1
    p = 0
    R = 2000
    # raygam = R * numpy.cos(teta)
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
    # qpoly = p * R * gam2 / (2 * p + R * gam1)
    att = 1.0 # numpy.exp(-k * 0.5 * (t1 + t2) * numpy.imag(chizero))
    s2max = 0.25 * t1 * t2
    u2max = u2 * s2max  # Omega = k**2 chi_h chi_hbar / 4 ? (end of pag 490)
    gamma = t2 / t1
    a = numpy.sin(2 * teta) * t1 * 0.5
    kin = 0.25 * (t1 - t2) * chizero / a
    kinx = numpy.real(kin)
    kiny = numpy.imag(kin)
    # com = numpy.sin(alfa) * (1 + gam1 * gam2 * (1 + poisson_ratio))
    kp3 = 0.5 * k * (gamma * a)**2
    mu1 = (numpy.cos(alfa) * 2 * fam1 * gam1 + numpy.sin(alfa) * (fam1**2 + poisson_ratio * gam1**2)) / (numpy.sin(2*teta)* numpy.cos(teta))
    mu2 = (numpy.cos(alfa) * 2 * fam2 * gam2 + numpy.sin(alfa) * (fam2**2 + poisson_ratio * gam2**2)) / (numpy.sin(2*teta)* numpy.cos(teta))
    a1 = (0.5 * thickness / numpy.cos(teta)) * (numpy.cos(alfa) * numpy.sin(teta1) + poisson_ratio*numpy.sin(alfa) * numpy.cos(teta1))
    a2 = (0.5 * thickness / numpy.cos(teta)) * (numpy.cos(alfa) * numpy.sin(teta2) + poisson_ratio*numpy.sin(alfa) * numpy.cos(teta2))
    acrist = -h * numpy.sin(alfa) * (1 + gam1 * gam2 * (1 + poisson_ratio)) / R # A in eq 17
    Aeq17 = acrist

    # y is defined before eq 23: y = A gamma (a^2-u^2) / sin^2(2theta)
    # y = Aeq17 * (1 - (v[i] / a) ** 2)
    #
    acmax = acrist * s2max   # A in Eq 17
    g = gamma * acrist * R / kp2
    beta = u2max / acmax   # beta = Omega / A TODO acmax is zero when alfa is zero!!!!!!!!!!!!!!!!!!
    pe = p * R / (gamma**2 * (R - p * mu2) - g * p)

    # q-scan
    if plot_qscan:
        print("Calculating q-scan...")
        qq = numpy.linspace(100, 5000, 100)
        yy = numpy.zeros_like(qq)
        for j in range(qq.size):
            amplitude = integrate_sgplus(0, qq[j],
                                   k=k, a=a, mu1=mu1, R=R,
                                   beta=beta, acmax=acmax, kiny=kiny,
                                   gamma=gamma, teta=teta, Aeq17=Aeq17, # NEWWWWWWWWWW
                                   ) / numpy.sqrt(numpy.abs(lambda1 * qq[j]))
            yy[j] = att * numpy.abs(amplitude)**2
        plot(qq, yy,
             xtitle='q [mm]', ytitle="Intensity on axis", title="alfa=%g deg" % (alfa_deg),
             show=0)
        qdyn, _, imax = get_max(qq, yy)
        qposition = qdyn
    else:
        qposition = 1680.96

    # x-scan
    if plot_xscan:
        print("Calculating x-scan...")
        # xx = numpy.linspace(-0.0025, .0025, 200)
        xx = numpy.linspace(-a, a, 1000)
        yy = numpy.zeros_like(xx)
        for j in range(xx.size):
            amplitude = integrate_sgplus(xx[j], qposition,
                                   k=k, a=a, mu1=mu1, R=R,
                                   beta=beta, acmax=acmax, kiny=kiny,
                                   ) / numpy.sqrt(numpy.abs(lambda1 * qposition))
            yy[j] = att * numpy.abs(amplitude)**2
        plot(xx, yy,
             xtitle='xi [mm]', ytitle="Intensity on axis", title="alfa=%g deg" % (alfa_deg),
             show=0)

if __name__ == "__main__":

    #
    # for symmetric (alpha=0) use paper2-16.py
    #


    #
    # asymmetric Fig. 2
    #
    run_asymmetric(plot_qscan=1, plot_xscan=0)

    plot_show()