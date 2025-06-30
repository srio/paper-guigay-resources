#
# Fig 2 in Acta Cryst. (2016). A72, 489–499 Guigay and Ferrero Dynamical focusing by bent Laue crystals
#

import numpy
import mpmath
from scipy.special import jv as BesselJ
import scipy.constants as codata
from crystal_data import get_crystal_data
from srxraylib.plot.gol import plot, set_qt, plot_show
import time

# sgplus[x_, q_] :=
#   2*NIntegrate[
#     Hypergeometric1F1[I*kap, 1, I*acmax*(1 - (v/a)^2)]*
#      Exp[0.5*I*k*v^2*invle[q]]*Cos[k*v*(x/q - I*kiny)], {v, 0, a}];

def sgplus(x, q, npoints=1000):
    v = numpy.linspace(0, a, npoints)
    y = numpy.zeros_like(v, dtype=complex)

    invle = 1 / q - mu1 / R

    for i in range(v.size):
        y[i] = mpmath.hyp1f1(1j * kap, 1, 1j * acmax * (1 - (v[i] / a)**2)) * \
            numpy.exp(1j * k * 0.5 * v[i]**2 * invle) * numpy.cos(k * v[i] * (x / q - 1j * kiny))
    return 2 * numpy.trapz(y, x=v)


def get_max(xx, yy, verbose=1):
    i = numpy.argmax(yy)
    if verbose: print("Maximum found at x=%g, y=%g" % (xx[i],yy[i]))
    return xx[i], yy[i], i

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
    # alpha=0 : see file paper2016symmetric
    #

    #
    # asymmetric Fig. 2
    #

    #
    # inputs (in mm) ===========================================================
    #
    if True:
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
        acrist = -h * numpy.sin(alfa) * (1 + gam1 * gam2 * (1 + poisson_ratio)) / R   # A in Eq 17
        acmax = acrist * s2max
        g = gamma * acrist * R / kp2
        kap = u2max / acmax   # beta = Omega / A TODO acmax is zero when alfa is zero!!!!!!!!!!!!!!!!!!
        pe = p * R / (gamma**2 * (R - p * mu2) - g * p)

        # q-scan
        if False:
            print("Calculating q-scan...")
            t0 = time.time()
            qq = numpy.linspace(100, 5000, 1000)
            yy = numpy.zeros_like(qq)
            for j in range(qq.size):
                yy[j] = numpy.abs(sgplus(0, qq[j], npoints=500) ** 2 * att / (lambda1 * qq[j]))
            print("Time in calculating q-scan %f s" % (time.time() - t0))
            plot(qq, yy,
                 xtitle='q [mm]', ytitle="Intensity on axis", title="alfa=%g deg" % (alfa_deg),
                 show=0)
            qdyn, _, imax = get_max(qq, yy)
            qposition = qdyn

        else:
            qposition = 2459.26


        # x-scan at finite q
        if False:
            print("Calculating x-scan...")
            xx = numpy.linspace(-0.0025, .0025, 200)
            yy = numpy.zeros_like(xx)
            for j in range(xx.size):
                yy[j] = numpy.abs(sgplus(xx[j], qposition) ** 2 * att / (lambda1 * qposition))
            plot(xx, yy,
                 xtitle='xi [mm]', ytitle="Intensity on axis", title="alfa=%g deg" % (alfa_deg),
                 show=0)



        # x-scan at q=0
        if True:
            print("Calculating x-scan... a=", a)
            # xx = numpy.linspace(-0.0025, .0025, 200)
            xx = numpy.linspace(-a * 0.99, a * 0.99, 200)
            yy = numpy.zeros_like(xx)

            omega = 0.25 * (t1 - t2) * chizero / a  # omega following the definition found after eq 22
            omega_real = numpy.real(omega)
            omega_imag = numpy.imag(omega)
            xc_over_q = omega_real - t1 * numpy.sin(alfa + teta) / (2 * R)

            for j in range(xx.size):
                x = xx[j]
                # equation 23
                amplitude = numpy.exp((1j * k * chizero.real - k * chizero.imag) * 0.25 * (t1 + t2)) * \
                        mpmath.hyp1f1(1j * kap, 1, 1j * acmax * (1 - (x / a) ** 2)) * \
                        numpy.exp(-0.5 * 1j * x**2 * (1 / qposition - mu1 / R) - \
                                  1j * qposition * x * (x - xc_over_q * qposition) / qposition) * \
                        numpy.exp(- x * k * omega_imag)

                yy[j] = numpy.abs(amplitude)**2

            plot(xx, yy,
                 xtitle='xi [mm]', ytitle="Intensity at q=0", title="alfa=%g deg" % (alfa_deg),
                 show=0)



    plot_show()