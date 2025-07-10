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

# equation 24?
# file Wilkins_p0.nb
# sgplus[x_, q_] :=
#   2*NIntegrate[
#     Hypergeometric1F1[I*kap, 1, I*acmax*(1 - (v/a)^2)]*
#      Exp[0.5*I*k*v^2*invle[q]]*Cos[k*v*(x/q - I*kiny)], {v, 0, a}];
def sgplus_fig2(x, q, npoints=1000):
    v = numpy.linspace(0, a, npoints)
    y = numpy.zeros_like(v, dtype=complex)
    invle = 1 / q - mu1 / R
    for i in range(v.size):
        y[i] = mpmath.hyp1f1(1j * kap, 1, 1j * acmax * (1 - (v[i] / a)**2)) * \
            numpy.exp(1j * k * 0.5 * v[i]**2 * invle) * \
            numpy.cos(k * v[i] * (x / q - 1j * kiny))
    return 2 * numpy.trapz(y, x=v)

# sgmoins[x_, q_] :=
#   2*NIntegrate[
#     Hypergeometric1F1[I*kap, 1, I*acmax*(1 - (v/a)^2)]*
#      Exp[0.5*I*k*v^2*invle[q]]*Cos[k*v*(x/q - I*kiny)], {v, 0, a}];
def sgmoins_fig2(x, q, npoints=1000):
    return sgplus_fig2(x, q, npoints=npoints)


#
# fig 5
#
def sgplus_fig5(x, q, npoints=1000):
    v = numpy.linspace(-a, a, npoints)
    y = numpy.zeros_like(v, dtype=complex)
    qe = q * R / (R - q * mu1 - g * q)
    be = 1 / qe + 1 / pe
    invle = 1 / (pe + qe) + g / R
    for i in range(v.size):
        yprime = acmax * (1 - (v[i] / a)**2)
        kum =  mpmath.hyp1f1(1j * kap, 1, 1j * yprime)        # kum = 1.0

        y[i] = kum * \
                            numpy.exp(1j * k * 0.5 * v[i]**2 * invle) * \
                            numpy.exp(- k * v[i] * kiny) * \
                            numpy.cos(k * v[i] * x / (q * pe * be))

    return numpy.trapz(y, x=v), be


# eq 30 for q=0 (integral in nu)
def integral_eq30(x, npoints=1000):
    v = numpy.linspace(-a, a, npoints)
    y = numpy.zeros_like(v, dtype=complex)

    for i in range(v.size):
        s = 0      # ????????????????????????????????????
        mu2prime = mu2 * gamma**2
        rho = poisson_ratio / (1 - poisson_ratio)
        a_2 = thickness / numpy.cos(teta) * \
              (numpy.cos(alfa) * numpy.sin(teta2) + rho * numpy.sin(alfa) * numpy.cos(teta2))
        # SP(x, v) eq 29
        SP = chizero * 0.25 * (t1 + t2) +\
             v[i] * omega - \
             (mu1 * x**2 + x * t1 * numpy.sin(teta1)) / (2 * R) - \
             (mu2prime * (v[i] - x)**2 - a_2 * gamma * (v[i] - x)) / (2 * R)  + \
             g * (a + x) * (v[i] - x) / R # equation 29 with u=x, nv

        yprime = acrist * gamma * (a ** 2 - v[i] ** 2) / (numpy.sin(2 * teta)) ** 2  # defined before eq 29
        kum = mpmath.hyp1f1(1j * kap, 1, 1j * yprime)

        y[i] = gamma / numpy.sqrt(lambda1 * p) * \
               numpy.exp(1j * k * (gamma * x - gamma * v[i] - s)**2 / (2 * p)) * \
               kum * \
               numpy.exp(1j * k * SP)



    return numpy.trapz(y, x=v)

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
    # common values
    #
    fig = 2 # 5  # use 100 for flat
    do_qscan = 0
    do_xscan = 0
    do_qzero = 1

    R = 2000
    poisson_ratio = 0.2201
    SG = 1.0

    npoints_q = 500
    npoints_x = 200

    use_automatic_chi = 1
    if fig == 2:
        photon_energy_in_keV = 80.0
        thickness = 1.0  # mm
        p = 0.0  # mm
        alfa_deg = 0.05
        qmax = 5000
        qposition = 0.001 # 2459.26
        factor = 0.9891
    elif fig == 3:
        photon_energy_in_keV = 80.0
        thickness = 1.0  # mm
        p = 20000.0  # mm
        alfa_deg = 1.5
        qmax = 8000
        qposition = 0.001 # -152.305 # 4675.35
        factor = 2 # 5 # 0.9934
        SG = -1
    elif fig in [4, 5, 6]:
        photon_energy_in_keV = 20.0
        thickness = 0.250  # mm
        p = 29000.0  # mm
        alfa_deg = 2.0 # ALWAYS POSITIVE; USE SG TO CHANGE SIGN
        qmax = 10000
        qposition = 0.001  # extra q position
        factor = 3 # 1.2 # 0.9891
        SG = 1
        npoints_x = 1000
    elif fig == 100: # flat
        photon_energy_in_keV = 20.0
        thickness = 0.1  # mm
        p = 1e6  # mm
        alfa_deg = 2.0 # ALWAYS POSITIVE; USE SG TO CHANGE SIGN
        qmax = 10000
        qposition = 1e6  # extra q position
        factor = 4500 # 1.2 # 0.9891
        R = 1e6
        SG = -1
        npoints_x = 1000
    else:
        raise NotImplementedError()

    #
    # asymmetric Fig. 2 (p=0)
    #
    if fig == 2:
        if use_automatic_chi:
            teta, chizero, chih = get_crystal_data("Si", hkl=[1, 1, 1], photon_energy_in_keV=photon_energy_in_keV,
                                                   verbose=False)
            lambda1 = codata.h * codata.c / codata.e / (photon_energy_in_keV * 1e3) * 1e3  # in mm
            chimh = -1j * chih
            chih2 = chih * chimh
            """
            >>>>>>>>>> chizero: (-1.5057156520331863e-07+2.935993357131102e-11j)
            >>>>>>>>>> chih: (-5.6659637601051554e-08-5.6630277667480283e-08j)
            >>>>>>>>>> chimh: (-5.6630277667480283e-08+5.6659637601051554e-08j)
            >>>>>>>>>> chih*chihbar: (6.417302019772712e-15-3.326184386578894e-18j)
            """
        else:
            teta = 1.4161222418 * numpy.pi / 180
            lambda1 = 0.01549817566e-6
            chizero = -0.150710e-6 + 1j * 0.290718e-10
            chih = numpy.conjugate(-5.69862 + 1j * 5.69574) * 1e-8
            chimh = -1j * chih
            chih2 = chih * chimh

        print("photon_energy_in_keV:", photon_energy_in_keV)
        print("lambda1 in mm:", lambda1)
        print("lambda1 in m, A:", lambda1 * 1e-3, lambda1 * 1e-3 * 1e10)
        print("CrystalSi 111")
        print(">>>>>>>>>> teta_deg:", teta * 180 / numpy.pi)
        print(">>>>>>>>>> p:", p)
        print(">>>>>>>>>> qposition:", qposition)
        print(">>>>>>>>>> R:", R)

        k = 2 * numpy.pi / lambda1
        h = 2 * k * numpy.sin(teta)

        print(">>>>>>>>>> chizero:", chizero)
        print(">>>>>>>>>> chih:", chih)
        print(">>>>>>>>>> chimh:", chimh)
        print(">>>>>>>>>> chih*chihbar:", chih2)


        u2 = 0.25 * chih2 * k**2
        raygam = R * numpy.cos(teta)
        kp = k * numpy.sin(2 * teta)
        kp2 = k * numpy.sin(2 * teta)**2

        #
        # TODO: Not working for alfa_deg=0
        #

        alfa = alfa_deg * numpy.pi / 180
        teta1 = alfa + SG * teta
        teta2 = alfa - SG * teta
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
        if do_qscan:
            print("Calculating q-scan...")
            t0 = time.time()
            qq = numpy.linspace(100, qmax, npoints_q)
            yy = numpy.zeros_like(qq)
            if alfa > 0 :
                for j in range(qq.size):
                    amplitude = sgplus_fig2(0, qq[j], npoints=500)
                    yy[j] = numpy.abs(amplitude ** 2 * att / (lambda1 * qq[j]))
            else:
                for j in range(qq.size):
                    amplitude = sgmoins_fig2(0, qq[j], npoints=500)
                    yy[j] = numpy.abs(amplitude ** 2 * att / (lambda1 * qq[j]))
            print("Time in calculating q-scan %f s" % (time.time() - t0))
            plot(qq, yy,
                 xtitle='q [mm]', ytitle="Intensity on axis", title="alfa=%g deg" % (alfa_deg),
                 show=0)
            qdyn, _, imax = get_max(qq, yy)
            qposition = qdyn


        # x-scan at finite q
        if do_xscan:
            print("Calculating x-scan...")
            xx = numpy.linspace(-0.0025, .0025, npoints_x)
            # xx = numpy.linspace(-a * factor * 100000, a * factor * 100000, npoints_x)
            yy = numpy.zeros_like(xx)
            for j in range(xx.size):
                yy[j] = numpy.abs(sgplus_fig2(xx[j], qposition) ** 2 * att / (lambda1 * qposition))
            plot(xx, yy,
                 xtitle='xi [mm]', ytitle="Intensity", title="alfa=%g deg q=%.1f mm" % (alfa_deg, qposition),
                 show=0)

        # (equation 23)
        # x-scan at q=0
        if do_qzero:
            print("Calculating x-scan at q=0... a=", a)
            omega = 0.25 * (t1 - t2) * chizero / a  # omega following the definition found after eq 22
            omega_real = numpy.real(omega)
            omega_imag = numpy.imag(omega)
            xc_over_q = omega_real - t1 * numpy.sin(alfa + teta) / (2 * R)

            # xx = numpy.linspace(-0.0025, .0025, 200)


            xx = numpy.linspace(-a * factor, a * factor, npoints_x)
            yy_amplitude = numpy.zeros_like(xx, dtype=complex)
            for j in range(xx.size):
                x = xx[j]
                # equation 23
                amplitude = numpy.exp((1j * k * chizero.real - k * chizero.imag) * 0.25 * (t1 + t2)) * \
                        mpmath.hyp1f1(1j * kap, 1, 1j * acmax * (1 - (x / a) ** 2)) * \
                        numpy.exp(-1j * x**2 * k * mu1 / 2 / R) *\
                        numpy.exp(1j * x * k * (omega_real - t1 * numpy.sin(teta1) / 2 / R)) * \
                        numpy.exp(- x * k * omega_imag)

                yy_amplitude[j] = amplitude

            plot(xx / a, numpy.abs(yy_amplitude)**2,
                 xtitle='xi/a [mm]', ytitle="Intensity at q=0", title="alfa=%g deg SG=%d" % (alfa_deg, SG),
                 show=0)


            #
            # write wofry wavefront
            #
            if True:
                filename = "tmp2016.h5"
                from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
                output_wavefront = GenericWavefront1D.initialize_wavefront_from_arrays(
                    1e-3 * xx, yy_amplitude, y_array_pi=None, wavelength=1e-10)
                output_wavefront.set_photon_energy(1e3 * photon_energy_in_keV)
                output_wavefront.save_h5_file(filename,
                                              subgroupname="wfr",intensity=True,phase=False,overwrite=True,verbose=False)
                print("File %s written to disk" % filename)


    #
    #
    #
    #
    #


    #
    # asymmetric Fig. 5 (p finite)
    #
    else: # if fig in [3,4,5,6]:
        if use_automatic_chi:
            teta, chizero, chih = get_crystal_data("Si", hkl=[1, 1, 1], photon_energy_in_keV=photon_energy_in_keV,
                                                   verbose=False)
            lambda1 = codata.h * codata.c / codata.e / (photon_energy_in_keV * 1e3) * 1e3  # in mm
            chimh = -1j * chih
            chih2 = chih * chimh
            """

            """
        else:
            if fig in [4,5,6]:
                teta = 5.67318 * (numpy.pi / 180)
                lambda1 = 0.619927e-7
                chih = (-0.922187 + 1j * 0.913110) * 1e-6
                chimh = 1j * chih
                chizero = -0.242370e-5 + 1j * 0.918640e-8
                chih2 = chih * chimh
            else:
                raise NotImplementedError()

        print("photon_energy_in_keV:", photon_energy_in_keV)
        print("lambda1 in mm:", lambda1)
        print("lambda1 in m, A:", lambda1 * 1e-3, lambda1 * 1e-3 * 1e10)
        print("CrystalSi 111")
        print(">>>>>>>>>> teta_deg:", teta * 180 / numpy.pi)
        print(">>>>>>>>>> p:", p)
        print(">>>>>>>>>> qposition:", qposition)
        print(">>>>>>>>>> R:", R)

        k = 2 * numpy.pi / lambda1
        h = 2 * k * numpy.sin(teta)

        print("chizero:", chizero)
        print(">>>>>>>>>> chih:", chih)
        print(">>>>>>>>>> chimh:", chimh)
        print(">>>>>>>>>> chih*chihbar:", chih2)


        u2 = 0.25 * chih2 * k**2
        raygam = R * numpy.cos(teta)
        kp = k * numpy.sin(2 * teta)
        kp2 = kp * numpy.sin(2 * teta)

        #
        # TODO: Not working for alfa_deg=0
        #

        alfa = alfa_deg * numpy.pi / 180
        teta1 = alfa + SG * teta
        teta2 = alfa - SG * teta
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
        com = numpy.sin(alfa) * (1 + gam1 * gam2 * (1 + poisson_ratio))
        kp3 = 0.5 * k * (gamma * a)**2
        # different from fig 2 (using SG)
        mu1 = SG * (numpy.cos(alfa) * 2 * fam1 * gam1 + numpy.sin(alfa) * (fam1**2 + poisson_ratio * gam1**2)) / (numpy.sin(2*teta)* numpy.cos(teta))
        mu2 = -SG * (numpy.cos(alfa) * 2 * fam2 * gam2 + numpy.sin(alfa) * (fam2**2 + poisson_ratio * gam2**2)) / (numpy.sin(2*teta)* numpy.cos(teta))
        a1 = SG * (0.5 * thickness / numpy.cos(teta)) * (numpy.cos(alfa) * numpy.sin(teta1) + poisson_ratio * numpy.sin(alfa) * numpy.cos(teta1))
        a2 = -SG * (0.5 * thickness / numpy.cos(teta)) * (numpy.cos(alfa) * numpy.sin(teta2) + poisson_ratio * numpy.sin(alfa) * numpy.cos(teta2))
        acrist = -SG * h * com / R   # A in Eq 17

        acmax = acrist * s2max
        g = gamma * acrist * R / kp2
        kap = u2max / acmax   # beta = Omega / A TODO acmax is zero when alfa is zero!!!!!!!!!!!!!!!!!!

        # WARNING DIFFERNT FROM Fig 2 (+p)
        pe = p * R / (gamma**2 * (R + p * mu2) - g * p)
        print(">>>>>>>>>> a: ", a)
        # print(">>>>> block 1: ", poisson_ratio, teta, alfa, lambda1, k, h, chizero, chih, chimh, chih2, u2, thickness, p, R, raygam, kp, kp2)
        #
        # print(">>>>> block 2: ", SG, teta1, teta2, fam1, fam2, gam1, gam2, t1, t2, qpoly)
        # print("att", att, s2max, u2max, gamma, a, kin, kinx, kiny, com, kp3, mu1, mu2, a1, a2, acrist, acmax, g, kap, pe)

        # q-scan
        if do_qscan:
            print("Calculating q-scan...")
            t0 = time.time()
            qq = numpy.linspace(100, qmax, npoints_q)
            yy = numpy.zeros_like(qq)
            if alfa > 0 :
                for j in range(qq.size):
                    amplitude, be = sgplus_fig5(0, qq[j], npoints=500)
                    amplitude *= numpy.sqrt(att / (lambda1 * qq[j] * p * be))
                    # yy[j] = numpy.abs(amplitude ** 2 * att / (lambda1 * qq[j] * p * be))
                    yy[j] = numpy.abs(amplitude) ** 2
            else:
                for j in range(qq.size):
                    amplitude, be = sgmoins_fig5(0, qq[j], npoints=500)
                    amplitude *= numpy.sqrt(att / (lambda1 * qq[j] * p * be))
                    yy[j] = numpy.abs(amplitude) ** 2
            print("Time in calculating q-scan %f s" % (time.time() - t0))
            plot(qq, yy,
                 xtitle='q [mm]', ytitle="Intensity on axis", title="alfa=%g deg; SG=%d" % (alfa_deg, SG),
                 show=0)
            qdyn, _, imax = get_max(qq, yy)
            qposition = qdyn


        # x-scan at finite q
        if do_xscan:
            print("Calculating x-scan...")
            # xx = numpy.linspace(-0.005, .005, 200)
            xx = numpy.linspace(-factor * a, factor * a, npoints_x)
            yy_amplitude = numpy.zeros_like(xx, dtype=complex)
            for j in range(xx.size):
                amplitude, be = sgplus_fig5(xx[j], qposition, npoints=500)
                amplitude *= numpy.sqrt(att / (lambda1 * qposition * p * be))
                # omitted phase (see just after equation 30)
                s = 0
                amplitude *= numpy.exp(1j * k * xx[j]**2 / 2 / qposition) * \
                             numpy.exp(1j * k * s**2 / 2 / p) * \
                             numpy.exp(1j * k * chizero.real * (t1 + t2) / 4)
                # omitted phase (see just before equation 31)
                m = g * a / R + gamma * (s / p + a**2 / 2 / R) ## CHECK, shown after eq 30
                amplitude *= numpy.exp(- 1j * (k / 2 / be) *\
                                       ( xx[j] / qposition + t1 * numpy.sin(teta1) / 2 / R + m)**2
                                       )
                yy_amplitude[j] = amplitude
            plot(xx, numpy.abs(yy_amplitude)**2,
                 xtitle='x [mm]', ytitle="Intensity", title="alfa=%g deg SG=%d q=%.1f mm" % (alfa_deg, SG, qposition),
                 grid=1, show=0)

            # write wofry wavefront
            if 1:
                filename = "tmp2016.h5"
                from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
                output_wavefront = GenericWavefront1D.initialize_wavefront_from_arrays(
                    1e-3 * xx, yy_amplitude, y_array_pi=None, wavelength=1e-10)
                output_wavefront.set_photon_energy(1e3 * photon_energy_in_keV)
                output_wavefront.save_h5_file(filename,
                                              subgroupname="wfr",intensity=True,phase=False,overwrite=True,verbose=False)
                print("File %s written to disk" % filename)

        # (equation 30 without propagation)
        # x-scan at q=0
        if do_qzero:
            print("Calculating x-scan at q=0... a=", a)
            t0 = time.time()
            omega = 0.25 * (t1 - t2) * chizero / a  # omega following the definition found after eq 22
            xx = numpy.linspace(-a * factor, a * factor, npoints_x)

            xx = numpy.linspace(-0.6, 0.8, npoints_x)
            yy_amplitude = numpy.zeros_like(xx, dtype=complex)
            for j in range(xx.size):
                x = xx[j]
                amplitude = integral_eq30(xx[j], npoints=500)
                yy_amplitude[j] = amplitude

            print("Calculation time: ", time.time() - t0)
            plot(xx, numpy.abs(yy_amplitude)**2,
                 xtitle='x [mm]', ytitle="Intensity", title="alfa=%g deg SG=%d  q=zero" % (alfa_deg, SG),
                 show=0)

            # write wofry wavefront
            if 1:
                filename = "tmp2016_q0.h5"
                from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
                output_wavefront = GenericWavefront1D.initialize_wavefront_from_arrays(
                    1e-3 * xx, yy_amplitude, y_array_pi=None, wavelength=1e-10)
                output_wavefront.set_photon_energy(1e3 * photon_energy_in_keV)
                output_wavefront.save_h5_file(filename,
                                              subgroupname="wfr",intensity=True,phase=False,overwrite=True,verbose=False)
                print("File %s written to disk" % filename)

    plot_show()



    plot_show()