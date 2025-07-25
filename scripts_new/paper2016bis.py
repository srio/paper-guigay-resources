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
def integrate_sgplus(x, q, k=0.0, a=0.0, mu1=0.0, R=0.0, beta=0.0, acmax=0.0, omega_imag=0.0,
                    gamma=0, teta=0, Aeq17=0, # NEWWWWWWWWWWWWWWW
                     ):
    v = numpy.linspace(0, a, 1000)
    yy = numpy.zeros_like(v, dtype=complex)

    invle = 1 / q - mu1 / R

    for i in range(v.size):
        # y is defined just before eq 23
        y = Aeq17 *  gamma * (a**2 - v[i]**2) / (numpy.sin(2 * teta))**2
        # y =  acmax * (1 - (v[i] / a)**2)
        # print(">>>> y: ", y0, y, Aeq17, gamma, a, teta)

        yy[i] = mpmath.hyp1f1(1j * beta, 1, 1j * y) * \
            numpy.exp(1j * k * 0.5 * v[i]**2 * invle) * numpy.cos(k * v[i] * (x / q - 1j * omega_imag))
    return 2 * numpy.trapz(yy, x=v)

def integrate_sgplus_notcentered(x, q, k=0.0, a=0.0, mu1=0.0, R=0.0, beta=0.0, acmax=0.0, omega_imag=0.0,
                    gamma=0, teta=0, Aeq17=0, xc_over_q=0, # NEWWWWWWWWWWWWWWW
                     ):
    v = numpy.linspace(-a, a, 2000)
    yy = numpy.zeros_like(v, dtype=complex)

    invle = 1 / q - mu1 / R

    for i in range(v.size):
        # y is defined just before eq 23
        y = Aeq17 *  gamma * (a**2 - v[i]**2) / (numpy.sin(2 * teta))**2
        # y =  acmax * (1 - (v[i] / a)**2)
        # print(">>>> y: ", y0, y, Aeq17, gamma, a, teta)

        # print(">>>>", xc_over_q * q, numpy.exp(k * v[i] * (-1j * (x - xc_over_q * q) / q - omega_imag)))
        yy[i] = mpmath.hyp1f1(1j * beta, 1, 1j * y) * \
            numpy.exp(1j * k * 0.5 * v[i]**2 * invle) * \
            numpy.exp(k * v[i] * (-1j * (x - xc_over_q * q) / q - omega_imag))
    return numpy.trapz(yy, x=v)


# equation 23
def sgplus_qzero(x, q, k=0.0, a=0.0, mu1=0.0, R=0.0, beta=0.0, acmax=0.0, omega_imag=0.0,
                 gamma=0, teta=0, Aeq17=0, xc_over_q=0,
                 t1=0, t2=0, teta1=0, omega_real=0, chizero=0, # NEWWWWWWWWWWWWWWW
                 ):

    # y is defined just before eq 23
    y = Aeq17 *  gamma * (a**2 - x**2) / (numpy.sin(2 * teta))**2

    yy = mpmath.hyp1f1(1j * beta, 1, 1j * y) * \
        numpy.exp( (1j * k * chizero.real - k * chizero.imag) * 0.25 * (t1 + t2) ) * \
        numpy.exp( -1j * x**2 * k * mu1 / 2 / R + \
                   1j * x * k * (omega_real - t1 * numpy.sin(teta1) / 2 / R) - \
                   k * x * omega_imag)
    return yy


def get_max(xx, yy, verbose=1):
    i = numpy.argmax(yy)
    if verbose: print("Maximum found at x=%g, y=%g" % (xx[i],yy[i]))
    return xx[i], yy[i], i

def run_asymmetric(fig=2, use_automatic_chi=1, plot_qscan=1, plot_xscan=1, plot_qzero=1):
    # teta = 1.4161222418 * numpy.pi / 180
    # lambda1 = 0.01549817566e-6
    # chizero = -0.150710e-6 + 1j * 0.290718e-10
    # chih = numpy.conjugate(-5.69862 + 1j * 5.69574) * 1e-8


    if fig == 2:
        photon_energy_in_keV = 80.0
        thickness = 1.0 # mm
        p = 0.0 # mm
        qmax = 5000
        qposition = 2466.0  # extra q position
        # qposition = 1679
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
            chihm = -1j * chih
        else:# # 20 keV
            teta = numpy.radians(5.67310)
            lambda1 = 0.619927e-7
            chizero = -0.242370e-5 + 1j * 0.918640e-8
            chih = (-0.922187 + 1j * 0.9131110) * 1e-6
            chihm = -1j * chih

    print("photon_energy_in_keV:", photon_energy_in_keV)
    print("CrystalSi 111")
    print(">>>>>>>>>> teta_deg:", teta * 180 / numpy.pi)
    print(">>>>>>>>>> chizero:", chizero)
    print(">>>>>>>>>> chih:", chih)
    print(">>>>>>>>>> lambda1:", lambda1)


    k = 2 * numpy.pi / lambda1
    h = 2 * k * numpy.sin(teta)

    chimh = -1j * chih
    chih2 = chih * chimh
    Omega = 0.25 * chih2 * k**2  #  this is Omega (see just after eq 7b)
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
    u2max = Omega * s2max
    gamma = t2 / t1
    # a is defined after eq 22:
    # a = numpy.sin(2 * teta) * t1 * 0.5
    a = (t1 + t2) / 2 * numpy.sin(2*teta) / (1+gamma)

    omega = 0.25 * (t1 - t2) * chizero / a # omega following the definition found after eq 22

    print(">>>> omega=", omega, chizero * (1 - gamma) / 2 / numpy.sin(2 * teta))
    omega_real = numpy.real(omega)
    omega_imag = numpy.imag(omega)
    # com = numpy.sin(alfa) * (1 + gam1 * gam2 * (1 + poisson_ratio))
    kp3 = 0.5 * k * (gamma * a)**2
    mu1 = (numpy.cos(alfa) * 2 * fam1 * gam1 + numpy.sin(alfa) * (fam1**2 + poisson_ratio * gam1**2)) / (numpy.sin(2*teta)* numpy.cos(teta))
    mu2 = (numpy.cos(alfa) * 2 * fam2 * gam2 + numpy.sin(alfa) * (fam2**2 + poisson_ratio * gam2**2)) / (numpy.sin(2*teta)* numpy.cos(teta))
    a1 = (0.5 * thickness / numpy.cos(teta)) * (numpy.cos(alfa) * numpy.sin(teta1) + poisson_ratio*numpy.sin(alfa) * numpy.cos(teta1))
    a2 = (0.5 * thickness / numpy.cos(teta)) * (numpy.cos(alfa) * numpy.sin(teta2) + poisson_ratio*numpy.sin(alfa) * numpy.cos(teta2))
    acrist = -h * numpy.sin(alfa) * (1 + gam1 * gam2 * (1 + poisson_ratio)) / R # A in eq 17
    Aeq17 = acrist

    xc_over_q = omega_real - t1 * numpy.sin(alfa + teta) / (2 * R)

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
        qq = numpy.linspace(100, qmax, 100)
        yy = numpy.zeros_like(qq)
        for j in range(qq.size):
            amplitude = integrate_sgplus(0, qq[j],
                                   k=k, a=a, mu1=mu1, R=R,
                                   beta=beta, acmax=acmax, omega_imag=omega_imag,
                                   gamma=gamma, teta=teta, Aeq17=Aeq17, # NEWWWWWWWWWW
                                   ) / numpy.sqrt(numpy.abs(lambda1 * qq[j]))
            yy[j] = att * numpy.abs(amplitude)**2
        plot(qq, yy,
             xtitle='q [mm]', ytitle="Intensity on axis", title="alfa=%g deg" % (alfa_deg),
             show=0)
        qdyn, _, imax = get_max(qq, yy)
        qposition = qdyn
    else:
        qposition = 0.0

    # x-scan centered
    if plot_xscan:
        print("Calculating x-scan (centered)...")
        xx = numpy.linspace(-0.0025, .0025, 200)
        # xx = numpy.linspace(-a, a, 1000)
        yy = numpy.zeros_like(xx)
        for j in range(xx.size):
            amplitude = integrate_sgplus(xx[j], qposition,
                                   k=k, a=a, mu1=mu1, R=R,
                                   beta=beta, acmax=acmax, omega_imag=omega_imag,
                                   gamma=gamma, teta=teta, Aeq17=Aeq17,  # NEWWWWWWWWWW
                                   ) / numpy.sqrt(numpy.abs(lambda1 * qposition))
            yy[j] = att * numpy.abs(amplitude)**2
        plot(xx, yy, xx + xc_over_q * qposition, yy,
             xtitle='x - xc[mm]', ytitle="Intensity on axis", title="alfa=%g deg" % (alfa_deg),
             show=0)

        if 0: # test using long equation 24
            print("Calculating x-scan (not centered, longer calculation)...")
            # xx = numpy.linspace(-0.0025, .0025, 200)
            xx1 = numpy.linspace(-0.015, -0.005, 200)
            yy1 = numpy.zeros_like(xx1)
            for j in range(xx1.size):
                amplitude = integrate_sgplus_notcentered(xx1[j], qposition,
                                       k=k, a=a, mu1=mu1, R=R,
                                       beta=beta, acmax=acmax, omega_imag=omega_imag,
                                       gamma=gamma, teta=teta, Aeq17=Aeq17, xc_over_q=xc_over_q # NEWWWWWWWWWW
                                       ) / numpy.sqrt(numpy.abs(lambda1 * qposition))
                yy1[j] = att * numpy.abs(amplitude)**2
            plot(xx1, yy1, xx, yy, xx + xc_over_q * qposition, yy,
                 xtitle='x [mm] (NOT CENTERED)', ytitle="Intensity on axis", title="alfa=%g deg" % (alfa_deg),
                 show=0)

    if plot_qzero:
        print("Calculating x-scan for q=0...")
        xx1 = numpy.linspace(-a, a, 200)
        yy1 = numpy.zeros_like(xx1)
        for j in range(xx1.size):
            # amplitude = sgplus_qzero(xx1[j], qposition,
            #                        k=k, a=a, mu1=mu1, R=R,
            #                        beta=beta, acmax=acmax, omega_imag=omega_imag,
            #                        gamma=gamma, teta=teta, Aeq17=Aeq17, xc_over_q=xc_over_q, # NEWWWWWWWWWW
            #                        t1=t1, t2=t2, teta1=teta1, omega_real=omega_real, chizero=chizero)

            # equation 23
            x = xx1[j]
            y = Aeq17 * gamma * (a ** 2 - x ** 2) / (numpy.sin(2 * teta)) ** 2
            # y = acmax * (1 - (x / a) ** 2)
            amplitude = mpmath.hyp1f1(1j * beta, 1, 1j * y)
            # amplitude *= numpy.exp((- k * chizero.imag) * 0.25 * (t1 + t2))
            # amplitude *= numpy.exp(- k * x * omega_imag)
            print(x, amplitude, numpy.abs(amplitude)**2)


            #* \
                 # numpy.exp((1j * k * chizero.real - k * chizero.imag) * 0.25 * (t1 + t2)) * \
                 # numpy.exp(-1j * x ** 2 * k * mu1 / 2 / R + \
                 #           1j * x * k * (omega_real - t1 * numpy.sin(teta1) / 2 / R) - \
                 #           k * x * omega_imag)
            yy1[j] = numpy.abs(amplitude)**2

        plot(xx1, yy1,
             xtitle='x [mm]', ytitle="Intensity on axis at q=0", title="alfa=%g deg" % (alfa_deg),
             show=0)

if __name__ == "__main__":

    #
    # for symmetric (alpha=0) use paper2-16.py
    #


    #
    # asymmetric Fig. 2
    #
    run_asymmetric(fig=2, use_automatic_chi=0, plot_qscan=1, plot_xscan=0, plot_qzero=0)

    plot_show()