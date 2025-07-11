#
# Figs in Acta Cryst. (2016). A72, 489–499 Guigay and Ferrero Dynamical focusing by bent Laue crystals
#

import numpy
from scipy.special import jv as BesselJ
import scipy.constants as codata
from crystal_data import get_crystal_data
from srxraylib.plot.gol import plot, set_qt, plot_show

# curved symmetric Laue, source at finite p
# Eq. 13 in doi:10.1107/S0108767312044601  (2013) or
# Eq. 23 in https://doi.org/10.1107/S1600577521012480 (2022)
def guigay2013_eq13_Dxq(x, q, k=0.0, Z=0.0, a=0.0, RcosTheta=0.0, p=0.0, on_x_at_maximum=0, npoints=2000):
    v = numpy.linspace(-a, a, npoints)
    y = numpy.zeros_like(v, dtype=complex)
    pe = p * RcosTheta / (RcosTheta + p)
    qe = q * RcosTheta / (RcosTheta - q)
    lesym = pe + qe
    # pe + qe # RcosTheta * q / (RcosTheta - q)
    # fact_eta = (2 * x * qe / q / lesym +  a / RcosTheta) # diverges at q=0
    fact_eta = (2 * x * (RcosTheta / (RcosTheta - q)) / lesym + a / RcosTheta)
    if on_x_at_maximum:
        x = 0.0
        fact_eta = 0.0

    for i in range(v.size):
        arg_bessel = Z * numpy.sqrt((a + v[i]) * (a - v[i]))
        y[i] = BesselJ(0, arg_bessel) * numpy.exp(1j * k * 0.5 * (v[i] ** 2 / lesym - v[i] * fact_eta))

    return numpy.trapz(y, x=v)

# eq 10  in doi:10.1107/S0108767312044601
def guigay2013_eq10_Dup0q0(u, k=0.0, Z=0.0, a=0.0, RcosTheta=0.0, p=0.0, npoints=2000):
    return BesselJ(0, Z * numpy.sqrt(a ** 2 - u ** 2)) * numpy.exp(-1j * k * 0.5 * u * (a + u) / RcosTheta)

# eq 12  in doi:10.1107/S0108767312044601  (2013)
def guigay2013_eq12_Du(u, k=0.0, Z=0.0, a=0.0, RcosTheta=0.0, p=0.0):
    v = numpy.linspace(-a, +a, 2000)
    y = numpy.zeros_like(v, dtype=complex)
    lambda1 = 2 * numpy.pi / k
    pe = 1 / (1 / p + 1 / RcosTheta)
    for i in range(v.size):
        arg1 = a**2 - v[i]**2
        if arg1 < 0: arg1 = 0

        y[i] = 1 / numpy.sqrt(1j * lambda1 * p) * \
               numpy.exp(1j * k * 0.5 * (u - v[i]) ** 2 / pe) * \
               numpy.exp(-1j * k * 0.5 / RcosTheta * a * v[i]) * \
               BesselJ(0, Z * numpy.sqrt(arg1))
    return numpy.trapz(y, x=v) * numpy.exp(-1j * k * 0.5 * u**2 / RcosTheta)

# curved symmetric Laue, source at finite p, q=0
# First equation in section 4.2  in doi:10.1107/S0108767312044601  (2013)
def guigay2013_eq12_integralVStau(u, k=0.0, Z=0.0, a=0.0, RcosTheta=0.0, p=0.0, npoints=2000):
    t = numpy.linspace(u - a, u + a, npoints)
    y = numpy.zeros_like(t, dtype=complex)
    lambda1 = 2 * numpy.pi / k
    for i in range(t.size):
        arg1 = a**2 - (u - t[i])**2
        if arg1 < 0: arg1 = 0
        y[i] = 1 / numpy.sqrt(1j * lambda1 * p) * \
               numpy.exp(1j * k * 0.5 * t[i] ** 2 / p) * \
               numpy.exp(1j * k * 0.5 / RcosTheta * (t[i] * (t[i] + a) - u * (u + a))) * \
               BesselJ(0, Z * numpy.sqrt(arg1))
    return numpy.trapz(y, x=t)

def get_max(xx, yy, verbose=1):
    i = numpy.argmax(yy)
    if verbose: print("Maximum found at x=%g, y=%g" % (xx[i],yy[i]))
    return xx[i], yy[i], i

def run_symmetric(fig=2, plot_qscan=1, plot_xscan=1, plot_qzero=0, factor=1.0, qposition=None, use_automatic_chi=True):
    #
    # inputs (in mm) ===========================================================
    #

    if fig == 2:
        photon_energy_in_keV = 80.0
        thickness = 1.0 # mm
        p = 0.0 # mm
        qmax = 5000
        if qposition is None: qposition = 2466.0  # extra q position
        # if qposition is None: qposition = 1679
    elif fig == 3:
        photon_energy_in_keV = 80.0
        thickness = 1.0  # mm
        p = 20000.0  # mm
        qmax = 4000
        if qposition is None: qposition = 912.826
    elif fig in [4, 5, 6]:
        photon_energy_in_keV = 20.0
        thickness = 0.250 # mm
        p = 29000.0 # mm
        qmax = 10000
        if use_automatic_chi:
            if qposition is None: qposition = 3556.5 # extra q position
        else:
            if qposition is None: qposition = 550
        # if qposition is None: qposition = 3579 # 0.00001
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
        if fig in [2, 3]:
            teta = numpy.radians(1.4161222418)
            lambda1 = 0.01549817556e-6
            chizero = -0.150710e-6 + 1j * 0.290718e-10
            chih = (-5.69862 + 1j * 5.69574) * 1e-8
            chihm = 1j * chih
        elif fig in [4, 5, 6]: # # 20 keV
            teta = numpy.radians(5.67310)
            lambda1 = 0.619927e-7
            chizero = -0.242370e-5 + 1j * 0.918640e-8
            chih = (-0.922187 + 1j * 0.9131110) * 1e-6
            chihm = 1j * chih
        else:
            raise NotImplementedError()


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
    RcosTheta = R * numpy.cos(teta)
    print("RcosTheta", RcosTheta)

    #
    # this part uses the source at p=finite, R finite, and calculates I(xi, qdyn) and I(focus,q) to obtain qdyn
    #


    if plot_qscan:
        qq = numpy.linspace(100, qmax, 1000)
        yy_ampl = numpy.zeros_like(qq, dtype=complex)
        yy = numpy.zeros_like(qq)


        for j in range(qq.size):
            yya = guigay2013_eq13_Dxq(0, qq[j], k=k, Z=Zsym, a=asym, RcosTheta=RcosTheta, p=p, on_x_at_maximum=1)
            yya *= numpy.sqrt(attsym / (lambda1 * numpy.abs(p + qq[j])))
            yy_ampl[j] = yya
        yy = numpy.abs(yy_ampl)**2

        if p == 0.0:
            q_lensequation = 0
        else:
            q_lensequation = 1.0 / (2 / RcosTheta - 1 / p)


        qdyn, _, imax  = get_max(qq, yy)
        plot(qq, yy,
             [q_lensequation, q_lensequation], [0, yy.max()], legend=['Dynamical theory', 'Lens equation'],
             xtitle='q [mm]', ytitle="Intensity on axis", title="R=%g mm p=%g mm" % (R, p),
             xrange=[qq.min(),qq.max()], show=0)
    else:
        qdyn = 1000.0

    if plot_xscan: # lateral scan (I vs chi)
        xi = numpy.linspace(-asym * factor, asym * factor, 1000)
        yy1_ampl = numpy.zeros_like(xi, dtype=complex)
        for j in range(xi.size):
            yy1_ampl[j] = numpy.sqrt(attsym / (lambda1 * numpy.abs(p + qdyn))) * \
                     guigay2013_eq13_Dxq(xi[j], qdyn, k=k, Z=Zsym, a=asym, RcosTheta=RcosTheta, p=p)
        yy1 =  numpy.abs(yy1_ampl) ** 2

        yy2_ampl = numpy.zeros_like(xi, dtype=complex)
        for j in range(xi.size):
            yy2_ampl[j] = numpy.sqrt(attsym / (lambda1 * (p + qposition))) * \
                          guigay2013_eq13_Dxq(xi[j], qposition, k=k, Z=Zsym, a=asym, RcosTheta=RcosTheta, p=p)

        yy2 =  numpy.abs(yy2_ampl) ** 2


        plot(xi, yy1, xi, yy2, legend=['q=%.1f mm' % qdyn, 'q=%0.1f mm' % qposition],
             xtitle='xi [mm]', ytitle="Intensity",
             title="xi scan R=%g mm, p=%.1f, ReflInt = %g" % (R, p, yy1.sum() * (xi[1] - xi[0])),
             show=0)
        # write wofry wavefront
        if 1:
            filename = "tmp2016.h5"
            from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
            output_wavefront = GenericWavefront1D.initialize_wavefront_from_arrays(
                1e-3 * xi, yy2_ampl, y_array_pi=None, wavelength=1e-10)
            output_wavefront.set_photon_energy(1e3 * photon_energy_in_keV)
            output_wavefront.save_h5_file(filename,
                                          subgroupname="wfr", intensity=True, phase=False, overwrite=True,
                                          verbose=False)
            print("File %s written to disk AT A DISTANCE=%f mm" % (filename, qposition))

    if plot_qzero: # lateral scan (I vs chi)
        yy_ampl = numpy.zeros(2000, dtype=complex)
        if p == 0: # EQUATION 10
            x0 = 0
            xi = numpy.linspace(-asym, asym, yy_ampl.size)
            # for j in range(xi.size):
            #     yy_ampl[j] = BesselJ(0, Zsym * numpy.sqrt(asym**2 - xi[j]**2)) * \
            #        numpy.exp(-1j * k * 0.5 * xi[j] * (asym + xi[j]) / RcosTheta)
            yy_ampl = guigay2013_eq10_Dup0q0(xi, k=k, Z=Zsym, a=asym, RcosTheta=RcosTheta, p=p)
        else:  # EQUATION 12
            x0 = -asym / (1 / p + 1 / RcosTheta) / 2 / RcosTheta
            xi = numpy.linspace(-1.0 * asym + x0, 1.0 * asym + x0, yy_ampl.size)

            for j in range(xi.size):
                yy_ampl[j] = guigay2013_eq12_Du(xi[j], k=k, Z=Zsym, a=asym, RcosTheta=RcosTheta, p=p)
                # Eq 12 before change of variables
                # yy_ampl[j] = guigay2013_eq12_integralVStau(xi[j], k=k, Z=Zsym, a=asym, RcosTheta=RcosTheta, p=p)

        yy1 = numpy.abs(yy_ampl)**2
        plot(xi / asym, yy1,
             xtitle='x/a', ytitle="Intensity",
             title="xi scan R=%g mm, p=%.1f, q=zero, a=%g mm" % (R, p, asym),
             show=0)

        plot(xi, yy1,
             xi - x0, yy1,
             legend=['vs x', 'vs x-x0; x0=%f' % x0],
             xtitle='x [mm]', ytitle="Intensity",
             title="xi scan R=%g mm, p=%.1f, q=zero, a=%g mm" % (R, p, asym),
             show=0)


        # write wofry wavefront
        if 1:
            filename = "tmp2016_q0.h5"
            from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
            output_wavefront = GenericWavefront1D.initialize_wavefront_from_arrays(
                1e-3 * xi, yy_ampl, y_array_pi=None, wavelength=1e-10)
            output_wavefront.set_photon_energy(1e3 * photon_energy_in_keV)
            output_wavefront.save_h5_file(filename,
                                          subgroupname="wfr", intensity=True, phase=False, overwrite=True,
                                          verbose=False)
            print("File %s written to disk" % filename)

if __name__ == "__main__":


    # run_symmetric(fig=2, plot_xscan=1, use_automatic_chi=0)
    # run_symmetric(fig=2, plot_xscan=1, use_automatic_chi=1)

    # note that with use_automatic_chi=0 the maximum is at the second main peak (like in the paper)
    # but for use_automatic_chi=1 the maximum is at the first peak.
    # run_symmetric(fig=4, plot_xscan=1, use_automatic_chi=0)
    # run_symmetric(fig=4, plot_xscan=1, use_automatic_chi=1)

    run_symmetric(fig=2, plot_qscan=1, plot_xscan=1, plot_qzero=0, factor=1.0, qposition=None,  use_automatic_chi=0)

    plot_show()