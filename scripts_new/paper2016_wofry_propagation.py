import numpy
from scipy.special import jv as BesselJ
import scipy.constants as codata

#
# Fig. 4-6 in 2016
#


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


if __name__ == "__main__":
    from crystal_data import get_crystal_data
    from srxraylib.plot.gol import plot, set_qt, plot_show



    # photon_energy_in_keV = 17.0
    photon_energy_in_keV = 20.0

    filename = "crystal_amplitude_%d.h5" % (1e3 * photon_energy_in_keV)


    do_calculate = 1

    if do_calculate:
        set_qt()

        #
        # inputs (in mm) ===========================================================
        #

        thickness = 0.250 # mm


        p = 29e3 # mm
        rayon = 2000.0 # mm

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

        raygam = rayon * numpy.cos(teta)
        print("raygam", raygam)


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
        # this part uses the source at p=finite, R finite, and calculates I(xi, qdyn) and I(focus,q) to obtain qdyn
        #
        if True:
            qq = numpy.linspace(0, 6000, 1000)
            yy = numpy.zeros_like(qq)

            for j in range(qq.size):
                yy[j] = attsym / (lambda1 * (p + qq[j])) * numpy.abs(integral_psisym(0, qq[j],
                                                            k=k, Z=Zsym, a=asym, RcosTheta=raygam, p=p, on_x_at_maximum=1)) ** 2

            q_lensequation = 1.0 / (2 / raygam - 1 / p)
            plot(qq, yy,
                 [q_lensequation, q_lensequation], [0, yy.max()], legend=['Dynamical theory', 'Lens equation'],
                 xtitle='q [mm]', ytitle="Intensity on axis", title="R=%g mm p=%g mm" % (rayon, p), show=0)
            qdyn, _, imax  = get_max(qq, yy)


            xi = numpy.linspace(-2*asym, 2*asym, 2000)
            yy1 = numpy.zeros_like(xi)
            for j in range(xi.size):
                yy1[j] = attsym / (lambda1 * (p + qdyn)) * \
                         numpy.abs(integral_psisym(xi[j], qdyn,
                                                   k=k, Z=Zsym, a=asym, RcosTheta=raygam, p=p)) ** 2

            yy2_ampl = numpy.zeros_like(xi, dtype=complex)
            yy2 = numpy.zeros_like(xi)
            for j in range(xi.size):
                yy2_ampl[j] = numpy.sqrt(attsym / (lambda1 * (p + 0.0))) * \
                              integral_psisym(xi[j], 0.0, k=k, Z=Zsym, a=asym, RcosTheta=raygam, p=p)
                yy2[j] =  numpy.abs(yy2_ampl[j]) ** 2

            #
            # write wofry wavefront
            #
            from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D
            output_wavefront = GenericWavefront1D.initialize_wavefront_from_arrays(
                1e-3 * xi, yy2_ampl, y_array_pi=None, wavelength=1e-10)
            output_wavefront.set_photon_energy(1e3 * photon_energy_in_keV)
            output_wavefront.save_h5_file(filename,
                                          subgroupname="wfr",intensity=True,phase=False,overwrite=True,verbose=False)

            plot(xi, yy1, xi, yy2, legend=['q=%.1f' % qdyn, 'q=0'],
                 xtitle='xi [mm]', ytitle="Intensity",
                 title="xi scan R=%g mm, p=%.1f, ReflInt = %g" % (rayon, p, yy1.sum() * (xi[1] - xi[0])),
                 show=0)

    #
    # propagate
    #

    #
    # Import section
    #
    import numpy

    from syned.beamline.beamline_element import BeamlineElement
    from syned.beamline.element_coordinates import ElementCoordinates
    from wofry.propagator.propagator import PropagationManager, PropagationElements, PropagationParameters

    from wofry.propagator.wavefront1D.generic_wavefront import GenericWavefront1D

    from wofryimpl.propagator.propagators1D.fresnel import Fresnel1D
    from wofryimpl.propagator.propagators1D.fresnel_convolution import FresnelConvolution1D
    from wofryimpl.propagator.propagators1D.fraunhofer import Fraunhofer1D
    from wofryimpl.propagator.propagators1D.integral import Integral1D
    from wofryimpl.propagator.propagators1D.fresnel_zoom import FresnelZoom1D
    from wofryimpl.propagator.propagators1D.fresnel_zoom_scaling_theorem import FresnelZoomScaling1D

    from srxraylib.plot.gol import plot, plot_image

    plot_from_oe = 0  # set to a large number to avoid plots

    ##########  SOURCE ##########

    #
    # create output_wavefront
    #


    output_wavefront = GenericWavefront1D.load_h5_file(filename)


    if plot_from_oe <= 0: plot(output_wavefront.get_abscissas(), output_wavefront.get_intensity(), title='SOURCE', show=0)

    ##########  OPTICAL SYSTEM ##########

    ##########  OPTICAL ELEMENT NUMBER 1 ##########

    input_wavefront = output_wavefront.duplicate()
    from wofryimpl.beamline.optical_elements.ideal_elements.screen import WOScreen1D

    optical_element = WOScreen1D()

    # drift_before 0.6517 m
    #
    # propagating
    #
    #
    propagation_elements = PropagationElements()
    beamline_element = BeamlineElement(optical_element=optical_element,
                                       coordinates=ElementCoordinates(p=0.552553, q=0.000000,
                                                                      angle_radial=numpy.radians(0.000000),
                                                                      angle_azimuthal=numpy.radians(0.000000)))
    propagation_elements.add_beamline_element(beamline_element)
    propagation_parameters = PropagationParameters(wavefront=input_wavefront, propagation_elements=propagation_elements)
    # self.set_additional_parameters(propagation_parameters)
    #
    propagation_parameters.set_additional_parameters('magnification_x', 1.0)
    #
    propagator = PropagationManager.Instance()
    try:
        propagator.add_propagator(FresnelZoom1D())
    except:
        pass
    output_wavefront = propagator.do_propagation(propagation_parameters=propagation_parameters,
                                                 handler_name='FRESNEL_ZOOM_1D')

    #
    # ---- plots -----
    #
    if plot_from_oe <= 1: plot(output_wavefront.get_abscissas(), output_wavefront.get_intensity(),
                               title='OPTICAL ELEMENT NR 1', show=0)

    plot_show()