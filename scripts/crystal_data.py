#
# example of using xraylib to get crystal data
#
# see:
#       http://dx.doi.org/10.1016/j.sab.2011.09.011   (paper)
#       https://github.com/tschoonj/xraylib/          (code)
#       http://lvserver.ugent.be/xraylib-web          (web interface, but crystals not included!)
#




#
#  import block
#
import xraylib
import numpy as np
import scipy.constants.codata


def get_crystal_data(crystal_id="Si",hkl=[1,1,1],photon_energy_in_keV=12.398,verbose=True):
    #
    # get crystal data for silicon crystal
    #
    cryst = xraylib.Crystal_GetCrystal(crystal_id)

    # print some info
    if verbose:
        print ("  Unit cell dimensions [A] are %f %f %f" % (cryst['a'],cryst['b'],cryst['c']))
        print ("  Unit cell angles are %f %f %f" % (cryst['alpha'],cryst['beta'],cryst['gamma']))
        print ("  Unit cell volume [A] is %f" % (cryst['volume']) )

    #
    # define miller indices and compute dSpacing
    #


    hh = hkl[0]
    kk = hkl[1]
    ll = hkl[2]
    debyeWaller = 1.0
    rel_angle = 1.0  # ratio of (incident angle)/(bragg angle) -> we work at Bragg angle

    dspacing = xraylib.Crystal_dSpacing(cryst,hh,kk,ll )
    if verbose: print("dspacing: %f A \n"%dspacing)
    #
    # define energy and get Bragg angle
    #
    ener = photon_energy_in_keV # 12.398 # keV
    braggAngle = xraylib.Bragg_angle(cryst,ener,hh,kk,ll )
    if verbose: print("Bragg angle: %f degrees \n"%(braggAngle*180/np.pi))


    #
    # get the structure factor (at a given energy)
    #
    f0 = xraylib.Crystal_F_H_StructureFactor(cryst, ener, 0, 0, 0, debyeWaller, 1.0)
    fH = xraylib.Crystal_F_H_StructureFactor(cryst, ener, hh, kk, ll, debyeWaller, 1.0)
    if verbose: print("f0: (%f , %f) \n"%(f0.real,f0.imag))
    if verbose: print("fH: (%f , %f) \n"%(fH.real,fH.imag))

    #
    # convert structure factor in chi (or psi) = - classical_e_radius wavelength^2 fH /(pi volume)
    #
    codata = scipy.constants.codata.physical_constants
    codata_c, tmp1, tmp2 = codata["speed of light in vacuum"]
    codata_h, tmp1, tmp2 = codata["Planck constant"]
    codata_ec, tmp1, tmp2 = codata["elementary charge"]
    codata_r, tmp1, tmp2 = codata["classical electron radius"]

    ev2meter = codata_h*codata_c/codata_ec
    wavelength = ev2meter/(ener*1e3)
    if verbose: print("Photon energy: %f keV \n"%ener)
    if verbose: print("Photon wavelength: %f A \n"%(1e10*wavelength))

    volume = cryst['volume'] *1e-10*1e-10*1e-10 # volume of silicon unit cell in m^3
    cte = - codata_r * wavelength*wavelength/(np.pi * volume)

    chi0 = cte*f0
    chiH = cte*fH

    if verbose: print("chi0: (%e , %e) \n"%(chi0.real,chi0.imag))
    if verbose: print("chiH: (%e , %e) \n"%(chiH.real,chiH.imag))

    return braggAngle, np.conjugate(chi0), np.conjugate(chiH)


if __name__ == "__main__":

    # theta, chi0, chiH = get_crystal_data("Si",hkl=[1,1,1],photon_energy_in_keV=12.398,verbose=True)

    theta, chi0, chiH = get_crystal_data("Si", hkl=[1, 1, 1], photon_energy_in_keV=17.0, verbose=True)
