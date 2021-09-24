#(* 03/05/19 Lauesym17; Si111 17kev, distances in mm, p source distance, q observation distance *)
import numpy
from scipy.special import jv as BesselJ
import scipy.constants as codata
from crystal_data import get_crystal_data
import scipy.constants as codata
from srxraylib.plot.gol import plot, set_qt

set_qt()



Pi = numpy.pi
I = 1j
Sin = numpy.sin
Cos = numpy.cos
Conjugate = numpy.conjugate
Sqrt = numpy.sqrt
Exp = numpy.exp
Im = numpy.imag
Re = numpy.real
Abs = numpy.abs


# lesym[q_]:=pe+raygam*q/(raygam-q);{"pe",pe}
lesym = lambda q: pe+raygam*q/(raygam-q)

#
#
#
# psizero[x_,q_]:=2*NIntegrate[BesselJ[0,sym*Sqrt[(asym+v)*(asym-v)]]*Exp[I*k*0.5*(v-x)^2/q],{v,0,asym}];
def psizero(x,q):
    v = numpy.linspace(0, asym, 1000)
    y = numpy.zeros_like(v,dtype=complex)
    for i in range(v.size):
        y[i] = BesselJ(0, sym * Sqrt((asym + v[i]) * (asym - v[i]))) * Exp(I * k * 0.5 * (v[i] - x) ** 2 / q)
    return 2 * numpy.trapz(y,x=v)

#
#
#
# psipinfini[x_,q_]:=2*NIntegrate[BesselJ[0,sym*Sqrt[(asym+v)*(asym-v)]]*Exp[I*k*0.5*v^2*(raygam-q)/raygam^2]*Cos[k*v*x/raygam],{v,0,asym}];
def psipinfini(x,q):
    v = numpy.linspace(0, asym, 1000)
    y = numpy.zeros_like(v,dtype=complex)
    for i in range(v.size):
        y[i] = BesselJ(0, sym * Sqrt((asym + v[i]) * (asym - v[i]))) * \
               Exp(I * k * 0.5 * v[i]**2 * (raygam - q) / raygam**2) * Cos(k * v[i] * x / raygam)
    return 2 * numpy.trapz(y,x=v)

# psisym[x_,q_]:=2*NIntegrate[BesselJ[0,sym*Sqrt[(asym+v)*(asym-v)]]*Exp[I*k*0.5*v^2/lesym[q]]*Cos[k*v*x*p/(pe*(p+q))],{v,0,asym}];
def psisym(x,q):
    v = numpy.linspace(0, asym, 1000)
    y = numpy.zeros_like(v,dtype=complex)
    for i in range(v.size):
        y[i] = BesselJ(0, sym * Sqrt((asym + v[i]) * (asym - v[i]))) * \
               Exp(I * k * 0.5 * v[i]**2 / lesym(q) ) * Cos(k * v[i] * x * p / (pe*(p+q)))
    return 2 * numpy.trapz(y,x=v)

def print_max(xx,yy):
    i = numpy.argmax(yy)
    print("Maxumum found at x=%g, y=%g" % (xx[i],yy[i]))
    return xx[i]


if __name__ == "__main__":
    #
    # TODO: note that there are variables defined in main() and passed implicitely to the functions. Not nice!
    #




    #
    # inputs (in mm) ===========================================================
    #
    photon_energy_in_keV = 17.0
    # photon_energy_in_keV = 8.3


    crystal_id="Si"
    hkl=[1,1,1]

    #
    # end inputs ===========================================================
    #

    # lambda1= 0.7293259e-7
    # print(">>> Photon energy: %g eV" % (codata.h * codata.c / codata.e / (lambda1 * 1e-3)))
    # chizero=Conjugate(-0.335984e-5-I*0.177494e-7)
    # chih=Conjugate((-0.128146+I*0.126392)*1e-5)
    # teta=6.678539*(Pi/180)
    # print(">>>>>>>>>> before teta, chizero, chih, lambda1:", teta, chizero, chih, lambda1)

    teta, chizero, chih = get_crystal_data(crystal_id, hkl=hkl, photon_energy_in_keV=photon_energy_in_keV, verbose=False)
    lambda1 = codata.h * codata.c / codata.e / (photon_energy_in_keV * 1e3) * 1e3 # in mm
    print("photon_energy_in_keV:", photon_energy_in_keV)
    print(">>>>>>>>>> after teta_deg:", teta*180/numpy.pi)
    print(">>>>>>>>>> after chizero *1e6:", chizero*1e6)

    chihm=-I*chih
    chih2=chih*chihm

    print(">>>>>>>>>> chih:", chih)
    print(">>>>>>>>>> chihm:", chihm)
    print(">>>>>>>>>> chih2 * 1e12:", chih2*1e12)
