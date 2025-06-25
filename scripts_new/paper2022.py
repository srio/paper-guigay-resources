#(* 03/05/19 Lauesym17; Si111 17kev, distances in mm, p source distance, q observation distance *)
import numpy
from scipy.special import jv as BesselJ
import scipy.constants as codata
from crystal_data import get_crystal_data
import scipy.constants as codata
from srxraylib.plot.gol import plot, set_qt, plot_show

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
def psizero(x, q):
    v = numpy.linspace(-asym, asym, 2000)
    y = numpy.zeros_like(v, dtype=complex)
    for i in range(v.size):
        y[i] = BesselJ(0, sym * Sqrt((asym + v[i]) * (asym - v[i]))) * Exp(I * k * 0.5 * (v[i] - x) ** 2 / q)
    return numpy.trapz(y,x=v)

# psisym[x_,q_]:=2*NIntegrate[BesselJ[0,sym*Sqrt[(asym+v)*(asym-v)]]*Exp[I*k*0.5*v^2/lesym[q]]*Cos[k*v*x*p/(pe*(p+q))],{v,0,asym}];
def psisym(x, q):
    v = numpy.linspace(0, asym, 1000)
    y = numpy.zeros_like(v, dtype=complex)
    for i in range(v.size):
        y[i] = BesselJ(0, sym * Sqrt((asym + v[i]) * (asym - v[i]))) * \
               Exp(I * k * 0.5 * v[i]**2 / lesym(q) ) * Cos(k * v[i] * x * p / (pe*(p+q)))
    return 2 * numpy.trapz(y,x=v)

def psisym_NEW(x, q):
    v = numpy.linspace(-asym, asym, 2000)
    y = numpy.zeros_like(v, dtype=complex)
    qe = q * raygam / (raygam - q)
    for i in range(v.size):
        y[i] = BesselJ(0, sym * Sqrt((asym + v[i]) * (asym - v[i]))) * \
               Exp(I * k * 0.5 * v[i]**2 / lesym(q) ) * \
               Exp(-I * k * 0.5 * v[i] * ( 2 * x * qe / q / lesym(q) + asym / raygam) )
    return numpy.trapz(y, x=v)


if __name__ == "__main__":
    #
    # TODO: note that there are variables defined in main() and passed implicitely to the functions. Not nice!
    #

    #
    # inputs (in mm) ===========================================================
    #
    photon_energy_in_keV = 17.0
    # photon_energy_in_keV = 8.3
    t = 0.250
    p=50000
    rayon=1000

    crystal_id="Si"
    hkl=[1,1,1]

    #
    # end inputs ===========================================================
    #

    teta, chizero, chih = get_crystal_data(crystal_id, hkl=hkl, photon_energy_in_keV=photon_energy_in_keV, verbose=False)
    lambda1 = codata.h * codata.c / codata.e / (photon_energy_in_keV * 1e3) * 1e3 # in mm
    print("photon_energy_in_keV:", photon_energy_in_keV)
    print(">>>>>>>>>> after teta_deg:", teta*180/numpy.pi)
    print(">>>>>>>>>> after chizero:", chizero)
    print(">>>>>>>>>> chih:", chih)
    print(">>>>>>>>>> lambda1:", lambda1)



    k=2*Pi/lambda1
    h=2*k*Sin(teta)
    chihm=-I*chih
    chih2=chih*chihm

    raygam=rayon*Cos(teta)
    print("raygam",raygam)

    sym=k*Sqrt(chih2)/Sin(2*teta)
    attsym=1.0 # Exp(-k*Im(chizero)*t/Cos(teta))
    print("attsym",attsym)
    asym=t*Sin(teta)
    print("asym,halfwith of reflected beam [**a in mm**]",asym)
    zasym=sym*asym
    print("zasym",zasym)
    qzero=asym*Sin(2*teta)/Re(Sqrt(chih2))
    print("qzero, dynamical focal length [**q0 in mm]",qzero)
    qpoly=p*raygam/(2*p+raygam)
    print("qpoly, polychromatic focal distance",qpoly)
    pe=p*raygam/(raygam+p)
    print("pe",pe)


    #
    # this part is used to create Fig. 4. Run it 2 times, or 8 and 17 keV
    #
    if False:
        qq = numpy.concatenate((numpy.linspace(-5000, -200, 1000), numpy.linspace(200, 5000, 1000)))

        yy = numpy.zeros_like(qq)
        for j in range(qq.size):
            yy[j] = attsym * Abs(psizero(0, qq[j])**2 / (lambda1 * qq[j]))

        plot(qq,yy, show=0)

        if True:
            xi = numpy.linspace(-asym, asym, 1000)
            yy1 = numpy.zeros_like(xi)

            i = numpy.argmax(yy)
            qq1 = qq[i]
            print("Lateral profile at q=%g" % (qq1))
            for j in range(xi.size):
                yy1[j] = attsym * Abs(psizero(xi[j], qq1) ** 2 / (lambda1 * qq1))
            plot(xi, yy1, show=0)


    # This is for Fig 5
    if True:
        if True:
            qq = numpy.linspace(0, 3000, 1000)
            yy = numpy.zeros_like(qq)
            for j in range(qq.size):
                yy[j] = attsym * Abs(psisym(0, qq[j]) ** 2 / (lambda1 * (p + qq[j])))
            plot(qq, yy, show=0)
            i = numpy.argmax(yy)
            qq1 = qq[i]
        else:
            qq1 = 651.652 # 624.625 for 17 keV

        if True:
            xi = numpy.linspace(-asym, asym, 1000)
            yy1 = numpy.zeros_like(xi)
            yy2 = numpy.zeros_like(xi)
            title = "Lateral profile at q=%g" % (qq1)
            print(title)
            for j in range(xi.size):
                yy1[j] = attsym * Abs(psisym(xi[j], qq1) ** 2 / (lambda1 * (p + qq1)))
                yy2[j] = attsym * Abs(psisym_NEW(xi[j], qq1) ** 2 / (lambda1 * (p + qq1)))
            plot(xi, yy1, xi, yy2, legend=['Cos','Exp'], xtitle='x', title=title, show=0)

    plot_show()
