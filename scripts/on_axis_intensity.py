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


if __name__ == "__main__":
    #
    # TODO: note that there are variables defined in main() and passed implicitely to the functions. Not nice!
    #




    #
    # inputs (in mm) ===========================================================
    #
    photon_energy_in_keV = 17.0
    photon_energy_in_keV = 8.3
    t = 0.250
    p=50000
    rayon=1000

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
    print(">>>>>>>>>> after chizero:", chizero)
    print(">>>>>>>>>> chih:", chih)
    print(">>>>>>>>>> lambda1:", lambda1)



    k=2*Pi/lambda1
    h=2*k*Sin(teta)
    chihm=-I*chih
    chih2=chih*chihm

    # {raygam,993.214}
    # {attsym,0.630104}
    # {asym,halfwith of reflected beam,0.0348896}
    # {zasym,23.4177 -0.161369 I}
    # {qzero, dynamical focal length,4478.23}
    # {qpoly, polychromatic focal distance,491.723}
    # {pe,973.869}
    #
    raygam=rayon*Cos(teta)
    print("raygam",raygam)

    sym=k*Sqrt(chih2)/Sin(2*teta)
    attsym=Exp(-k*Im(chizero)*t/Cos(teta))
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


    # print(">>>>>",[psizero(0,1000)]) # -0.00137474 - 0.00105539 I

    #
    # this part is used to create Fig. 4. Run it 2 times, or 8 and 17 keV
    #
    if False:
        # Plot[attsym*Abs[psizero[0,q]^2/(lambda*q)],{q,500,5000},AxesOrigin->{0,0},PlotRange->All,PlotStyle->{RGBColor[1,0,0]}]
        # Plot[attsym*Abs[psizero[0,q]^2/(lambda*q)],{q,-5000,-500},AxesOrigin->{0,0},PlotRange->All,PlotStyle->{RGBColor[1,0,0]}]
        for tmp in ["negative","positive"]:
            if tmp == "negative":
                qq = numpy.linspace(-5000, -200, 1000)
            elif tmp == "positive":
                qq = numpy.linspace(200, 5000, 1000)

            filename = 'mypoints1_%dkeV_%s.txt' % (photon_energy_in_keV, tmp)
            f = open(filename,'w')
            yy = numpy.zeros_like(qq)
            for j in range(qq.size):
                yy[j] = attsym*Abs(psizero(0,qq[j])**2/(lambda1*qq[j]))
                f.write("%g %g\n" % (qq[j], yy[j]))
            f.close()
            print("File written to disk: %s" % filename)
            print_max(qq, yy)
            print_max(qq[0:qq.size//2], yy[0:qq.size//2])
            plot(qq,yy)


    # this is for p at infinity...  NOT USED....
    if False:
        # intpinf=Plot[attsym*Abs[psipinfini[0,q]^2/(lambda*raygam)],{q,0,3000},AxesOrigin->{0,0},PlotRange->All,PlotStyle->{RGBColor[1,0,0]}]

        qq = numpy.linspace(0,3000, 1000)

        filename = 'mypoints2_%dkeV.txt' % photon_energy_in_keV
        f = open(filename,'w')
        yy = numpy.zeros_like(qq)
        for j in range(qq.size):
            yy[j] = attsym*Abs(psipinfini(0,qq[j])**2/(lambda1*raygam))
            f.write("%g %g\n" % (qq[j], yy[j]))
        f.close()
        print("File written to disk: %s" % filename)
        plot(qq,yy)

    # This is for Fig 5. q-scan...
    if True:
        # intpfini=Plot[attsym*Abs[psisym[0,q]^2/(lambda*(p+q))],{q,0,3000},AxesOrigin->{0,0},PlotRange->All,PlotStyle->{RGBColor[0,1,0]}]
        qq = numpy.linspace(0, 3000, 1000)
        filename = 'mypoints3_%dkeV.txt' % photon_energy_in_keV
        f = open(filename,'w')
        yy = numpy.zeros_like(qq)
        for j in range(qq.size):
            yy[j] = attsym * Abs(psisym(0, qq[j]) ** 2 / (lambda1 * (p + qq[j])))
            f.write("%g %g\n" % (qq[j], yy[j]))
        f.close()
        print("File written to disk: %s" % filename)
        print_max(qq, yy)
        nn = qq.size
        print_max(qq[nn // 3:-1], yy[nn // 3:-1])
        plot(qq, yy)

        #
        #  compare position of best focus with eqs XXX from the paper
        #
        print(">>>>>R, q0, p, ", rayon, qzero, p)
        q1 = rayon * (p * (qzero - rayon) + qzero * rayon) / (p * qzero + (rayon + qzero) * rayon)
        q2 = rayon * (p * (qzero + rayon) + qzero * rayon) / (p * qzero + (qzero - rayon) * rayon)
        print(">>>>> q1, q2: ", q1, q2)

        # these are the values from the run with the flat crystal
        if photon_energy_in_keV > 10:
            qzero_numerical = 2559.0
        else:
            qzero_numerical = 2862.0
        print(">>>>> Energy, R, q0 NUMERICAL, p, ", photon_energy_in_keV, rayon, qzero_numerical, p)
        q1 = rayon * (p * (qzero_numerical - rayon) + qzero_numerical * rayon) / (p * qzero_numerical + (rayon + qzero_numerical) * rayon)
        q2 = rayon * (p * (qzero_numerical + rayon) + qzero_numerical * rayon) / (p * qzero_numerical + (qzero_numerical - rayon) * rayon)
        print(">>>>> Calculated q1, q2 using Eqs XXX: ", q1, q2)


        #
        # calculate xi_c
        #
        pe = 1 / (1 / p + 1 / (rayon * numpy.cos(teta)))
        qe = 1 / (1 / qq - 1 / (rayon * numpy.cos(teta)))
        Le = pe + qe
        xi_c = - asym * qq * Le / (2 * qe * rayon * numpy.cos(teta))

        plot(qq,xi_c/asym,title=">>>>>>>>>xi_e")
        print(">>>>",teta, 180*teta/numpy.pi, qq[1], xi_c[1])


    if False:
        # intpfini=Plot[attsym*Abs[psisym[0,q]^2/(lambda*(p+q))],{q,0,3000},AxesOrigin->{0,0},PlotRange->All,PlotStyle->{RGBColor[0,1,0]}]
        # qq = numpy.linspace(0, 3000, 1000)
        xi = numpy.linspace(-0.02,0.02, 1000)
        # filename = 'mypoints44_%dkeV.txt' % photon_energy_in_keV
        filename = 'mypoints4_%dkeV.txt' % photon_energy_in_keV
        f = open(filename,'w')
        yy1 = numpy.zeros_like(xi)
        yy2 = numpy.zeros_like(xi)
        if photon_energy_in_keV < 10:
            qq1 = 651
            qq2 = 1330
        else:
            qq1 = 625
            qq2 = 1372
        for j in range(xi.size):
            yy1[j] = attsym * Abs(psisym(xi[j], qq1) ** 2 / (lambda1 * (p + qq1)))
            yy2[j] = attsym * Abs(psisym(xi[j], qq2) ** 2 / (lambda1 * (p + qq2)))
            f.write("%g %g %g\n" % (xi[j], yy1[j], yy2[j]))
        f.close()
        print("File written to disk: %s" % filename)
        plot(xi, yy1, xi, yy2)

