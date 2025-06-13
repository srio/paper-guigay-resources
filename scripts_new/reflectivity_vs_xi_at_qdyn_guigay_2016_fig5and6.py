import numpy
from scipy.special import jv as BesselJ
import scipy.constants as codata
from crystal_data import get_crystal_data
from srxraylib.plot.gol import plot, set_qt, plot_show
import mpmath

# sgplus[x_, q_] :=
#   2*NIntegrate[
#     Hypergeometric1F1[I*kap, 1, I*acmax*(1 - (v/a)^2)]*
#      Exp[0.5*I*k*v^2*invle[q]]*Cos[k*v*(x/q - I*kiny)], {v, 0, a}];

def sgplus(x, q):
    v = numpy.linspace(0, a, 1000)
    y = numpy.zeros_like(v, dtype=complex)
    for i in range(v.size):
        y[i] = mpmath.hyp1f1(I * kap, 1, I * acmax * (1 - (v[i] / a)**2)) * \
            Cos(k * v[i] * x / (q * pe * be(q))) * \
            Exp(I * k * 0.5 * v[i]**2 * invle(q) - k * v[i] * kiny) #* \
        # Exp(I * k * 0.5 * v[i] ** 2 * invle(q) - k * v * kiny)  # * \
    return numpy.trapz(y, x=v)  # * 2 ??

# curved symmetric Laue, source at finite p
# Eq. 13 in doi:10.1107/S0108767312044601  or
# Eq. 23 in https://doi.org/10.1107/S1600577521012480
def integral_psisym(x, q, Z=0.0, k=0.0, a=0.0, RcosTheta=0.0, p=0.0, on_x_at_maximum=0):
    v = numpy.linspace(0, a, 1000)
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

    return 2 * numpy.trapz(y, x=v)


def get_max(xx, yy, verbose=1):
    i = numpy.argmax(yy)
    if verbose: print("Maximum found at x=%g, y=%g" % (xx[i],yy[i]))
    return xx[i], yy[i], i



if __name__ == "__main__":
    import mpmath




    # run_symmetric()

    #
    # asymmetric
    #

    Pi = numpy.pi
    Sin = numpy.sin
    Cos = numpy.cos
    Exp = numpy.exp
    I = 1j
    Conjugate = numpy.conjugate
    Im = numpy.imag
    Re = numpy.real
    Abs = numpy.abs

    teta = 5.67318*Pi/180
    alfa = 2.0*Pi/180
    lambda1 = 0.619927e-7
    k = 2*Pi/lambda1
    h = 2*k*Sin(teta)
    chizero = -0.242370e-5 + I*0.918640e-8
    chih = -0.922187e-6 + I*0.913110e-6
    chimh = I*chih
    chih2 = chih*chimh
    u2 = 0.25*chih2*k**2
    t = 0.250
    p = 29000.0
    rayon = 2000
    raygam = rayon*Cos(teta)
    kp = k*Sin(2*teta)
    kp2 = k*Sin(2*teta)**2
    rau = 0.2201
    pha = 0.67



    teta1 = alfa + teta
    teta2 = alfa - teta
    fam1 = Sin(teta1)
    fam2 = Sin(teta2)
    gam1 = Cos(teta1)
    gam2 = Cos(teta2)
    t1 = t/gam1
    t2 = t/gam2
    qpoly = p*rayon*gam2/(2*p + rayon*gam1)
    att = Exp(-k*0.5*(t1 + t2)*Im(chizero))
    s2max = 0.25*t1*t2; u2max = u2*s2max
    gama = t2/t1
    a = Sin(2*teta) * t1 * 0.5
    kin = 0.25*(t1 - t2)*chizero/a
    kinx = Re(kin)
    kiny = Im(kin)
    com = Sin(alfa) * (1 + gam1 * gam2 * (1 + rau))
    kp3 = 0.5 * k * (gama*a)**2
    mu1 = (Cos(alfa) * 2 * fam1 * gam1 + Sin(alfa) * (fam1**2 + rau * gam1**2)) / (Sin(2*teta)* Cos(teta))
    mu2 = -(Cos(alfa) * 2 * fam2 * gam2 + Sin(alfa) * (fam2**2 + rau * gam2**2)) / (Sin(2*teta)* Cos(teta))
    a1 = (0.5 * t / Cos(teta)) * (Cos(alfa) * Sin(teta1) + rau*Sin(alfa) * Cos(teta1))
    a2 = -(0.5 * t / Cos(teta)) * (Cos(alfa) * Sin(teta2) + rau*Sin(alfa) * Cos(teta2))
    acrist = -h * Sin(alfa) * (1 + gam1 * gam2 * (1 + rau)) / rayon
    acmax = acrist * s2max
    g = gama * acrist * rayon / kp2
    kap = u2max / acmax
    pe = p * rayon / (gama**2 * (rayon + p * mu2) - g * p)


    # qe[q_] := q*rayon/(rayon - q*(mu1 + g));
    qe = lambda q:  q * rayon / (rayon - q * mu1 - g * q)

    # be[q_] := 1/qe[q] + 1/pe;
    be = lambda q: 1 / qe(q) + 1 / pe


    # le[q_] := (pe + qe[q])/(1 + g*(pe*qe[q]));
    # le = lambda q: (pe + qe(q)) / (1 + g * (pe * qe(q)))

    # invle[q_] := 1/q - mu1/rayon;
    invle = lambda q: 1 / (p + qe(q)) + g / rayon

    # q-scan
    if 1:
        qq = numpy.linspace(100, 10000, 100)
        yy = numpy.zeros_like(qq)
        for j in range(qq.size):
            yy[j] = Abs(sgplus(0, qq[j]) ** 2 * att / (lambda1 * qq[j] * p * be(qq[j])))
            print(j, qq[j], yy[j])
        plot(qq, yy)
        qdyn, _, imax = get_max(qq, yy)

    # x-scan
    if 0:
        xx = numpy.linspace(-2.5e-3, 2.5e-3, 200)
        yy = numpy.zeros_like(xx)
        qposition = 1679.0
        for j in range(xx.size):
            yy[j] = Abs(sgplus(xx[j], qposition) ** 2 * att / (lambda1 * qposition))
            print(j, xx[j], yy[j])
        plot(xx, yy)


# intplus =
#  Plot[Abs[sgplus[0, q]^2*att/(lambda*q)], {q, 100, 5000},
#   AxesOrigin -> {2000, 0}, PlotRange -> All,
#   PlotStyle -> {RGBColor[0, 1, 0]}]
# psplus = Table[Abs[sgplus[0, q]^2*att/(lambda*q)], {q, 1600, 1800, 1}]
# focplus =
#  Plot[Abs[sgplus[xprim*0.001, 1679]^2*
#     att/(lambda*1679)], {xprim, -2.5, 2.5}, AxesOrigin -> {0, 0},
#   PlotRange -> All, PlotStyle -> {RGBColor[0, 1, 0]}]









