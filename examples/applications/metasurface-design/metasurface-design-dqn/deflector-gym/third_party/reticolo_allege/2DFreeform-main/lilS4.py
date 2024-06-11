import numpy as np
import S4
import matplotlib.pyplot as plt
import time
import argparse

##########################

parser = argparse.ArgumentParser()

parser.add_argument('--nG', default=100, help="diffraction order", type=int)
parser.add_argument('--wl', default=1250, help = "wavelength", type=float)
parser.add_argument('--ang', default=55, help ="angle", type=float)
parser.add_argument('--ncells', default=256, help ="number of cells", type=int)

args = parser.parse_args()

#variables setting
nG = args.nG
wl = args.wl
ang = args.ang
ncells = args.ncells
period = abs(wl/np.sin(ang/180*np.pi))
freq = 1/wl
S = S4.New(Lattice=((period,0),(0,period/2)), NumBasis=nG)


# Permittivities & thickness[um]
eps_SiO2 = 1.4504**2
eps_Si = 3.5750**2   #shd be altered
grating_thickness = 0.325

#import & save the structure
gratingMatrix = np.load('struct.npy')
np.savetxt('gratingMatrix.csv',gratingMatrix,delimiter=",")

ynum = np.shape(gratingMatrix)[0]
xnum = np.shape(gratingMatrix)[1]
print('x: ', xnum, ' y: ', ynum)


start = time.time()

S.SetFrequency(freq)
S.SetMaterial(Name = 'SiO2', Epsilon = eps_SiO2)
S.SetMaterial(Name = 'Vacuum', Epsilon = 1)
S.SetMaterial(Name = 'Si', Epsilon = eps_Si)

S.AddLayer(Name = 'top', Thickness = 0, Material= 'Si')
S.AddLayer(Name = 'grating', Thickness = grating_thickness, Material = 'Vacuum')
S.AddLayer(Name = 'bottom', Thickness = 0, Material = 'Vacuum')

S.SetExcitationPlanewave(
        IncidenceAngles = (0,0),
        sAmplitude = 0,
        pAmplitude = 1
        )

S.SetOptions( # these are the defaults
    Verbosity = 1,
    LatticeTruncation = 'Circular',
    DiscretizedEpsilon = False,
    DiscretizationResolution = 8,
    PolarizationDecomposition = True,
    PolarizationBasis = 'Normal',
    LanczosSmoothing = False,
    SubpixelSmoothing = False,
    ConserveMemory = False
    )

for i1 in range(xnum):
    for i2 in range(ynum):
        if gratingMatrix[i2][i1]:
            S.SetRegionRectangle(
                        Layer='grating',
                        Material = 'SiO2',
                        #Center = ((i1-(howMany-1)/2)*period/howMany,(i2-(howMany-1)/2)*period/howMany),
                        Center = (-period/2+period/(2*xnum) + i1*(period/xnum), -period/4+period/(4*ynum) + i2*(period/ynum)),
                        Angle = 0,
                        Halfwidths = (period/(2*xnum), period/(4*ynum))
                        )
    

P = np.asarray(S.GetPowerFluxByOrder(Layer = 'bottom', zOffset = 0))
efficiency = np.real(P[1,0])*100
end = time.time()

timecost = end - start

print('time1')
print(timecost,  's')
print('eff1')
print(efficiency,  '%')




Glist = S.GetBasisSet()
(fo, bo)= S.GetPoyntingFlux(Layer = 'top')

Pi = S.GetPowerFluxByOrder(Layer = 'grating')
Po = S.GetPowerFluxByOrder(Layer = 'bottom')
diff_air = abs(Po[0][0])
diff_glass = abs(Pi[1][1])
effs = diff_glass/diff_air

end2 = time.time()
print('time2')
print(end2-end, 's')
print('eff2')
print(effs, '%')

        
