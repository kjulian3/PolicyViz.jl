#Regression Constants
const RANGEMAX = 62000.0 #ft
const ranges = [499, 800, 2000, 3038, 5316, 6450, 7200, 7950, 8725, 10633, 13671, 16709, 19747, 
                22785, 25823, 28862, 31900, 34938, 37976, 41014, 48608, 60760]

const thetas = linspace(-pi,pi,41)
const psis   = linspace(-pi,pi,41)
const sos    = [100, 200, 300, 400, 500, 600, 700, 800]
const sis    = [0, 100, 200, 300, 400, 500, 600, 700, 800]
const taus   = [0, 1, 5, 10, 20, 40, 60, 80, 100]
const pas    = [1, 2, 3, 4, 5]
const pasTrue = [0, 1.5, -1.5, 3.0, -3.0]
const NSTATES = length(ranges)*length(thetas)*length(psis)*length(sos)*length(sis)*length(taus)*length(pas)
const ACTIONS = deg2rad([0.0 1.5 -1.5 3.0 -3.0])



#Deep RL Constants
const RangeMax = 3000.0 # meters
const RANGEMIN = 0.0 # meters
const PsiDim   = 41
const PsiMin   = -pi #[rad]
const PsiMax   = pi  #[rad]

const psis_drl      = linspace(PsiMin,PsiMax,PsiDim)
const sos_drl       = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
const sis_drl       = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

const Actions= [-20.0,-10.0,0.0,10.0,20.0,-6.0]

const STATE_DIM = 5
const ACTION_DIM = 6    

const Thetamin = -pi #[rad]
const Thetamax = pi #[rad]
const Bearingmin = -pi  # [rad]
const Bearingmax = pi  # [rad]
const Speedmin = 10.0  # [m/s]
const Speedmax = 20.0  # [m/s]

const Rangedim = 67
const Thetadim = 41
const Bearingdim = 37 #37 #3
const Speeddim = 5 #5 #2
const Ranges = [0., 10., 20., 30., 40., 50., 60., 80., 100., 120., 
                140., 160., 180., 200., 230., 250., 270., 290., 310., 330.,
                350., 370., 390., 410., 430., 450., 470., 490., 510., 530.,
                570., 600., 630., 660., 690., 720., 750., 780., 810., 840.,
                870., 900., 930., 960., 1000., 1040., 1080., 1120., 1160., 1200.,
                1240., 1280., 1320., 1360., 1400., 1450., 1500., 1550., 1600., 1650.,
                1700., 1750., 1800., 1850., 1900., 1950., 2000.]
const Thetas = collect(linspace(Thetamin, Thetamax,Thetadim))
const Bearings = collect(linspace(Bearingmin, Bearingmax, Bearingdim))
const Speeds = collect(linspace(Speedmin, Speedmax, Speeddim))
const NSTATES_drl = Rangedim*Thetadim*Bearingdim*Speeddim^2+1

#Compatible with X and Y grids rather than range and theta
const Xdim = 71 #51 #8  
const Ydim = 71 #51 #8
const Xs = collect(linspace(-RangeMax,RangeMax,Xdim))
const Ys = collect(linspace(-RangeMax,RangeMax,Ydim))
const NSTATES_drl_xandy = Xdim*Ydim*Bearingdim*Speeddim^2+1