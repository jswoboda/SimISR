#!/usr/bin/env python
"""
Created on Tue Dec 29 15:49:01 2015

@author: John Swoboda
"""

import tables
from const.physConstants import v_C_0
from utilFunctions import readconfigfile
import scipy.fftpack as scfft
from IonoContainer import IonoContainer,makeionocombined
from RadarDataSim.utilFunctions import spect2acf
import scipy as sp
import pdb
class RadarSpaceTimeOperator(object):
    """ This is a class to hold the operator methods for the ISR simulator.
    
        Variables:
           Cart_Cords_In - Ntbegx3 array of cartisian coordinates for the input space.
           Sphere_Cords_In - Ntbegx3 array of spherical coordinates for the input space.
           Cart_Cords_Out - Ntoutx3 array of cartisian coordinates for the output space.
           Sphere_Cords_Out - Ntoutx3 array of spherical coordinates for the output space.
           Time_In - A Ntbegx2 numpy array with the start and stop times of the input data.
           Time_Out - A Ntoutx2 numpy array with the start and stop times of the output data.
           RSTMat - A list of matricies or a single matrix that is the forward between physical space
               to the discrete samples space of the radar.
           blocks - A tuple that holds the number of block matricies in overall forward operator.
           blocksize - A tuple that holds the shape of the outmatrix size.
           blockloc - An Ntout x Ntbeg array that holds the corresponding spatial forward model.
    """
    def __init__(self,ionoin=None,configfile=None,timein=None,RSTOPinv=None,invmat=None):
        """ This will create the RadarSpaceTimeOperator object.
            Inputs
                ionoin - The input ionocontainer. This can be either an string that is a ionocontainer file,
                    a list of ionocontainer objects or a list a strings to ionocontainer files
                config  - The ini file that used to set up the simulation.
                timein - A Ntx2 numpy array of times.
                RSTOPinv - The inverse operator object.
                invmat - The inverse matrix to the original operator.
        """
            
        d2r = sp.pi/180.0

        # First determine if this will be an inverse operator. If not build up the object.
        if RSTOPinv is None:

            (sensdict,simparams) = readconfigfile(configfile)
            # determine if the input ionocontainer is a string, a list of strings or a list of ionocontainers.
            if isinstance(ionoin,basestring):
                ionoin = IonoContainer.readh5(ionoin)
            elif isinstance(ionoin,list):
                if isinstance(ionoin[0],basestring):
                    ionoin = IonoContainer.readh5(ionoin[0])
                else:
                    ionoin=ionoin[0]
            #Input location
            self.Cart_Coords_In = ionoin.Cart_Coords
            self.Sphere_Coords_In = ionoin.Sphere_Coords

            # Set the input times
            if timein is None:
                self.Time_In = ionoin.Time_Vector
            else:
                self.Time_In = timein

            #Create an array of output location based off of the inputs
            rng_vec2 = simparams['Rangegatesfinal']
            nrgout = len(rng_vec2)

            angles = simparams['angles']
            nang =len(angles)

            ang_data = sp.array([[iout[0],iout[1]] for iout in angles])
            rng_all = sp.repeat(rng_vec2,(nang),axis=0)
            ang_all = sp.tile(ang_data,(nrgout))

            self.Sphere_Coords_Out = sp.column_stack((rng_all,ang_all))
            (R_vec,Az_vec,El_vec) = (self.Sphere_Coords_Out[:,0],self.Sphere_Coords_Out[:,1],
                self.Sphere_Coords_Out[:,2])
            xvecmult = sp.cos(Az_vec*d2r)*sp.cos(El_vec*d2r)
            yvecmult = sp.sin(Az_vec*d2r)*sp.cos(El_vec*d2r)
            zvecmult = sp.sin(El_vec*d2r)
            X_vec = R_vec*xvecmult
            Y_vec = R_vec*yvecmult
            Z_vec = R_vec*zvecmult
            
            self.Cart_Coords_Out = sp.column_stack((X_vec,Y_vec,Z_vec))
            self.Time_Out = sp.column_stack((simparams['Timevec'],simparams['Timevec']+simparams['Tint']))+self.Time_In[0,0]
            self.simparams=simparams
            self.sensdict=sensdict
            self.lagmat = self.simparams['amb_dict']['WttMatrix']
            # create the matrix
            (self.RSTMat,self.overlaps,self.blocklocs) = makematPA(ionoin.Sphere_Coords,ionoin.Cart_Coords,ionoin.Time_Vector,configfile,ionoin.Velocity)
        elif configfile is None:

            self.Cart_Coords_Out = RSTOPinv.Cart_Coords_In
            self.Sphere_Coords_Out = RSTOPinv.Sphere_Coords_In
            self.Time_Out =  RSTOPinv.Time_In

            self.RSTMat = invmat

            self.Cart_Coords_In = self.Cart_Coords_Out
            self.Sphere_Coords_In =self.Sphere_Coords_Out
            self.Time_In = self.Time_Out
            self.lagmat = sp.diag(sp.ones(14))

    def mult_iono(self,ionoin_list):
        """ 
            This will apply the forward model to the contents of an ionocontainer object. It is assuming that 
            this is an ionocontainer holding the spectra.
        """

        ntout = self.Time_Out.shape[0]
        nlout = self.Cart_Coords_Out.shape[0]
        blist_in,blist_out = self.blocklocs
        amb_dict = self.simparams['amb_dict']
        ambmat = amb_dict['WttMatrix']
        overlaps = self.overlaps
        np = ambmat.shape[0]
        t_s = self.sensdict['t_s']
        tau_out = t_s*sp.arange(np)
        if isinstance(ionoin_list,list)or isinstance(ionoin_list,str):
            
            Iono_in = makeionocombined(ionoin_list)
        else:
            Iono_in=ionoin_list
        
        
        ionocart = Iono_in.Cart_Coords
       
        tau,acf=spect2acf(Iono_in.Param_Names,Iono_in.Param_List)
        np_in =acf.shape[-1]
        
        outdata = sp.zeros((nlout,ntout,np),dtype=acf.dtype)
        assert sp.allclose(ionocart,self.Cart_Coords_In), "Spatial Coordinates need to be the same"

        for it_out in range(ntout):
            
            overlists = overlaps[it_out]
            irows = blist_out[it_out]
            curintimes = [i[0] for i in overlists]
            cur_outmat = self.RSTMat[irows[0]:irows[1],:]
            
            for it_in in curintimes:
               icols=    blist_in[it_in]
               cur_mat = cur_outmat[:,icols[0]:icols[1]]
               tempdata=sp.zeros((np_in,nlout),dtype=acf.dtype)
               for iparam in range(np_in):
                   tempdata[iparam]=cur_mat.dot(acf[:,it_in,iparam])
               
               outdata[:,it_out] = sp.transpose(sp.dot(ambmat,tempdata)) + outdata[:,it_out]

       
        outiono = IonoContainer(self.Sphere_Coords_Out,outdata,times=self.Time_Out,sensor_loc=Iono_in.Sensor_loc,
                               ver=1,coordvecs = ['r','theta','phi'],paramnames=tau_out)
        return outiono

    
    def invertdata(self,ionoin,alpha,tik_type = 'i',type=None,dims=None,order='C',max_it=100,tol=1e-2):
        """ This will invert the matrix and create a new operator object.
        """
        ntin = self.Time_In.shape[0]
        matsttimes = self.Time_Out[:,0]
        ntout = self.Time_Out.shape[0]
        ntcounts =sp.zeros(ntin)
        nlin = self.Cart_Coords_In.shape[0]
        nlout = self.Cart_Coords_Out.shape[0]
        blocklocs = self.blocklocs

        firsttime = True
        if isinstance(ionoin,list):
            ionolist = ionoin
        else:
            ionolist = [ionoin]

        for ionon,iiono in enumerate(ionolist):
            if isinstance(iiono,str):
                curiono = IonoContainer.readh5(iiono)
            else:
                curiono=iiono


            ionodata = curiono.Param_List
            ionotime = curiono.Time_Vector
            ionocart = curiono.Cart_Coords

            assert sp.allclose(ionocart,self.Cart_Coords_Out), "Spatial Coordinates need to be the same"

            ionosttimes = ionotime[:,0]

            keeptimes = sp.arange(ntin)[sp.in1d(matsttimes,ionosttimes)]

            (nl,nt,np) = ionodata.shape
            if firsttime:
                bdata = sp.zeros((nlin,ntin,np))
                invdata=sp.zeros((nlin,ntin,np))
                firsttime==False

            b_locsind = sp.arange(len(blocklocs))[sp.in1d(blocklocs[:,0],keeptimes)]
            b_locs = blocklocs[b_locsind]


            
            for ibn,(iin,iout) in enumerate(b_locs):
                if len(self.RSTMat)==1:
                    A = self.RSTMat[0]
                else:
                    A=self.RSTMat[b_locsind[ibn]]
                ntcounts[iin]=ntcounts[iin]+1
                
                for iparam in range(np):
                    bdata[:,iin,iparam]=A.transpose().dot(ionodata[:,iout,iparam])


                C = sp.dot(A.transpose(),A)
    
                if type is None or type.lower()=='i':
                    L=sp.sparse.eye(C.shape[0])
                elif type.lower()=='d':
                    L=diffmat(dims,order)
    
                Ctik=C+sp.power(alpha,2)*L
                M=L*Ctik
                xin = sp.ones(nlin,dtype=ionodata.dtype)
                for i in range(np):
                    (invdata[:,iin,i], error, iter, flag) = cgmat(Ctik, xin, bdata[:,iin,i], M, max_it, tol)


def makematPA(Sphere_Coords,Cart_Coords,timein,configfile,vel=None):
    """Make a Ntimeout*Nbeam*Nrng x Ntime*Nloc sparse matrix for the space time operator. 
       The output space will have range repeated first, then beams then time. 
       The coordinates will be [t0,b0,r0],[t0,b0,r1],[t0,b0,r2],...
       [t0,b1,r0],[t0,b1,r1], ... [t1,b0,r0],[t1,b0,r1],...[t1,b1,r0]...
       Inputs
           Sphere_Coords - A Nlocx3 array of the spherical coordinates of the input data.
           timein - A Ntbegx2 numpy array with the start and stop times of the input data.
           configfile - The ini file used for the simulation configuration.
           vel - A NlocxNtx3 numpy array of velocity.
       Outputs
           outmat - A list of matricies or a single matrix that is the forward between physical space
               to the discrete samples space of the radar.
           blocks - A tuple that holds the number of block matricies in overall forward operator.
           blocksize - A tuple that holds the shape of the outmatrix size.
           blockloc - An Ntout x Ntbeg array that holds the corresponding spatial forward model.
    """
    #
    (sensdict,simparams) = readconfigfile(configfile)
    timeout = simparams['Timevec']
    Tint = simparams['Tint']
    timeout = sp.column_stack((timeout,timeout+Tint)) +timein[0,0]

    rng_vec = simparams['Rangegates']
    rng_bin=sensdict['t_s']*v_C_0*1e-3/2.
    sumrule = simparams['SUMRULE']
    #
    minrgbin = -sumrule[0].min()
    maxrgbin = len(rng_vec)-sumrule[1].max()
    
    angles = simparams['angles']
    Nbeams = len(angles)

    
    rng_vec2 = simparams['Rangegatesfinal']
    nrgout = len(rng_vec2)
    pulse=simparams['Pulse']
    p_samps = pulse.shape[0]
    rng_len=p_samps*rng_bin
    Nlocbeg = Cart_Coords.shape[0]
    Nlocout= Nbeams*nrgout
    Ntbeg = len(timein)
    Ntout = len(timeout)
    if vel is None:
        vel=sp.zeros((Nlocbeg,Ntbeg,3))
    # set up blocks
    blocksize = (Ntout*Nbeams*nrgout,Nlocbeg*Ntbeg)

    
    # make the matrix
    outmat = sp.sparse.lil_matrix(blocksize,dtype =sp.float64)
    # determine the overlaps
    blockloc_in = [[i*Nlocbeg,(i+1)*Nlocbeg] for i in range(Ntbeg)]
    blockloc_out = [[i*Nlocout,(i+1)*Nlocout] for i in range(Ntout)]
    blockloc = [blockloc_in,blockloc_out]
    overlaps={}
    for iton,ito in enumerate(timeout):
        overlaps[iton]=[]
        for ix, x in enumerate(timein):
            if ( x[0]<ito[1] or x==(Ntbeg-1)) and x[1]>ito[0]:
                
                #find amount of time for overlap

                stp = sp.maximum(x[0],ito[0])
                if ix ==Ntbeg-1:
                    enp=ito[1]
                else:
                    enp = sp.minimum(x[1],ito[1])
                ratio = float(enp-stp)/Tint
                # need to find the start point
                T_1 = float(stp-x[0])
                newcartcoords1 = Cart_Coords-T_1*vel[:,ix]*1e-3 # check velocity is in km/s or m/s
                T_2=float(enp-x[0])
                newcartcoords2 = Cart_Coords-T_2*vel[:,ix]*1e-3 # check velocity is in km/s or m/s
                newcoorsds1 = cart2sphere(newcartcoords1)
                newcoorsds2 = cart2sphere(newcartcoords2)
                overlaps[iton].append([ix,ratio,newcoorsds1,newcoorsds2])
    # make the matrix
    for iton,ito in enumerate(timeout):
        cur_over = overlaps[iton]
        for it_in,it_info in enumerate(cur_over):
            
            cur_it,cur_ratio,Sp1,Sp2 = it_info
            rho1 = Sp1[:,0]
            Az1 = Sp1[:,1]
            El1 = Sp1[:,2]
            rho2 = Sp2[:,0]
            Az2 = Sp2[:,1]
            El2 = Sp2[:,2]
            # get the weights
            weights1 = {ibn:sensdict['ArrayFunc'](Az1,El1,ib[0],ib[1],sensdict['Angleoffset']) for ibn, ib in enumerate(angles)}
            weights2 = {ibn:sensdict['ArrayFunc'](Az2,El2,ib[0],ib[1],sensdict['Angleoffset']) for ibn, ib in enumerate(angles)}
    
            for ibn in range(Nbeams):
                

                print('\t\t Making Beam {0:d} of {1:d}'.format(ibn,Nbeams))
                weight1 = weights1[ibn]
                weight2 = weights2[ibn]
                for isamp in range(nrgout):
                    # make the row
                    irow = ibn + isamp*Nbeams + Nbeams*nrgout*iton
                    range_g = rng_vec2[isamp]
                    rnglims = [range_g-rng_len/2.,range_g+rng_len/2.]
                    # assume centered lag product.
                    rangelog = ((rho1>=rnglims[0])&(rho1<rnglims[1]))
                    # This is a nearest neighbors interpolation for the spectrums in the range domain
                    if sp.sum(rangelog)==0:
                        minrng = sp.argmin(sp.absolute(range_g-rho1))
                        rangelog[minrng] = True
                    #create the weights and weight location based on the beams pattern.
                    weight_cur =weight1[rangelog]
                    weight_cur = weight_cur/weight_cur.sum()
                    icols = sp.where(rangelog)[0] + Nlocbeg*cur_it
                    
                    weights_final = weight_cur*range_g**2/rho1[rangelog]**2
                    outmat[irow,icols] = weights_final*ratio*0.5 +outmat[irow,icols]
                    
                    # assume centered lag product.
                    rangelog = ((rho2>=rnglims[0])&(rho2<rnglims[1]))
                    # This is a nearest neighbors interpolation for the spectrums in the range domain
                    if sp.sum(rangelog)==0:
                        minrng = sp.argmin(sp.absolute(range_g-rho2))
                        rangelog[minrng] = True
                    #create the weights and weight location based on the beams pattern.
                    weight_cur =weight2[rangelog]
                    weight_cur = weight_cur/weight_cur.sum()
                    icols = sp.where(rangelog)[0]+ Nlocbeg*cur_it
    
                    weights_final = weight_cur*range_g**2/rho2[rangelog]**2
                    outmat[irow,icols] = weights_final*ratio*0.5+outmat[irow,icols]
                
                



    return(outmat,overlaps,blockloc)


def saveoutmat(filename,Sphere_Coords,timein,configfile):
    """ This will save the outmatrix into an h5 file.
    """
    outmat = makematPA(Sphere_Coords,timein,configfile)
    h5file = tables.openFile(filename, mode = "w", title = "Radar Matrix out.")
    h5file.createArray('/','RadarMatrix',outmat,'Static array')
    h5file.close()

def readradarmat(filename):
    """ This will read in a precomuted matrix form an h5 file.
    """
    h5file=tables.openFile(filename)
    outmat = h5file.root.RadarMatrix.read()
    h5file.close()
    return outmat
#%% extra functions
def getOverlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

#%% Math Functions
def cgmat(A,x,b,M=None,max_it=100,tol=1e-8):
    """ This function will performa conjuguate gradient search to find the inverse of
        an operator A, given a starting point x, and data b.
    """
    if M is None:
        M= sp.diag(A)
    bnrm2 = sp.linalg.norm(b)
    r=b-A.dot(x)
    rho=sp.zeros(max_it)
    for i in range(max_it):
        z=sp.linalg.solve(M,r)
        rho[i] = sp.dot(r,z)
        if i==0:
            p=z
        else:
            beta=rho/rho[i-1]
            p=z+beta*p

        q=A.dot(p)
        alpha=rho/sp.dot(p,q)
        x = x+alpha*p
        r = r-alpha*q
        error = sp.linalg.norm( r ) / bnrm2
        if error <tol:
            return (x,error,i,False)

    return (x,error,max_it,True)


def diffmat(dims,order = 'C'):
    """ This function will return a tuple of difference matricies for data from an 
        Nd array that has been rasterized. The order parameter determines whether 
        the array was rasterized in a C style (python) of FORTRAN style (MATLAB).
    """
    xdim = dims[0]
    ydim = dims[1]
    dims[0]=ydim
    dims[1]=xdim

    if order.lower() == 'c':
        dims = dims[::-1]

    outD = []
    for idimn, idim in enumerate(dims):
        if idim==0:
            outD.append(sp.array([]))
            continue
        e = sp.ones(idim)
        dthing = sp.column_stack((-e,e))
        D = sp.sparse.spdiags(dthing,[0,1],idim-1,idim)
        D = sp.vstack((D,D[-1]))
        if idimn>0:
            E = sp.sparse.eye(sp.prod(dims[:idimn]))
            D = sp.kron(D,E)

        if idimn<len(dims)-1:
            E = sp.sparse.eye(sp.prod(dims[idimn+1:]))
            D = sp.kron(E,D)

        outD.append(D)
    if order.lower() == 'c':
        outD=outD[::-1]
    Dy=outD[0]
    Dx = outD[1]
    outD[0]=Dx
    outD[1]=Dy

    return tuple(outD)

def cart2sphere(coordlist):
    r2d = 180.0/sp.pi
    d2r = sp.pi/180.0
    
    X_vec = coordlist[:,0]
    Y_vec = coordlist[:,1]
    Z_vec = coordlist[:,2]
    R_vec = sp.sqrt(X_vec**2+Y_vec**2+Z_vec**2)
    Az_vec = sp.arctan2(Y_vec,X_vec)*r2d
    El_vec = sp.arcsin(Z_vec/R_vec)*r2d
    sp_coords = sp.array([R_vec,Az_vec,El_vec]).transpose()
    return sp_coords
    
    
    
