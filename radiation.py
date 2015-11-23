#!/usr/bin/env python
"""
Simple script that plots the P and S wave radiation pattern
of an arbitrary moment tensor in 3D

UNTESTED
Matthias Meschede 2015
"""

import numpy as np
from obspy.imaging.scripts.mopad import MomentTensor, BeachBall
from mayavi import mlab

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

def main():
    mt = [7.982,-0.154, -7.574, 0.548, 7.147, -0.211] #focal mechanism
    #mt = [0.,0.,0, 0., 0., 0.] #focal mechanism

    mopad_mt = MomentTensor(mt,system='NED')
    bb = BeachBall(mopad_mt, npoints=200)
    bb._setup_BB(unit_circle=False)

    # extract the coordinates of the nodal lines
    neg_nodalline = bb._nodalline_negative
    pos_nodalline = bb._nodalline_positive

    #plot radiation pattern and nodal lines
    pointsp,dispp = farfieldP(mt)
    pointss,disps = farfieldS(mt)

    npointsp = pointsp.shape[1]
    normp = np.sum(pointsp*dispp,axis=0)
    norms = np.sum(pointss*disps,axis=0)
    rangep = np.max(np.abs(normp))
    ranges = np.max(np.abs(normp))

    fig1 = mlab.figure(size=(800,800),bgcolor=(0,0,0))
    pts1 = mlab.quiver3d(pointsp[0],pointsp[1],pointsp[2],dispp[0],dispp[1],dispp[2],
                         scalars=normp,vmin=-rangep,vmax=rangep)
    pts1.glyph.color_mode = 'color_by_scalar'
    mlab.plot3d(*neg_nodalline,color=(0,0.5,0),tube_radius=0.01)
    mlab.plot3d(*pos_nodalline,color=(0,0.5,0),tube_radius=0.01)
    plot_sphere(0.7)

    fig2 = mlab.figure(size=(800,800),bgcolor=(0,0,0))
    mlab.quiver3d(pointss[0],pointss[1],pointss[2],disps[0],disps[1],disps[2],
                  vmin=-ranges,vmax=ranges)
    mlab.plot3d(*neg_nodalline,color=(0,0.5,0),tube_radius=0.01)
    mlab.plot3d(*pos_nodalline,color=(0,0.5,0),tube_radius=0.01)
    plot_sphere(0.7)

    mlab.show()

    #fig = plt.figure()
    #ax  = fig.add_subplot(111, projection='3d')
    #ax.quiver(pointsp[0],pointsp[1],pointsp[2],disp[0],disp[1],disp[2],length=vlength)
    #ax.plot(neg_nodalline[0],neg_nodalline[1],neg_nodalline[2],c='r')
    #ax.plot(pos_nodalline[0],pos_nodalline[1],pos_nodalline[2],c='r')

    #fig = plt.figure()
    #ax  = fig.add_subplot(111, projection='3d')
    #ax.quiver(pointss[0],pointss[1],pointss[2],disp[0],disp[1],disp[2],length=vlength)
    #ax.plot(neg_nodalline[0],neg_nodalline[1],neg_nodalline[2],c='r')
    #ax.plot(pos_nodalline[0],pos_nodalline[1],pos_nodalline[2],c='r')

    #plt.show()

def plot_sphere(r):
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0:pi:101j, 0:2 * pi:101j]

    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)

    return mlab.mesh(x, y, z,color=(0,0,0))

def farfieldP(mt):
    """
    This function is based on Aki & Richards Eq 4.29
    """
    ndim = 3
    Mpq = fullmt(mt)

    #---- make spherical grid ----
    nlats  = 30
    colats = np.linspace(0.,np.pi,nlats)
    norms = np.sin(colats)
    nlons = (nlats*norms+1).astype(int)
    colatgrid,longrid = [],[]
    for ilat in range(nlats):
        nlon = nlons[ilat]
        dlon = 2.*np.pi/nlon
        lons = np.arange(0.,2.*np.pi,dlon)
        for ilon in range(nlon):
            colatgrid.append(colats[ilat])
            longrid.append(lons[ilon])
    npoints = len(longrid)

    #---- get cartesian coordinates of spherical grid ----
    points = np.empty( (ndim,npoints) )
    points[0] = np.sin(colatgrid)*np.cos(longrid)
    points[1] = np.sin(colatgrid)*np.sin(longrid)
    points[2] = np.cos(colatgrid)

    #---- precompute directional cosine array ----
    dists = np.sqrt(points[0]*points[0]+points[1]*points[1]+points[2]*points[2])
    gammas = np.empty( (ndim,npoints) )
    gammas[0] = points[0]/dists
    gammas[1] = points[1]/dists
    gammas[2] = points[2]/dists

    #---- initialize displacement array ----
    disp   = np.empty( (ndim,npoints) )

    #---- loop through points ----
    for ipoint in range(npoints):
      #---- loop through displacement component [n index] ----
      gamma = gammas[:,ipoint]
      gammapq = np.outer(gamma,gamma)
      gammatimesmt = gammapq*Mpq
      for n in range(ndim):
          disp[n,ipoint] = gamma[n]*np.sum(gammatimesmt.flatten())

    return points,disp

def farfieldS(mt):
    """
    This function is based on Aki & Richards Eq 4.29
    """
    ndim = 3
    Mpq = fullmt(mt)

    #---- make spherical grid ----
    nlats  = 30
    colats = np.linspace(0.,np.pi,nlats)
    norms = np.sin(colats)
    nlons = (nlats*norms+1).astype(int)
    colatgrid,longrid = [],[]
    for ilat in range(nlats):
        nlon = nlons[ilat]
        dlon = 2.*np.pi/nlon
        lons = np.arange(0.,2.*np.pi,dlon)
        for ilon in range(nlon):
            colatgrid.append(colats[ilat])
            longrid.append(lons[ilon])
    npoints = len(longrid)

    #---- get cartesian coordinates of spherical grid ----
    points = np.empty( (ndim,npoints) )
    points[0] = np.sin(colatgrid)*np.cos(longrid)
    points[1] = np.sin(colatgrid)*np.sin(longrid)
    points[2] = np.cos(colatgrid)

    #---- precompute directional cosine array ----
    dists = np.sqrt(points[0]*points[0]+points[1]*points[1]+points[2]*points[2])
    gammas = np.empty( (ndim,npoints) )
    gammas[0] = points[0]/dists
    gammas[1] = points[1]/dists
    gammas[2] = points[2]/dists

    #---- initialize displacement array ----
    disp   = np.empty( (ndim,npoints) )

    #---- loop through points ----
    for ipoint in range(npoints):
      #---- loop through displacement component [n index] ----
      gamma = gammas[:,ipoint]
      Mp = np.dot(Mpq,gamma)
      for n in range(ndim):
          psum = 0.0
          for p in range(ndim):
              deltanp = int(n==p)
              psum += (gamma[n]*gamma[p] - deltanp)*Mp[p]
          disp[n,ipoint] = psum

    return points,disp

def fullmt(mt):
    mt_full = np.array( ([[mt[0],mt[3],mt[4]],
                          [mt[3],mt[1],mt[5]],
                          [mt[4],mt[5],mt[2]]]) )
    return mt_full

#==== SCRIPT EXECUTION ====
if __name__ == "__main__":
    main()

