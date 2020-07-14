import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import logging
import subprocess
import time

#from numpy import pi, sin, cos, linspace, full, concatenate
from mpl_toolkits.mplot3d import Axes3D
from capytaine import (FloatingBody, BEMSolver, RadiationProblem,
                       DiffractionProblem, assemble_dataset)
from capytaine.io.legacy import write_dataset_as_tecplot_files
from capytaine.io.xarray import separate_complex_values
from capytaine.meshes.quality import (merge_duplicates, heal_normals, remove_unused_vertices,
                                      heal_triangles, remove_degenerated_faces)
from os import chdir, mkdir, getcwd
from os.path import splitext, join, exists
from datetime import datetime


def printpanel(outlines, x, y, z):
    '''write one panel to the output file - parameters are output file, 4 x
    coords, 4 y coords, 4 z coords.'''
    outlines.append("{:06.2f} {:06.2f} {:06.2f}\n".format(x[0], y[0], z[0]))
    outlines.append("{:06.2f} {:06.2f} {:06.2f}\n".format(x[1], y[1], z[1]))
    outlines.append("{:06.2f} {:06.2f} {:06.2f}\n".format(x[2], y[2], z[2]))
    outlines.append("{:06.2f} {:06.2f} {:06.2f}\n".format(x[3], y[3], z[3]))

# make the mesh for a single (vertical-for-now) member (which may have variable
# radius over its length)
def floatmesh(depths, radii, xc, yc, outlines, dz_max=0, da_max=0):
    """
    Function for creating cylindrical float with/without varying radius

    Args:
        depths:  list of depths along member at which radius will be specified
        radii: list of corresponding radii along member
        xc, yc: member center coordinates
        outlines: list of output lines that will be written to GDF file
        dz_max: maximum panel height
        da_max: maximum panel width (before doubling azimuthal discretization)

    Returns:

    Notes:
        no returns; calls printpanel() to write the mesh panels (4 x nodes make
        up one panel)
    """

    # discretization defaults
    if dz_max==0:
        dz_max = depths[-1]/20
        if da_max==0:
            da_max = np.max(radii)/8

    # ------------------ discretize radius profile according to dz_max --------

    # radius profile data is contained in r_rp and z_rp
    r_rp = [radii[0]]
    z_rp = [0.0]

    # step through each station and subdivide as needed
    for i_s in range(1, len(radii)):
        dr_s = radii[i_s] - radii[i_s-1]; # delta r
        dz_s = depths[ i_s] - depths[ i_s-1]; # delta z
        # subdivision size
        if dr_s == 0: # vertical case
            cos_m=1
            sin_m=0
            dz_ps = dz_max; # (dz_ps is longitudinal dimension of panel)
        elif dz_s == 0: # horizontal case
            cos_m=0
            sin_m=1
            dz_ps = 0.6*da_max
        else: # angled case - set panel size as weighted average based on slope
            m = dr_s/dz_s; # slope = dr/dz
            dz_ps = np.arctan(np.abs(m))*2/np.pi*0.6*da_max + np.arctan(abs(1/m))*2/np.pi*dz_max;
            cos_m = dz_s/np.sqrt(dr_s**2 + dz_s**2)
            sin_m = dr_s/np.sqrt(dr_s**2 + dz_s**2)
            #breakpoint()
        # make subdivision
        # local panel longitudinal discretization
        n_z = np.int(np.ceil( np.sqrt(dr_s*dr_s + dz_s*dz_s) / dz_ps ))
        # local panel longitudinal dimension
        d_l = np.sqrt(dr_s*dr_s + dz_s*dz_s)/n_z;
        for i_z in range(1,n_z+1):
            r_rp.append(  radii[i_s-1] + sin_m*i_z*d_l)
            z_rp.append(-depths[i_s-1] - cos_m*i_z*d_l)

    # fill in the bottom
    # local panel radial discretization #
    n_r = np.int(np.ceil( radii[-1] / (0.6*da_max) ))
    # local panel radial size
    dr = radii[-1] / n_r;

    for i_r in range(n_r):
        r_rp.append(radii[-1] - (1+i_r)*dr)
        z_rp.append(-depths[-1])
    # nBody = len(r_rp)
    # # heave plate
    # if (RHP > R):        # <----------- heave plate at bottom of spar

    ## local panel radial discretization
    # 	n_r = np.ceil( (RHP - radii[-1]) / dz_max )
    ## local panel radial size
    # 	dr = (RHP - radii[-1]) / n_r;
    ## note zero index - because heave plate panelizing is done seperately
    # 	for i_r in range(n_r+1):

    # 		r_rp.append( radii[-1] + i_r*dr );
    # 		z_rp.append(-H);

    # nHP = len(r_rp) - nBody

    # --------------- revolve radius profile, do adaptive paneling stuff ------

    npan =0;
    naz = np.int(8);

    # go through each point of the radius profile, panelizing from top to bottom:
    for i_rp in range(len(z_rp)-1):
        # check for beginning of heave plate section (has dipole panels)
        # if i_rp == nBody-1:
        # HPstart = npan + 1;
        # continue
        x=np.zeros(4)
        y=np.zeros(4)
        z=np.zeros(4)
        th1 = 0
        th2 = 0
        th3 = 0
        # rectangle coords - shape from outside is:  A D
        #                                            B C
        r1=r_rp[i_rp];
        r2=r_rp[i_rp+1];
        z1=z_rp[i_rp];
        z2=z_rp[i_rp+1];

        # scale up or down azimuthal discretization as needed
        while ( (r1*2*np.pi/naz >= da_max/2) and (r2*2*np.pi/naz >= da_max/2) ):
            naz = np.int(2*naz)
        while ( (r1*2*np.pi/naz < da_max/2) and (r2*2*np.pi/naz < da_max/2) ):
            naz = np.int(naz/2)

        # transition - increase azimuthal discretization
        if ( (r1*2*np.pi/naz < da_max/2) and (r2*2*np.pi/naz >= da_max/2) ):
            for ia in range(1, np.int(naz/2)+1):
                th1 = (ia-1  )*2*np.pi/naz*2;
                th2 = (ia-0.5)*2*np.pi/naz*2;
                th3 = (ia    )*2*np.pi/naz*2;

                x = np.array([xc+r1*np.cos(th1), xc+r2*np.cos(th1), xc+r2*np.cos(th2), xc+(r1*np.cos(th1)+r1*np.cos(th3))/2 ])
                y = np.array([yc+r1*np.sin(th1), yc+r2*np.sin(th1), yc+r2*np.sin(th2), yc+(r1*np.sin(th1)+r1*np.sin(th3))/2 ])
                z = np.array([z1            , z2            , z2            , z1                             ])

                printpanel(outlines, x, y, z)
                npan += 1

                x = np.array([xc+(r1*np.cos(th1)+r1*np.cos(th3))/2, xc+r2*np.cos(th2), xc+r2*np.cos(th3), xc+r1*np.cos(th3)])
                y = np.array([yc+(r1*np.sin(th1)+r1*np.sin(th3))/2, yc+r2*np.sin(th2), yc+r2*np.sin(th3), yc+r1*np.sin(th3)])
                z = np.array([z1                            , z2            , z2            , z1            ])

                printpanel(outlines, x, y, z)
                npan += 1

        # transition - decrease azimuthal discretization
        elif ( (r1*2*np.pi/naz >= da_max/2) and (r2*2*np.pi/naz < da_max/2) ):
            for ia in range(1, np.int(naz/2)+1):
                th1 = (ia-1  )*2*np.pi/naz*2;
                th2 = (ia-0.5)*2*np.pi/naz*2;
                th3 = (ia    )*2*np.pi/naz*2;
                x = np.array([xc+r1*np.cos(th1), xc+r2*np.cos(th1), xc+r2*(np.cos(th1)+np.cos(th3))/2, xc+r1*np.cos(th2)])
                y = np.array([yc+r1*np.sin(th1), yc+r2*np.sin(th1), yc+r2*(np.sin(th1)+np.sin(th3))/2, yc+r1*np.sin(th2)])
                z = np.array([z1               , z2               , z2                               , z1               ])

                printpanel(outlines, x, y, z);
                npan += 1;

                x = np.array([xc+r1*np.cos(th2), xc+r2*(np.cos(th1)+np.cos(th3))/2, xc+r2*np.cos(th3), xc+r1*np.cos(th3)])
                y = np.array([yc+r1*np.sin(th2), yc+r2*(np.sin(th1)+np.sin(th3))/2, yc+r2*np.sin(th3), yc+r1*np.sin(th3)])
                z = np.array([z1               , z2                               , z2               , z1               ])

                printpanel(outlines, x, y, z)
                npan += 1

        # no transition
        else:
            for ia in range(1, naz+1):
                th1 = (ia-1)*2*np.pi/naz;
                th2 = (ia  )*2*np.pi/naz;
                x = np.array([xc+r1*np.cos(th1), xc+r2*np.cos(th1), xc+r2*np.cos(th2), xc+r1*np.cos(th2)])
                y = np.array([yc+r1*np.sin(th1), yc+r2*np.sin(th1), yc+r2*np.sin(th2), yc+r1*np.sin(th2)])
                z = np.array([z1            , z2            , z2            , z1            ])

                printpanel(outlines, x, y, z)
                npan += 1

#	if (HPstart>0):
#		HPend = npan;
#
#	HPlim = [0, 0]
#
#	# update heave plate panel indices and total number of panels
#	HPlim[0]=HPstart + npantot;
#	HPlim[1]=HPend + npantot;
#
#	npantot = npantot + npan;
#	return npantot, HPlim
    return


def plot_profile(x, z):
    '''plot the profile to be revolved around z axis (x, z coordinates)'''
    plt.scatter(x, z)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def axisym_3d_panels(xLen, nTheta):
    """
    define nodes in each panel for x-z points revolved around z-axis

    Args:
        xLen: no. of points in cross section
        nTheta: no of repetition intervals

    Returns:
        list of panels (one line contains 4 nodes, which make up a panel)

    Notes:

    """
    panels = np.zeros((4, (xLen-1)*(nTheta-1)))
    iPanel = 0

    for i in range(xLen-1):
        for j in range(nTheta-1):
            panels[0, iPanel] = (i+1) + xLen*j
            panels[1, iPanel] = (i+1) + 1 + xLen*j
            panels[2, iPanel] = (i+1) + 1 + xLen*(j+1)
            panels[3, iPanel] = (i+1) + xLen*(j+1)
            iPanel = iPanel+1

    return panels


def axisym_3d_points(x, z, nTheta):
    '''take x, z coordinates and revolve around z axis at 360/nTheta spacing'''
    d2r = np.pi/180
    theta = np.linspace(0, 360*d2r, nTheta)
    x3d = np.zeros(len(x)*nTheta)
    y3d = np.zeros(len(x)*nTheta)
    z3d = np.zeros(len(x)*nTheta)
    iPoint = 0

    for i in range(nTheta):
        for j in range(len(x)):
            x3d[iPoint] = x[j]*np.cos(theta[i])
            y3d[iPoint] = x[j]*np.sin(theta[i])
            z3d[iPoint] = z[j]
            iPoint = iPoint+1

    return x3d, y3d, z3d


class Mesh:

    def __init__(self, meshName, points, panels, cog=[0.0, 0.0, 0.0]):
        self.meshName = meshName
        self.points = points
        self.x3d = self.points[0,:]
        self.y3d = self.points[1,:]
        self.z3d = self.points[2,:]
        self.panels = panels
        self.numPoints = len(self.points[0,:])
        self.numPanels = np.max(panels.shape)
        self.nemohSaveName = f'{meshName}.nemoh'
        self.wamitSaveName = f'{meshName}.gdf'
        self.cog = np.asarray(cog)
        self.xBody = np.append(self.cog, 0.0)

    def shear_x(self, xShearFactor):
        shearMat = np.array([[1,0,0], [0,1,0], [-xShearFactor,0,1]])
        for i in range(self.numPoints):
            vec = self.points[:,i]
            self.points[:,i] = vec.dot(shearMat)
        self.cog = self.cog.dot(shearMat)

    def scale_z(self, zScaleFactor):
        self.points[2,:] *= zScaleFactor
        self.cog[2] *= zScaleFactor

    def translate_x(self, xTranslateFactor):
        self.points[0,:] += xTranslateFactor
        self.cog[0] += xTranslateFactor

    def translate_z(self, zTranslateFactor):
        self.points[2,:] += zTranslateFactor
        self.cog[2] += zTranslateFactor

    def set_axes_equal(self, ax):
        '''Make axes of 3D plot have equal scale'''
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    def plot_points_3d(self):
        '''plot 3D points'''
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.x3d, self.y3d, self.z3d)
        self.set_axes_equal(ax)
        plt.show()

    def save_to_nemoh_mesh(self, saveName=None):
        '''save mesh points and panels to nemoh format'''
        if saveName==None:
            saveName = self.nemohSaveName

        f = open(saveName, "w")
        f.write(f'{len(self.panels[0,:])} {len(self.x3d)}\n')
        for i in range(self.numPoints):
            f.write(f'{i+1} {self.x3d[i]} {self.y3d[i]} {self.z3d[i]}\n')
        f.write('0 0 0 0\n')
        for i in range(self.numPanels):
            f.write(f'{self.panels[0, i]:.0f} {self.panels[1, i]:.0f} {self.panels[2, i]:.0f} {self.panels[3, i]:.0f}\n')
        f.write('0 0 0 0\n')
        f.close()

    def save_to_gdf_mesh(self, saveName=None, scale=1.0, gravity=9.81,
                         symmetry=[0, 0]):
        '''save mesh to gdf format'''
        if saveName==None:
            saveName = self.wamitSaveName

        currentTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        f = open(saveName, "w")
        f.write(f'{self.meshName} gdf mesh created at {currentTime} \n')
        f.write(f'{scale}    {gravity}\n')
        f.write(f'{symmetry[0]}    {symmetry[1]}\n')
        f.write(f'{self.numPanels}\n')

        for panel in range(self.numPanels):
            panelNodes = self.panels[:,panel]
            for i in range(4):
                idx = int(panelNodes[i]) - 1
                f.write(f'{self.x3d[idx]:>20.12e} {self.y3d[idx]:>20.12e} {self.z3d[idx]:>20.12e}\n')
        f.close()

    def write_wamit_pot(self, potFileName, meshFileNames, nBodys, xBodys,
                        waterDepth=-1, iRad=1, iDiff=1, nPer=-402,
                        per=[-0.02, 0.02], nBetas=1, betas=[0.0]):
        '''write wamit .pot file'''
        f = open(potFileName, 'w')
        f.write(f'{potFileName}\n')
        f.write(f' {waterDepth:<32} HBOT\n')
        f.write(f' {iRad:<8}{iDiff:<24} IRAD, IDIFF\n')
        f.write(f' {nPer:<32} NPER\n')
        f.write(f' {per[0]:<8}{per[1]:<24} PER\n')
        f.write(f' {nBeta:<32} NBETA\n')
        for beta in betas:
            f.write(f' {beta:<8}')
        f.write(f'BETA\n')
        f.write(f' {nBodys:<32} NBODY\n')
        for i, mesh in enumerate(meshFileNames):
            f.write(f' {mesh:<32}\n')
            f.write(f' {xBodys[i,0]:<10.4f}{xBodys[i,1]:<10.4f}{xBodys[i,2]:<10.4f}{xBodys[i,3]:<10.4f} XBODY(1-4)\n')
            f.write(f' {"1 1 1 1 1 1":<32} IMODE(1-6)\n')
        f.close()

    def write_wamit_fnames(self, modelName, meshFileNames):
        '''write wamit fnames.wam file'''
        f = open(f'fnames.wam', 'w')#, encoding='ascii')
        f.write(f'{modelName}.cfg\n')
        for mesh in meshFileNames:
            f.write(f'{mesh}\n')
        f.write(f'{modelName}.pot\n')
        f.write(f'{modelName}.frc\n')
        f.close()

    def write_wamit_cfg(self, modelName, iLog=1, iPerIn=2, irr=1, iSolve=1, numHeaders=1):
        '''write wamit .cfg file'''
        f = open(f'{modelName}.cfg', 'w')
        f.write(f'! {modelName}\n')
        f.write(f' ILOG={iLog:<19} (1 - panels on free surface)\n')
        f.write(f' IPERIN={iPerIn:<17} (1 - T, 2 - w)\n')
        f.write(f' IRR={irr:<20} (0 - not remove irr freq, 1 - remove irr freq, pannels on free surface)\n')
        f.write(f' ISOLVE={iSolve:<17} (0 - iterative solver, 1 - direct solver)\n')
        f.write(f' NUMHDR={numHeaders:<17} (0 - no output headers, 1 - output headers)\n')
        f.close()

    def write_wamit_frc(self, modelName, meshFileNames, meshCogZ):
        '''write wamit .frc file'''
        f = open(f'{modelName}.frc', 'w')
        f.write(f' {modelName}.frc\n')
        f.write(f' {"1 0 1 0 0 0 0 0 0":<24} IOPTN(1-9)\n')
        for i, mesh in enumerate(meshFileNames):
            f.write(f' {meshCogZ[i]:<24} VCG({i+1})\n')
            f.write(f' {"0.0  0.0  0.0":<24}\n')
            f.write(f' {"0.0  0.0  0.0":<24}\n')
            f.write(f' {"0.0  0.0  0.0":<24} XPRDCT\n')
        f.write(f' {0:<24d} NBETAH\n')
        f.write(f' {0:<24d} NFIELD\n')
        f.close()

    def write_wamit_config(self, ramGBMax=60.0, numCPU=10, licensePath=f'\wamitv7\license'):
        '''write wamit config.wam file'''
        f = open(f'config.wam', 'w')
        f.write(f' generic configuration file:  config.wam\n')
        f.write(f' RAMGBMAX={ramGBMax}\n')
        f.write(f' NCPU={numCPU}\n')
        f.write(f' USERID_PATH={licensePath} \t (directory for *.exe, *.dll, and userid.wam)\n')
        f.write(f' LICENSE_PATH={licensePath}')
        f.close()


class MultiMesh():
    def __init__(self, *Meshes, modelName):
        self.potFileName = f'{modelName}.pot'
        self.cfgFileName = f'{modelName}.cfg'
        self.frcFileName = f'{modelName}.frc'
        self.nBodys = len(Meshes)
        self.Meshes = Meshes
        self.modelName = modelName

    def write_meshes(self, bemCode='wamit'):
        for mesh in self.Meshes:
            if bemCode == 'wamit':
                mesh.save_to_gdf_mesh()
            elif bemCode == 'nemoh':
                mesh.save_to_nemoh_format()

    def write_wamit_pot(self, waterDepth=-1, iRad=1, iDiff=1, nPer=-402,
                        per=[-0.02, 0.02], nBetas=1, betas=[0.0]):
        '''write wamit .pot file'''
        f = open(self.potFileName, 'w')
        f.write(f'{self.potFileName}\n')
        f.write(f' {waterDepth:<32} HBOT\n')
        f.write(f' {iRad:<8}{iDiff:<24} IRAD, IDIFF\n')
        f.write(f' {nPer:<32} NPER\n')
        f.write(f' {per[0]:<8}{per[1]:<24} PER\n')
        f.write(f' {nBetas:<32} NBETA\n')
        for beta in betas:
            f.write(f' {beta:<7}')
        f.write(f'BETA\n')
        f.write(f' {self.nBodys:<32} NBODY\n')
        for mesh in self.Meshes:
            f.write(f' {mesh.wamitSaveName:<32}\n')
            for i in range(4):
                f.write(f' {mesh.xBody[i]:<7}')
            f.write(f'XBODY(1-4)\n')
            f.write(f' {"1 1 1 1 1 1":<32} IMODE(1-6)\n')
        f.close()

    def write_wamit_fnames(self):
        '''write wamit fnames.wam file'''
        f = open(f'fnames.wam', 'w')
        f.write(f'{self.cfgFileName}\n')
        for mesh in self.Meshes:
            f.write(f'{mesh.wamitSaveName}\n')
        f.write(f'{self.potFileName}\n')
        f.write(f'{self.frcFileName}\n')
        f.close()

    def write_wamit_cfg(self, iLog=1, iPerIn=2, irr=1, iSolve=1, numHeaders=1):
        '''write wamit .cfg file'''
        f = open(f'{self.cfgFileName}', 'w')
        f.write(f'! {self.modelName}\n')
        f.write(f' ILOG={iLog:<19} (1 - panels on free surface)\n')
        f.write(f' IPERIN={iPerIn:<17} (1 - T, 2 - w)\n')
        f.write(f' IRR={irr:<20} (0 - not remove irr freq, 1 - remove irr freq, pannels on free surface)\n')
        f.write(f' ISOLVE={iSolve:<17} (0 - iterative solver, 1 - direct solver)\n')
        f.write(f' NUMHDR={numHeaders:<17} (0 - no output headers, 1 - output headers)\n')
        f.close()

    def write_wamit_frc(self):
        '''write wamit .frc file'''
        f = open(f'{self.frcFileName}', 'w')
        f.write(f' {self.frcFileName}\n')
        f.write(f' {"1 0 1 0 0 0 0 0 0":<24} IOPTN(1-9)\n')
        for i, mesh in enumerate(self.Meshes):
            f.write(f' {"0.0":<24} VCG({i+1})\n')
            f.write(f' {"0.0  0.0  0.0":<24}\n')
            f.write(f' {"0.0  0.0  0.0":<24}\n')
            f.write(f' {"0.0  0.0  0.0":<24} XPRDCT\n')
        f.write(f' {0:<24d} NBETAH\n')
        f.write(f' {0:<24d} NFIELD\n')
        f.close()

    def write_wamit_config(self, ramGBMax=32.0, numCPU=6, licensePath=f'\wamitv7\license'):
        '''write wamit config.wam file'''
        f = open(f'config.wam', 'w')
        f.write(f' generic configuration file:  config.wam\n')
        f.write(f' RAMGBMAX={ramGBMax}\n')
        f.write(f' NCPU={numCPU}\n')
        f.write(f' USERID_PATH={licensePath} \t (directory for *.exe, *.dll, and userid.wam)\n')
        f.write(f' LICENSE_PATH={licensePath}')
        f.close()

materials = {'steel' : [7850, 3.0],
             'coated fabric' : [1400, 9.5],
             'sea water' : [1025, 0.0]}

class DiscWithTorus(Mesh):

    def __init__(self, meshName, points, panels,
                 discRadius, discThickness,
                 torusRadiusMinor, torusThickness,
                 discMaterial='steel',
                 torusInnerMaterial='sea water',
                 torusOuterMaterial='coated fabric'):
        super().__init__(meshName, points, panels)
        # disc properties
        self.discRadius = discRadius
        self.discThickness = discThickness
        self.discRho = materials[discMaterial][0]
        self.discVolume = np.pi*discRadius**2 * discThickness
        self.discMass = self.discVolume * self.discRho
        self.discIxx = (1.0/12.0) * self.discMass * (3*self.discRadius**2 +
                                                     self.discThickness**2)
        self.discIyy = self.discIxx
        self.discIzz = (1.0/2.0) * self.discMass * self.discRadius**2

        # torus properties
        self.torusRadiusMinor = torusRadiusMinor
        self.torusThickness = torusThickness
        self.torusRadiusMinorInner = torusRadiusMinor - torusThickness
        self.torusRadiusMajor = torusRadiusMinor + discRadius
        self.torusInnerRho = materials[torusInnerMaterial][0]
        self.torusOuterRho = materials[torusOuterMaterial][0]
        self.torusSurfaceArea = (4 * np.pi**2 * self.torusRadiusMajor *
                                 self.torusRadiusMinor) 
        self.torusOuterVolume = (self.torus_volume(self.torusRadiusMajor,
                                                   self.torusRadiusMinor)
                                 - self.torus_volume(self.torusRadiusMajor,
                                                     self.torusRadiusMinorInner))
        self.torusInnerVolume = self.torus_volume(self.torusRadiusMajor,
                                                  self.torusRadiusMinorInner) 
        self.torusInnerMass = self.torusInnerVolume * self.torusInnerRho
        self.torusOuterMass = self.torusOuterVolume * self.torusOuterRho
        self.torusInnerIxx = self.torus_ixx(self.torusInnerMass,
                                            self.torusRadiusMajor,
                                            self.torusRadiusMinorInner)
        self.torusInnerIyy = self.torusInnerIxx
        self.torusInnerIzz =  self.torus_izz(self.torusInnerMass,
                                             self.torusRadiusMajor,
                                             self.torusRadiusMinorInner)
        self.torusOuterIxx, self.torusOuterIyy, self.torusOuterIzz = (
            self.torus_hollow_ixxyyzz(self.torusOuterRho,
                                      self.torusRadiusMajor,
                                      self.torusRadiusMinor,
                                      self.torusRadiusMinorInner))
        self.torusMass = self.torusInnerMass + self.torusOuterMass
        self.torusIxx = self.torusInnerIxx + self.torusOuterIxx
        self.torusIyy = self.torusInnerIyy + self.torusOuterIyy
        self.torusIzz = self.torusInnerIzz + self.torusOuterIzz

        # combined disc & torus properties
        self.mass = self.discMass + self.torusMass
        self.Ixx = self.discIxx + self.torusIxx
        self.Iyy = self.discIyy + self.torusIyy
        self.Izz = self.discIzz + self.torusIzz

        # cost estimates
        self.discCost = self.discMass * materials[discMaterial][1]
        self.torusInnerCost = self.torusInnerMass * materials[torusInnerMaterial][1]
        self.torusOuterCost = self.torusOuterMass * materials[torusOuterMaterial][1]
        self.totalCost = self.discCost + self.torusInnerCost + self.torusOuterCost

    def torus_volume(self, rMajor, rMinor):
        return 2 * np.pi**2 * rMajor * rMinor**2

    def torus_ixx(self, mass, rMajor, rMinor):
        return (1.0/8.0) * mass * (4*rMajor**2 + 5*rMinor**2)

    def torus_izz(self, mass, rMajor, rMinor):
        return (1.0/4.0) * mass * (4*rMajor**2 + 3*rMinor**2)

    def torus_hollow_ixxyyzz(self, density, rMajor, rMinor, rMinorInner):
        volumeTotal = self.torus_volume(rMajor, rMinor)
        volumeInner = self.torus_volume(rMajor, rMinorInner)
        massTotal = volumeTotal * density
        massInner = volumeInner * density
        ixxTotal = self.torus_ixx(massTotal, rMajor, rMinor)
        ixxInner = self.torus_ixx(massInner, rMajor, rMinorInner)
        ixxOuter = ixxTotal - ixxInner
        iyyOuter = ixxOuter
        izzTotal = self.torus_izz(massTotal, rMajor, rMinor)
        izzInner = self.torus_izz(massInner, rMajor, rMinorInner)
        izzOuter = izzTotal - izzInner
        return ixxOuter, iyyOuter, izzOuter

    def write_report(self, filename=None):
        if filename==None:
            filename = f'./{self.meshName}.report'
        file = open(filename, 'w')
        currentTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        file.write(f'Report for {self.meshName} generated @ {currentTime}\n\n')
        file.write(f'Disc radius (m): {self.discRadius:.2f}\n')
        file.write(f'Disc thickness (m): {self.discThickness:.2f}\n')
        file.write(f'Torus radius (m): {self.torusRadiusMinor:.2f}\n')
        file.write(f'Torus thickness (m): {self.torusThickness:.2f}\n')
        file.write(f'Damper Ixx (kg m^2): {self.Ixx:.3f}\n')
        file.write(f'Damper Iyy (kg m^2): {self.Iyy:.3f}\n')
        file.write(f'Damper Izz (kg m^2): {self.Izz:.3f}\n')
        file.write(f'Damper mass (kg): {self.mass:.3f}\n')
        file.write(f'Damper cost ($): {self.totalCost:.2f}\n')

d2r = np.pi/180
r2d = 180/np.pi

def disc_with_torus_xsection(discRadius, discPoints, discThickness,
                             torusRadius, torusPoints):
    # define points along torus outer surface - in x-z cross-section
    xCentreTorus = discRadius + torusRadius
    zCentreTorus = 0.0
    # angleJoint: where torus meets disc, in degs
    angleJoint = np.sin((discThickness/2.0)/torusRadius) * r2d
    thetaTorus = np.linspace(-(90-angleJoint)*d2r, (270-angleJoint)*d2r, torusPoints)
    xTorus = xCentreTorus + torusRadius*np.sin(thetaTorus)
    zTorus = zCentreTorus + torusRadius*np.cos(thetaTorus)

    # define points along disc outer surface - in x-z cross-section
    linSpaceTop = np.linspace(270, 360, discPoints)
    cosSpaceTop = np.cos(d2r*linSpaceTop)
    xPlateTop = xTorus[0] * cosSpaceTop
    zPlateTop = np.full(len(xPlateTop), zTorus[0])
    linSpaceBottom = np.linspace(180, 270, discPoints)
    cosSpaceBottom = np.cos(d2r*linSpaceBottom)
    xPlateBottom = xTorus[-1] * cosSpaceBottom * -1
    zPlateBottom = np.full(len(xPlateBottom), zTorus[-1])

    # concatenate, remove duplicate points
    xPts = np.concatenate((xPlateTop[:-1], xTorus, xPlateBottom[1:]))
    zPts = np.concatenate((zPlateTop[:-1], zTorus, zPlateBottom[1:]))

    return xPts, zPts

def hemisphere_xsection(hemisphereRadius, hemispherePoints):
    linSpacing = np.linspace(0, 90, hemispherePoints)
    cosineSpacing = np.cos(d2r*linSpacing) * (90*d2r)
    arcXPts = hemisphereRadius*np.sin(cosineSpacing)
    arcZPts = -hemisphereRadius*np.cos(cosineSpacing)
    arcZPts[0] = 0.0 # enforce = 0.0 @ waterline
    # for closed hemisphere (@ waterline):
    topXPts = np.cos(d2r*np.linspace(270,360,int(hemispherePoints/2)))*arcXPts[0]
    topZPts = np.full(len(topXPts), 0.0) # enforce = 0.0 (waterline)
    xPts = np.append(topXPts[:-1], arcXPts)
    zPts = np.append(topZPts[:-1], arcZPts)

    return xPts, zPts
