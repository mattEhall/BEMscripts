# ----------------------------------------------------------------------------------
#    Python code for creating a mesh for a set of (one or more) axisymmetric bodies,
#       for use with WAMIT as a .gdf file. Could be adapted to other file types.
#
#                      Created by Matt Hall 2012ish-present
# ----------------------------------------------------------------------------------
#
# This is derived from CreateDesign.cpp, which is derived from CreateGeometry.cpp, which was from my master's work (2012).
# It's for making .gdf file meshes of platforms, and it uses adaptive azimuthal discretizations to control panel aspect ratios.
# 2020-06-08: modifying to mesh based on individual members (see Mesher.py for additional meshing scraps)

import numpy as np

'''
function to return conical taper geometry calculations
void TaperCalc(double R1, double R2, double H, double* taperV, double* taperCV)
{
  if (R1 == R2)             # if just a cylinder
  {
      *taperV = pi*R1*R1*H;
      *taperCV = H/2.0;
  }
  else
  {
      double coneH = H/(1.-R2/R1);          # conical height
      double coneV = 1./3.*pi*R1*R1*coneH;  # cone volume
      double coneVtip = 1./3.*pi*R2*R2*(coneH-H);    # height from end of taper to cone tip
      *taperV = coneV-coneVtip;                       # taper volume
      *taperCV = ( coneV*1./4.*coneH - coneVtip*(1./4.*(coneH-H) + H) )/ *taperV; # from base
  }
}
'''

def printpanel(outlines, x, y, z):
    # writes one panel to the output file - parameters are output file, 4 x coords, 4 y coords, 4 z coords.
    
	outlines.append("{:06.2f} {:06.2f} {:06.2f}\n".format(x[0], y[0], z[0])) 
	outlines.append("{:06.2f} {:06.2f} {:06.2f}\n".format(x[1], y[1], z[1]))
	outlines.append("{:06.2f} {:06.2f} {:06.2f}\n".format(x[2], y[2], z[2]))
	outlines.append("{:06.2f} {:06.2f} {:06.2f}\n".format(x[3], y[3], z[3]))




# make the mesh for a single (vertical-for-now) member (which may have variable radius over its length)
def floatmesh(depths, radii, xc, yc, outlines, dz_max=0, da_max=0):

	# depths:  list of depths along member at which radius will be specified
	# radii: list of corresponding radii along member
	# xc, yc: member center coordinates
	# outlines: list of output lines that will be written to GDF file
	# dz_max: maximum panel height
	# da_max: maximum panel width (before doubling azimuthal discretization)
	
	
	# discretization defaults
	if dz_max==0:
		dz_max = depths[-1]/20
	if da_max==0:
		da_max = np.max(radii)/8
	
	

	# ------------------ discretize radius profile according to dz_max -------------

	
	# radius profile data is contained in r_rp and z_rp
	r_rp = [radii[0]] 
	z_rp = [0.0] 
	
	
	# step through each station and subdivide as needed
	for i_s in range(1, len(radii)):                       
	
		dr_s = radii[i_s] - radii[i_s-1];      # delta r
		dz_s = depths[ i_s] - depths[ i_s-1];      # delta z
        
		
		# subdivision size
		if dr_s == 0:          # vertical case
			cos_m=1
			sin_m=0
			dz_ps = dz_max;     # (dz_ps is longitudinal dimension of panel)
			
		elif dz_s == 0:          # horizontal case
			cos_m=0
			sin_m=1
			dz_ps = 0.6*da_max
			
		else:                  # angled case - set panel size as weighted average based on slope
			m = dr_s/dz_s;                    # slope = dr/dz
			dz_ps = np.arctan(np.abs(m))*2/np.pi*0.6*da_max + np.arctan(abs(1/m))*2/np.pi*dz_max; 
			cos_m = dz_s/np.sqrt(dr_s**2 + dz_s**2)
			sin_m = dr_s/np.sqrt(dr_s**2 + dz_s**2)
			
			#breakpoint()
			
		
		# make subdivision
		n_z = np.int(np.ceil( np.sqrt(dr_s*dr_s + dz_s*dz_s) / dz_ps ))    # local panel longitudinal discretization #
		
		d_l = np.sqrt(dr_s*dr_s + dz_s*dz_s)/n_z;                           # local panel longitudinal dimension
		
		for i_z in range(1,n_z+1):
		
			r_rp.append(  radii[i_s-1] + sin_m*i_z*d_l)
			z_rp.append(-depths[i_s-1] - cos_m*i_z*d_l)
	

	# fill in the bottom
	n_r = np.int(np.ceil( radii[-1] / (0.6*da_max) ))    # local panel radial discretization #
	dr = radii[-1] / n_r;                              # local panel radial size	
	
	for i_r in range(n_r): 
	
		r_rp.append(radii[-1] - (1+i_r)*dr)
		z_rp.append(-depths[-1])
	
	
#	nBody = len(r_rp)


#	# heave plate    
#	if (RHP > R):        # <----------- heave plate at bottom of spar
#	
#		n_r = np.ceil( (RHP - radii[-1]) / dz_max )    # local panel radial discretization #
#		dr = (RHP - radii[-1]) / n_r;                              # local panel radial size
#		for i_r in range(n_r+1):     # note zero index - this is because heave plate panelizing is done seperately
#		
#			r_rp.append( radii[-1] + i_r*dr );
#			z_rp.append(-H);
#
#	nHP = len(r_rp) - nBody



	# --------------- revolve radius profile, do adaptive paneling stuff ----------


	npan =0;
	naz = np.int(8);


	# go through each point of the radius profile, panelizing from top to bottom:
	for i_rp in range(len(z_rp)-1):  
    
	#	# check for beginning of heave plate section (has dipole panels)                         
	#	if i_rp == nBody-1:
	#		HPstart = npan + 1;
	#		continue
		
                  
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




#  ---------------- script to make the DeepCwind semi mesh using floatmesh -------------------


outlines = []  # a list to hold each line of the GDF file data

#              depths,      radii,       xc,    yc,   outlines, dz_max    da_max
floatmesh( [0,20],        [3.25,3.25],   0,      0,   outlines, dz_max=1, da_max=2)
floatmesh( [0,14,14,20],  [6,6,12,12],  14.43,  25,   outlines, dz_max=1, da_max=2)
floatmesh( [0,14,14,20],  [6,6,12,12], -28.87,   0,   outlines, dz_max=1, da_max=2)
floatmesh( [0,14,14,20],  [6,6,12,12],  14.43, -25,   outlines, dz_max=1, da_max=2)


npan = np.int(len(outlines)/4) # number of panels


# write GDF file
outfile = open("mesh.gdf","w")

outfile.write("WAMIT GDF file\n")
outfile.write("1.000   9.8066   ULEN, GRAV\n")
outfile.write("0       0        ISX, ISY\n")
outfile.write("{:d}      NEQN\n".format(npan))

for line in outlines:
	outfile.write(line)

outfile.close()




