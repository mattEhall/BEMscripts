import numpy as np
import meshBEM as mb
# from meshWriter import save_to_gdf_mesh

#-------- script to make the DeepCwind semi mesh using floatmesh --------

outlines = []  # a list to hold each line of the GDF file data

#              depths,      radii,      xc,    yc, outlines, dz_max,    da_max
mb.floatmesh([0,20],       [3.25,3.25],   0,      0, outlines, dz_max=1, da_max=2)
mb.floatmesh([0,14,14,20], [6,6,12,12],  14.43,  25, outlines, dz_max=1, da_max=2)
mb.floatmesh([0,14,14,20], [6,6,12,12], -28.87,   0, outlines, dz_max=1, da_max=2)
mb.floatmesh([0,14,14,20], [6,6,12,12],  14.43, -25, outlines, dz_max=1, da_max=2)

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
