import math
import numpy as np
import meshBEM as mb

# some variables
d2r = math.pi/180
radiusHemisphere = 10.0
arcPts = 13 # [13, 17, 21, 25, 29, 33, 37]
segments = 17 # [17, 21, 25, 29, 33, 37, 41]
# for cosine spacing
linspace = np.linspace(0, 90, arcPts)
cosineSpacing = np.cos(d2r*linspace)*(90*d2r)

# calculate x and z coordinates along the hemisphere's profile
xCurve = radiusHemisphere*np.sin(cosineSpacing)
zCurve = -radiusHemisphere*np.cos(cosineSpacing)
xTop = np.cos(d2r*np.linspace(270,360,int(arcPts/2)))*xCurve[0]
zTop = np.full(len(xTop), 0.0) #zCurve[0])
zCurve[0] = 0.0 # force top point to be exactly equal to waterline
# join all points from curve and top 
xPts = np.append(xTop[:-1], xCurve)
zPts = np.append(zTop[:-1], zCurve)
# create 3D mesh points and panels
ntheta = segments#[i]
x3d, y3d, z3d = mb.create_3d_points(xPts, zPts, ntheta)
pts = np.array([x3d, y3d, z3d])
panels = mb.create_3d_panels(len(xPts), ntheta)

# hemiMesh = mb.Mesh(meshName='hemiMesh', x3d=x3d, y3d=y3d, z3d=z3d, panels=panels)
meshName = 'hemiMesh'
hemiMesh = mb.Mesh(meshName, x3d, y3d, z3d, panels)
hemiMesh.save_to_nemoh_mesh()
hemiMesh.save_to_gdf_mesh()


# numPanels = len(panels[0,:])
# modelName = f'hemisphere{numPanels}'
# modelDir = join(getcwd(), modelName)
# if exists(modelDir):
#     chdir(modelDir)
# else:
#     mkdir(modelDir)
#     chdir(modelDir)
# save_to_gdf_mesh(f'{modelName}.gdf', x3d, y3d, z3d, panels)
# hemisphere = FloatingBody.from_file(f'{modelName}.gdf',
#                                     file_format='gdf')
# # hemisphereMesh = heal_normals(hemisphere.mesh)
# # hemisphereNew = FloatingBody(mesh=hemisphereMesh)
# #hemisphere.show()
# potFileName = f'{modelName}.pot'
# meshFileName = f'{modelName}.gdf'
# write_wamit_pot(fileName=potFileName, meshFile=meshFileName,
#                 xbody=[0.0, 0.0, 0.0, 0.0])
# write_wamit_fnames(modelName)
# write_wamit_cfg(modelName)
# write_wamit_frc(modelName, cogZ=[-4.0])
# write_wamit_config()
# startTime = time.time()
# subprocess.run('wamit')
# compTime = time.time() - startTime
# chdir('..')
# with open('computation_time_log.txt', 'a') as log:
#     log.write(f'{modelName:<20} {compTime:.3f}\n')
# log.close()
