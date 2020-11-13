import numpy as np
import meshBEM as mb

pts = mb.stepped_cylinder_open_xsection(5.0, -15.0, 9,
                                        10.0, -5.0, 7,
                                        5, 5)

mb.plot_profile(pts[0,:], pts[1,:])

nTheta = 17
panels = mb.axisym_3d_panels(len(pts[0,:]), nTheta)
points = mb.axisym_3d_points(pts[0,:], pts[1,:], nTheta)

mesh = mb.Mesh(f'steppedCylinder', points, panels, cog=[0.0, 0.0, -10.0])
mesh.plot_points_3d()
mesh.save_to_nemoh_mesh()
