import numpy as np


import open3d as o3d

mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh().create_sphere()
print(mesh)
pts = np.asanyarray(mesh.vertices)
idx = np.asanyarray(mesh.triangles)
np.save("../pts.npy", pts)
np.save("../idx.npy", idx)


