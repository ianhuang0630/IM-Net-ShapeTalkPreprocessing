import trimesh
import numpy as np
from util3d.voxel.convert import mesh_to_binvox
from binvox_rw_py import binvox_rw
import scipy.io as sio
import os

def load_obj(fn):
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()
    vertices = []; normals = []; faces=[];
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))

    f = np.vstack(faces) - 1 
    v = np.vstack(vertices)
    return v, f

def load_off(fn):
    with open(fn, 'r') as f:
        geo = trimesh.exchange.off.load_off(fn)
    f = geo['faces']
    v = geo['vertices']
    return v,f

vertices, faces = load_obj('model_normalized.obj')
binvox_path = 'model_normalized.binvox'
dim = 256 
mins = np.min(vertices, axis=0)
maxs = np.max(vertices, axis=0)
center = (mins + maxs) / 2
r = np.max(maxs - mins) / 2
lower = center - r
upper = center + r
bounding_box = tuple(lower) + tuple(upper)
mesh_to_binvox(
    vertices, faces, binvox_path, dim,
    bounding_box=bounding_box)

print('binvoxed.')

# read the binvox
with open(binvox_path, 'rb') as f:
    vox =  binvox_rw.read_as_3d_array(f)

print(vox.dims)
voxel_grid = vox.data
  
# now we divide the voxel grid into steps of 16
blocksizes = [int(el/16) for el in  vox.dims]
  
datablocks = []
ijk2indices = np.zeros((16,16,16))
for i in range(16):
    for j in range(16):
        for k in range(16):
            datablock = voxel_grid[i*16: i*16+blocksizes[0],
                                   j*16: j*16+blocksizes[1],
                                   k*16: k*16+blocksizes[2]]
            # you might consider asking if the matrix is a repeat, and if so, then don't
            # append datablocks, and overwrite the index.
            match = None
            for db_i, db in enumerate(datablocks):
                if np.array_equal(db, datablock):
                    print('duplicate block found at index {}'.format(db_i))
                    match = db_i
                    break
              
            if match is None:
                ijk2indices[i, j, k] = len(datablocks)
                datablocks.append(datablock) 
            else:
                ijk2indices[i, j, k] = match

datablocks = np.stack(datablocks)
# converting to 1-index
ijk2indices = ijk2indices + 1

save_output = {'b': datablocks, 'bi': ijk2indices}
sio.savemat('model_normalized.mat', save_output)
print('saved .mat')




