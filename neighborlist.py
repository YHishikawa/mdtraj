from __future__ import print_function, division, absolute_import
import numpy as np
from collections import defaultdict

class NeighborListBuilder(object):
    def __init__(self, unitcell_vectors, max_distance):
        unitcell_vectors = np.asarray(unitcell_vectors, dtype=np.float64)
        max_distance = float(max_distance)
        if not unitcell_vectors.shape == (3, 3):
            raise ValueError('unitcell_vectors must be of shape (3,3)')

        unitcell_lengths = np.sum(unitcell_vectors**2, axis=-1)**0.5
        unitcell_inverse = np.linalg.inv(unitcell_vectors)
        n_voxels = np.floor(unitcell_lengths / max_distance)
        edge_lengths = unitcell_lengths / n_voxels
        voxel_size_r = np.dot(unitcell_inverse, edge_lengths)

        self.voxels = defaultdict(lambda: [])
        self.unitcell_vectors = unitcell_vectors  # (3,3)
        self.unitcell_inverse = unitcell_inverse  # (3,3)
        self.unitcell_lengths = unitcell_lengths  # (3,)
        self.max_distance = max_distance          # float
        self.n_voxels = n_voxels                  # (3,)
        self.edge_lengths = edge_lengths          # (3,)
        self.voxel_size_r = voxel_size_r          # (3,)
        print('n voxels', self.n_voxels)

    def add_location(self, i, x):
        idx = self.get_index(x)
        self.voxels[idx].append((i, x))

    def get_index(self, x):
        s = np.dot(self.unitcell_inverse, x)
        idx = tuple(np.mod(np.floor(s / self.voxel_size_r), self.n_voxels).astype(np.int))
        return idx

    def get_neighbors(self, x):
        print('x.shape', x)
        center_idx = self.get_index(x)
        d_index = (self.max_distance / self.edge_lengths).astype(np.int) + 1
        print('center_idx', center_idx)
        print('d_index', d_index)
        neighbors = []


        for ix in range(center_idx[0] - d_index[0], center_idx[0] + d_index[0] + 1):
            ix = np.mod(ix, self.n_voxels[0])
            for iy in range(center_idx[1] - d_index[1], center_idx[1] + d_index[1] + 1):
                iy = np.mod(iy, self.n_voxels[1])
                for iz in range(center_idx[2] - d_index[2], center_idx[2] + d_index[2] + 1):
                    iz = np.mod(iz, self.n_voxels[2])
                    # print('x=%d, y=%d, z=%d' % (ix, iy, iz))

                    indx = (int(ix), int(iy), int(iz))
                    if indx in self.voxels:
                        # print("HELLO")
                        voxel_j = self.voxels[indx]
                        for atom_j in voxel_j:
                            d = self.get_distance(x, atom_j[1])
                            if 0 < d < self.max_distance:

                                neighbors.append((atom_j[0], d**2))
        return neighbors

    def get_neighbors_naive(self, x):
        for y in (atom[1] for voxel in self.voxels.values() for atom in voxel):
            d = self.get_distance(x, y)
            if 0 < d < self.max_distance:
                print('neoghbors', x, y)

    def get_distance(self, x, y):
        r = x-y
        s = np.dot(self.unitcell_inverse, r)
        s -= np.round(s)
        r = np.dot(self.unitcell_vectors, s)
        return np.sum(r**2)**0.5


def main():
    unitcell = [[5.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 8.0]]
    positions = np.array([
        [0.0, 0.0, -1.0,],
        [0.0, 0.0, 0.0,],
        [1.0, 1.0, 1.0,],
        [1.0, 2.0, 1.0,],
        [2.0, 2.0, 2.0,],
        [3.0, 3.0, 3.0,],
        [4.9, 4.9, 9.9],
    ])

    builder = NeighborListBuilder(unitcell, 2.5)
    for i, x in enumerate(positions):
        builder.add_location(i, x)


    # print('s', np.dot(builder.unitcell_inverse, positions.T).T)
    print(builder.get_neighbors(positions[0]))

    # for i, x in enumerate(positions):
#         builder.get_neighbors(x)
#
#     print()
#     for i, x in enumerate(positions):
#         builder.get_neighbors_naive(x)
    # print(builder.voxels)
    # print((1,2,3) in builder.voxels)

if __name__ == '__main__':
    main()
