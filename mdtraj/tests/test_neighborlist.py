import numpy as np
from mdtraj.geometry.neighborlist import NeighborList
random = np.random.RandomState(0)

def test_1():
    unitcell = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    positions = random.rand(20, 3).astype(np.float32)
    cutoff = 0.2

    neighborlist = NeighborList(cutoff, positions, unitcell)

    pairs1, d1 = neighborlist.get_neighbors(0, naive=False)
    pairs2, d2 = neighborlist.get_neighbors(0, naive=True)
    print(pairs1)
    print(pairs2)

    print(d1)
    print(d2)
    print(d1==d2)

