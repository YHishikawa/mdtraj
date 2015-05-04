import time
import numpy as np
import mdtraj as md
from mdtraj.testing import get_fn
from mdtraj.geometry.neighborlist import NeighborList
random = np.random.RandomState(0)


def test_1():
    cutoff = 0.1
    positions = random.rand(200, 3).astype(np.float32)
    unitcells = [
        [[1, 0, 0], [0, 1, 0],     [0, 0, 1]],
        [[2, 0, 0], [0.5, 1.5, 0], [-0.3, -0.7, 2.2]],
        [[2, 0, 0], [0.0, 1.5, 0], [0, 0.0, 2.2]],
    ]

    for u in unitcells:
        neighborlist = NeighborList(cutoff, positions, u)
        pairs1, d1 = neighborlist.all_neighbors(naive=False)
        pairs2, d2 = neighborlist.all_neighbors(naive=True)

        np.testing.assert_array_almost_equal(
            pairs1[np.argsort(d1)], pairs2[np.argsort(d2)])
        np.testing.assert_array_almost_equal(
            np.sort(d1), np.sort(d2))


def test_2():
    t = md.load(get_fn('1ncw.pdb.gz'))
    neighbors = NeighborList(0.2, t.xyz[0], t.unitcell_vectors[0])

    pairs1, d1 = neighbors.all_neighbors(naive=False)
    pairs2, d2 = neighbors.all_neighbors(naive=True)
    pairs1 = {tuple(e) for e in pairs1}
    pairs2 = {tuple(e) for e in pairs2}

    assert pairs1 == pairs2
    assert set(d1) == set(d2)
