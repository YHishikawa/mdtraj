import time
import numpy as np
import mdtraj as md
from mdtraj.testing import get_fn
from mdtraj.geometry.neighborlist import NeighborList



def test_0():
    random = np.random.RandomState(0)
    cutoff = 0.1
    positions = random.rand(100, 3).astype(np.float32)
    positions = positions[[1,14], :]
    unitcell = random.rand(3, 3)
    unitcell[1,0] = 0
    unitcell[2,0] = 0
    unitcell[2,1] = 0

    unitcell = np.array([
        [ 0.9065555,   0.77404733,  0.33314515],
        [ 0.,          0.40724117,  0.23223414],
        [ 0.,          0.,          0.72559436]])
    positions = np.array([
        [ 0.54488319,  0.42365479,  0.64589411],
        [ 0.69763118,  0.06022547,  0.6667667 ]],
        dtype=np.float32)


    print(unitcell)
    print(positions)

    neighborlist = NeighborList(cutoff, positions, unitcell)
    pairs1, d1 = neighborlist.get_neighbors(0, naive=True)
    pairs2, d2 = neighborlist.get_neighbors(0, naive=False)

    #pairs1 = sorted({tuple(e) for e in pairs1})
    #pairs2 = sorted({tuple(e) for e in pairs2})

    print("pairs1", pairs1)
    print("pairs2", pairs2)

    #assert pairs1 == pairs2
    #assert set(d1) == set(d2)



def test_1():
    random = np.random.RandomState(0)
    cutoff = 0.1
    positions = random.rand(6, 3).astype(np.float32)
    unitcells = [
        random.rand(3, 3),
        #[[1, 0, 0], [0, 1, 0],     [0, 0, 1]],
        #[[2, 0, 0], [0.5, 1.5, 0], [-0.3, -0.7, 2.2]],
        #[[2, 0, 0], [0.0, 1.5, 0], [0, 0.0, 2.2]],
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
