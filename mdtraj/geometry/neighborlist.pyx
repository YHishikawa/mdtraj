import numpy as np
from libc.stdint cimport int64_t
from libcpp.vector cimport vector

cdef extern from "math.h":
    float sqrtf(float x) nogil

cdef extern from "NeighborList.hpp" namespace "MDTraj":
    cdef cppclass AtomPair:
        const size_t i, j
        const float d2

    cdef cppclass CNeighborList "MDTraj::NeighborList":
        CNeighborList(float, size_t, float*, float*) nogil except +
        void getNeighbors(size_t, vector[AtomPair]&) nogil
        void getNeighborsNaive(size_t, vector[AtomPair]&) nogil


cdef class NeighborList:
    cdef CNeighborList *thisptr
    cdef float[:, ::1] positions

    def __cinit__(self, float max_distance, float[:, ::1] positions, unitcell_vectors):
        cdef size_t n_atoms = positions.shape[0]
        cdef size_t n_dims = positions.shape[1]
        cdef float[:, ::1] cell_vectors = np.ascontiguousarray(unitcell_vectors, dtype=np.float32)
        if n_dims != 3:
            raise ValueError('positions must have shape (n_atoms, 3)')
        if not (cell_vectors.shape[0] == cell_vectors.shape[0] == 3):
            raise ValueError('unitcell vectors must hav shape (3, 3)')

        self.positions = positions  # incremenet the refcount
        with nogil:
            self.thisptr = new CNeighborList(max_distance, n_atoms, &positions[0,0], &cell_vectors[0,0])

    def __dealloc__(self):
        del self.thisptr

    def get_neighbors(self, size_t i, naive=False):
        cdef size_t j, size
        cdef vector[AtomPair] neighbors

        if naive:
            with nogil:
                self.thisptr.getNeighborsNaive(i, neighbors)
        else:
            with nogil:
                 self.thisptr.getNeighbors(i, neighbors)

        size = neighbors.size()
        cdef int64_t[:, ::1] pairs = np.zeros((size, 2), dtype=np.int64)
        cdef float[::1] distance = np.zeros(size, dtype=np.float32)
        for j in range(size):
            pairs[j, 0] = neighbors[j].i
            pairs[j, 1] = neighbors[j].j
            distance[j] = sqrtf(neighbors[j].d2)

        sort_index = np.argsort(distance)
        return np.asarray(pairs)[sort_index], np.asarray(distance)[sort_index]

