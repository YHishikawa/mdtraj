#ifndef MDTRAJ_NEIGHBORLIST_H_
#define MDTRAJ_NEIGHBORLIST_H_

#include "vectorize.hpp"
#include "Unitcell.hpp"
#include <vector>
#include <map>
#include <set>
#include <cmath>
#include <cstdlib>
#include "util.h"

static INLINE int imod(int i, int n) {
    return (i > 0 ? i : i + n) % n;
}


class AtomPair {
public:
    AtomPair(size_t i_, size_t j_, float d2_) : i(i_), j(j_), d2(d2_) {};
    const size_t i;
    const size_t j;
    const float d2;
};

class NeighborList {
public:
    NeighborList(float max_distance, size_t n_atoms, const float* positions, const float unitcell_vectors[9]);
    ~NeighborList();
    void getNeighbors(size_t i, std::vector<AtomPair>& neighbors) const;


private:
    void loadPositions(const float* positions);
    size_t getVoxelIndex(const fvec4& s) const;
    size_t getVoxelIndex(int nx, int ny, int nz) const;
    ivec4 getVoxelIndexVector(const fvec4& s) const;
    void addLocation(size_t i);


    const float max_distance_;
    const float max_distance2_;
    const size_t n_atoms_;
    float* positions_r_;
    const Unitcell unitcell_;
    fvec4 unitcell_vectors_[3];
    fvec4 unitcell_lengths_;
    // fvec4 n_voxels_;
    ivec4 n_voxels_;
    fvec4 voxel_size_;
    fvec4 voxel_size_r_;
    ivec4 vox_intex_mult_;
    std::map<size_t, std::vector<size_t> > voxelMap_;

};

/* ------------------ */


NeighborList::NeighborList(float max_distance, size_t n_atoms, const float* positions, const float unitcell_vectors[9])
        : max_distance_(max_distance)
        , max_distance2_(max_distance*max_distance)
        , n_atoms_(n_atoms)
        , unitcell_(unitcell_vectors)
{
    unitcell_vectors_[0] = load3(&unitcell_vectors[0]);
    unitcell_vectors_[1] = load3(&unitcell_vectors[3]);
    unitcell_vectors_[2] = load3(&unitcell_vectors[6]);
    unitcell_lengths_ = fvec4(
        sqrt(dot3(unitcell_vectors_[0], unitcell_vectors_[0])),
        sqrt(dot3(unitcell_vectors_[1], unitcell_vectors_[1])),
        sqrt(dot3(unitcell_vectors_[2], unitcell_vectors_[2])),
        0.0);

    n_voxels_ = floor(unitcell_lengths_ / max_distance);
    voxel_size_ = unitcell_lengths_ / n_voxels_;
    voxel_size_r_ = unitcell_.to_recip(voxel_size_);

    int n[4];
    n_voxels_.store(n);
    int vox_intex_mult[4] = {n[0]*n[1]*n[2], n[1]*n[2], n[2], 0};
    vox_intex_mult_ = ivec4(vox_intex_mult);

    positions_r_ = static_cast<float*>(calloc(n_atoms * 4, sizeof(float)));
    loadPositions(positions);
    for (size_t i = 0; i < n_atoms_; i++)
        addLocation(i);
}



void NeighborList::loadPositions(const float* positions) {
    fvec4 x, y, z;
    fvec4 x_r, y_r, z_r;

    const float* positions_ptr = positions;
    float* positions_r_ptr = positions_r_;

    size_t i = 0;
    for(; i < (n_atoms_ >> 2); i += 4) {
        aos_deinterleaved_loadu(positions_ptr, &x.val, &y.val, &z.val);
        unitcell_.to_recip(x, y, z, x_r, y_r, z_r);

        fvec4 w_r(0.0f);
        transpose(x_r, y_r, z_r, w_r);
        x_r.store(positions_r_ptr);
        y_r.store(positions_r_ptr + 4);
        z_r.store(positions_r_ptr + 8);
        w_r.store(positions_r_ptr + 12);

        positions_ptr += 12;
        positions_r_ptr += 16;
    }
    for(; i < n_atoms_; i++) {
        fvec4 r = load3(positions_ptr);
        fvec4 s = unitcell_.to_recip(r);
        s.store(positions_r_ptr);
        positions_ptr += 3;
        positions_r_ptr += 4;
    }

    // for (i = 0; i < n_atoms_; i++) {
    //     printf("%.3f %.3f %.3f    ", positions_r_[i*4], positions_r_[i*4+1], positions_r_[i*4+2]);
    //     fvec4 s(positions_r_ + 4*i);
    //     s.print();
    // }
}


NeighborList::~NeighborList() {
    free(static_cast<void*>(positions_r_));
}

void NeighborList::addLocation(size_t i) {
    fvec4 s(positions_r_ + 4*i);
    size_t idx = getVoxelIndex(s);
    if (voxelMap_.find(idx) == voxelMap_.end())
        voxelMap_[idx] = std::vector<size_t>();

    voxelMap_[idx].push_back(i);
}

ivec4 NeighborList::getVoxelIndexVector(const fvec4& s) const {
    fvec4 f = floor(s / voxel_size_r_);
    f = fmod(f + (f < fvec4(0.0f)&n_voxels_), n_voxels_);
    return f;
}

size_t NeighborList::getVoxelIndex(const fvec4& s) const {
    ivec4 voxelIndexVector = getVoxelIndexVector(s);
    return sum(voxelIndexVector * vox_intex_mult_);
}

size_t NeighborList::getVoxelIndex(int nx, int ny, int nz) const {
    ivec4 voxelIndexVector(nx, ny, nz, 0);
    return sum(voxelIndexVector * vox_intex_mult_);
}

void NeighborList::getNeighbors(size_t i, std::vector<AtomPair>& neighbors) const {
    fvec4 s(positions_r_ + 4*i);
    int center_idx[4];
    getVoxelIndexVector(s).store(center_idx);


    int d_idx[4];
    int n_vox[4];
    ivec4 d_index = floor(max_distance_ / voxel_size_) + 1;
    d_index.store(d_idx);
    n_voxels_.store(n_vox);

    int minx = center_idx[0] - d_idx[0];
    int maxx = center_idx[0] + d_idx[0];
    int miny = center_idx[1] - d_idx[1];
    int maxy = center_idx[1] + d_idx[1];
    int minz = center_idx[2] - d_idx[2];
    int maxz = center_idx[2] + d_idx[2];
    std::set<size_t> visitedVoxels;

    for (int ix = minx; ix <= maxx; ix++) {
        int x = imod(ix, n_vox[0]);

        for (int iy = miny; iy <= maxy; iy++) {
            int y = imod(iy, n_vox[1]);

            for (int iz = minz; iz <= maxz; iz++) {
                int z = imod(iz, n_vox[2]);

                size_t voxelIndex = getVoxelIndex(x, y, z);
                if (visitedVoxels.find(voxelIndex) != visitedVoxels.end())
                    continue;
                visitedVoxels.insert(voxelIndex);
                // printf("  x=%d, y=%d, z=%d\n", x, y, z);

                const std::map<size_t, std::vector<size_t> >::const_iterator voxelEntry = voxelMap_.find(voxelIndex);
                if (voxelEntry == voxelMap_.end()) continue; // no such voxel; skip
                const std::vector<size_t>& voxel = voxelEntry->second;

                for (size_t j = 0; j < voxel.size(); j++) {
                    size_t atom_j = voxel[j];
                    fvec4 sj(positions_r_ + 4*atom_j);
                    fvec4 s12 = s - sj;
                    fvec4 r12 = unitcell_.from_recip(s12 - round(s12));
                    float d2 = dot3(r12, r12);
                    if ((d2 > 0) && (d2 < max_distance2_)) {
                        AtomPair pair(i, atom_j, d2);
                        neighbors.push_back(pair);
                    }
                }

            }
        }
    }
    // return
}

#endif
