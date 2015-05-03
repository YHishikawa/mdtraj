#include "msvccompat.h"
#include "ssetools.h"
#include "geometryutils.h"

#include <set>
#include <map>
#include <vector>
#include <cmath>

typedef size_t AtomIndex;
typedef std::pair<AtomIndex, AtomIndex> AtomPair;
typedef std::vector<AtomPair>  NeighborList;

class VoxelIndex
{
public:
    VoxelIndex(int xx, int yy, int zz) : x(xx), y(yy), z(zz) {}

    // operator<() needed for map
    bool operator<(const VoxelIndex& other) const {
        if      (x < other.x) return true;
        else if (x > other.x) return false;
        else if (y < other.y) return true;
        else if (y > other.y) return false;
        else if (z < other.z) return true;
        else return false;
    }

    int x;
    int y;
    int z;
};

typedef std::vector< AtomIndex > Voxel;


static INLINE __m128 mult(__m128 x, const __m128 (*h_inv)[3])
{
    __m128 s;
    s = _mm_add_ps(_mm_add_ps(
       _mm_mul_ps((*h_inv)[0], _mm_shuffle_ps(x, x, _MM_SHUFFLE(0,0,0,0))),
       _mm_mul_ps((*h_inv)[1], _mm_shuffle_ps(x, x, _MM_SHUFFLE(1,1,1,1)))),
       _mm_mul_ps((*h_inv)[2], _mm_shuffle_ps(x, x, _MM_SHUFFLE(2,2,2,2))));
   return s;
}

static INLINE float dot3(__m128 x, __m128 y) {
    return _mm_cvtss_f32(_mm_dp_ps2(x, y, 0b01110001));
}




void computeNeighborList(
    NeighborList& neighborList,
    size_t n_atoms,
    float* positions,  // n_atoms x 3
    float box_matrix[9],
    double max_distance,
    bool reportSymmetricPairs = false
) {
    __m128 s;
    __m128 voxelSize;
    __m128 voxelSize_r;
    __m128 h[3];           // unitcell matrix
    __m128 h_inv[3];       // inverse of the unitcell matrix
    float box_lengths[3];  // length of the unitcell vectors
    float edge_lengths[3];
    int numVoxels[3];

    neighborList.clear();
    loadBoxMatrix(box_matrix, &h, &h_inv);
    for (int i = 0; i < 3; i++) {
        box_lengths[i] = sqrt(dot3(h[i], h[i]));
        numVoxels[i] = static_cast<int>(floor(box_lengths[i]/max_distance));
        edge_lengths[i] = box_lengths[i]/numVoxels[i];
    }

    printf("h0 length: %f\n", box_lengths[0]);
    printf("h1 length: %f\n", box_lengths[1]);
    printf("h2 length: %f\n", box_lengths[2]);
    printf("edge0 length: %f\n", edge_lengths[0]);
    printf("edge1 length: %f\n", edge_lengths[1]);
    printf("edge2 length: %f\n", edge_lengths[2]);
    printf("numVoxels 0: %d\n", numVoxels[0]);
    printf("numVoxels 1: %d\n", numVoxels[1]);
    printf("numVoxels 2: %d\n", numVoxels[2]);



    voxelSize = load_float3(edge_lengths);
    voxelSize_r = mult(voxelSize, &h_inv);
    // __m128 ss = mult(voxelSize_r, &h);

    // printf("ss  ");
    // printf_m128(ss);
    printf("box vectors\n  ");
    printf_m128(h[0]); printf("  ");
    printf_m128(h[1]); printf("  ");
    printf_m128(h[2]);

    printf("voxelSize\n  ");
    printf_m128(voxelSize);
    printf("voxelSize_r\n  ");
    printf_m128(voxelSize_r);
    printf("\n\n");

    std::map<VoxelIndex, Voxel> voxelMap;

    for (size_t atom_i = 0; atom_i < n_atoms; atom_i++) {
        // printf("atom index %lu:   ", atom_i);
        s = mult(load_float3(positions + atom_i*3), &h_inv);
        __m128 idx = _mm_floor_ps2(_mm_div_ps(s, voxelSize_r));
        VoxelIndex voxelIndex(idx[0], idx[1], idx[2]);
        if (voxelMap.find(voxelIndex) == voxelMap.end())
            voxelMap[voxelIndex] = Voxel();
        Voxel& voxel = voxelMap.find(voxelIndex)->second;
        voxel.push_back(atom_i);
    }

    printf("voxelMap.size() %lu\n", voxelMap.size());

    for (size_t atom_i = 0; atom_i < n_atoms; atom_i++) {
        printf("atom index %lu:   ", atom_i);
        s = mult(load_float3(positions + atom_i*3), &h_inv);
        __m128 idx = _mm_floor_ps2(_mm_div_ps(s, voxelSize_r));
        VoxelIndex voxelIndex(idx[0], idx[1], idx[2]);
        Voxel& voxel = voxelMap.find(voxelIndex)->second;

        printf("vx: %d %d %d\n", voxelIndex.x, voxelIndex.y, voxelIndex.z);

    }

}


#include <cstdio>

int main() {
    printf("Hello World!\n");

    const int n_atoms = 6;
    const double max_distance = 1.1;
    float positions[n_atoms*3] = {
        0.0, 0.0, -1.0,
        0.0, 0.0, 0.0,
        1.0, 1.0, 1.0,
        1.0, 2.0, 1.0,
        2.0, 2.0, 2.0,
        3.0, 3.0, 3.0,
    };

    float box_matrix[9] = {
        5.0, 0.0, 0.0,
        0.0, 5.0, 0.0,
        0.0, 0.0, 10.0,
    };

    NeighborList neighborList;
    computeNeighborList(neighborList, n_atoms, positions, box_matrix, max_distance);
}
