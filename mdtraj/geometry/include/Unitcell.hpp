#ifndef MDTRAJ_UNITCELL_H_
#define MDTRAJ_UNITCELL_H_

#include "vectorize.hpp"
#include "msvccompat.h"
#include <string.h>


class Unitcell {
public:
    Unitcell(const float h[9]);
    fvec4 to_recip(const fvec4& r) const;
    void to_recip(const fvec4& x, const fvec4& y, const fvec4& z, fvec4& x_r, fvec4& y_r, fvec4& z_r) const;
    fvec4 from_recip(const fvec4& s) const;
    void from_recip(const fvec4& x_r, const fvec4& y_r, const fvec4& z_r, fvec4& x, fvec4& y, fvec4& z) const;
    fvec4 minimum_image(const fvec4& r) const;

private:
    float h_[9];
    float h_inv_[9];
    fvec4 h_cols_[3];      // *columns* of the box matrix
    fvec4 h_inv_cols_[3];  // *columns* of the inverse of the box matrix
};

Unitcell::Unitcell(const float h[9]) {
    memcpy(h_, h, 9*sizeof(float));
    double det = h[0] * (h[4] * h[8] - h[5] * h[7])
               + h[3] * (h[7] * h[2] - h[8] * h[1])
               + h[6] * (h[1] * h[5] - h[2] * h[4]);
    fvec4 inverse_det(1.0 / det);

    h_inv_[0] = (1.0 / det) * (h[4]*h[8] - h[7]*h[5]);
    h_inv_[1] = (1.0 / det) * -(h[1]*h[8] - h[2]*h[7]);
    h_inv_[2] = (1.0 / det) * (h[1]*h[5] - h[2]*h[4]);
    h_inv_[3] = (1.0 / det) * -(h[3]*h[8] - h[5]*h[6]);
    h_inv_[4] = (1.0 / det) * (h[0]*h[8] - h[2]*h[6]);
    h_inv_[5] = (1.0 / det) * -(h[0]*h[5] - h[3]*h[2]);
    h_inv_[6] = (1.0 / det) * (h[3]*h[7] - h[6]*h[4]);
    h_inv_[7] = (1.0 / det) * -(h[0]*h[7] - h[6]*h[1]);
    h_inv_[8] = (1.0 / det) * (h[0]*h[4] - h[3]*h[1]);

    h_cols_[0]     = fvec4(h[0], h[3], h[6], 0.0f);
    h_cols_[1]     = fvec4(h[1], h[4], h[7], 0.0f);
    h_cols_[2]     = fvec4(h[2], h[5], h[8], 0.0f);
    h_inv_cols_[0] = fvec4(h_inv_[0], h_inv_[3], h_inv_[6], 0.0f);
    h_inv_cols_[1] = fvec4(h_inv_[1], h_inv_[4], h_inv_[7], 0.0f);
    h_inv_cols_[2] = fvec4(h_inv_[2], h_inv_[5], h_inv_[8], 0.0f);
}

fvec4 Unitcell::to_recip(const fvec4& r) const {
    return h_inv_cols_[0] * fvec4(_mm_shuffle_ps(r.val, r.val, _MM_SHUFFLE(0,0,0,0))) +
           h_inv_cols_[1] * fvec4(_mm_shuffle_ps(r.val, r.val, _MM_SHUFFLE(1,1,1,1))) +
           h_inv_cols_[2] * fvec4(_mm_shuffle_ps(r.val, r.val, _MM_SHUFFLE(2,2,2,2)));
}

void  Unitcell::to_recip(const fvec4& x, const fvec4& y, const fvec4& z,
                         fvec4& x_r, fvec4& y_r, fvec4& z_r) const {
    x_r = h_inv_[0] * x + h_inv_[1] * y + h_inv_[2] * z;
    y_r = h_inv_[3] * x + h_inv_[4] * y + h_inv_[5] * z;
    z_r = h_inv_[6] * x + h_inv_[7] * y + h_inv_[8] * z;
 }


fvec4 Unitcell::from_recip(const fvec4& s) const {
    return h_cols_[0] * fvec4(_mm_shuffle_ps(s.val, s.val, _MM_SHUFFLE(0,0,0,0))) +
           h_cols_[1] * fvec4(_mm_shuffle_ps(s.val, s.val, _MM_SHUFFLE(1,1,1,1))) +
           h_cols_[2] * fvec4(_mm_shuffle_ps(s.val, s.val, _MM_SHUFFLE(2,2,2,2)));
}

void Unitcell::from_recip(const fvec4& x_r, const fvec4& y_r, const fvec4& z_r,
                          fvec4& x, fvec4& y, fvec4& z) const{
    x = h_[0] * x_r + h_[1] * y_r + h_[2] * z_r;
    y = h_[3] * x_r + h_[4] * y_r + h_[5] * z_r;
    z = h_[6] * x_r + h_[7] * y_r + h_[8] * z_r;
}

fvec4 Unitcell::minimum_image(const fvec4& r) const {
    fvec4 s = to_recip(r);
    s = s - round(s);
    return from_recip(s);
}

#endif
