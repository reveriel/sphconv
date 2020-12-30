#include "copy.h"
#include "conv.hpp"

using namespace taichi::Tlang;



/**
* init layer0 from points data,  voxelization is done at here
*/
void init_layer0(Expr &layer0, torch::Tensor &points, const VoxelizationConfig &vcfg)
{
    // CPU accessor
    auto points_a = points.accessor<float, 2>();

    for (int i = 0; i < points_a.size(0); i++) {
        float x = points_a[i][0];
        float y = points_a[i][1];
        float z = points_a[i][2];
        float refl = points_a[i][3];

        float x2y2 = x * x + y * y;
        float r = std::sqrt(x2y2 + z * z);
        float theta = std::acos(z / r);
        float phi = std::asin(y / std::sqrt(x2y2));

        int theta_idx = vcfg.theta_idx(theta);
        int phi_idx = vcfg.phi_idx(phi);
        int depth_idx = vcfg.depth_idx(r);

        if (in_range(theta_idx, 0, vcfg.v_res)
            && in_range(phi_idx, 0, vcfg.h_res)
            && in_range(depth_idx, 0, vcfg.d_res))
        {
            layer0.val<taichi::float32>(theta_idx, phi_idx, depth_idx, 0) = x;
            layer0.val<taichi::float32>(theta_idx, phi_idx, depth_idx, 1) = y;
            layer0.val<taichi::float32>(theta_idx, phi_idx, depth_idx, 2) = z;
            layer0.val<taichi::float32>(theta_idx, phi_idx, depth_idx, 3) = refl;
            // for (int j = 0; j < num_ch1; j++) {
            //     layer0.val<taichi::float32>(theta_idx, phi_idx, depth_idx, j) = refl;
            // }
        }
    }
}