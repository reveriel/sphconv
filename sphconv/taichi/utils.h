#pragma once

#include <taichi/lang.h>
#include "points.h"

using namespace taichi::Tlang;

void init_taichi_program();

Points gen_points_data(int num_points, int num_channel);

inline
Expr declare_global(std::string name, DataType t) {
    auto var_global = Expr(std::make_shared<IdExpression>(name));
    return global_new(var_global, t);
}