#pragma once

#include <taichi/lang.h>
#include "points.h"

using namespace taichi::Tlang;

void init_taichi_program();

Points gen_points_data(int num_points, int num_channel);
