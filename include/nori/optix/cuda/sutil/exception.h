//
// Created by roger on 09/12/2020.
//

#pragma once

#define ERROR_COLOR make_float3(255,192,203)

#define IF_PIXEL( x_, y_ )                                                     \
    const uint3 launch_idx__ = optixGetLaunchIndex();                          \
    if( launch_idx__.x == (x_) && launch_idx__.y == (y_) )                     \

#define PRINT_PIXEL( x_, y_, str, ... )                                        \
do                                                                             \
{                                                                              \
    const uint3 launch_idx = optixGetLaunchIndex();                            \
    if( launch_idx.x == (x_) && launch_idx.y == (y_) )                         \
    {                                                                          \
         printf( str, __VA_ARGS__  );                                          \
    }                                                                          \
} while(0);