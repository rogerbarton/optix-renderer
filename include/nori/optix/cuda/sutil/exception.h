//
// Created by roger on 09/12/2020.
//

#pragma once

#define ERROR_COLOR make_float3(1.f,0.f,0.8f)
#ifndef NORI_OPTIX_RELEASE

#define IF_PIXEL( x_, y_ )                                                     \
    const uint3 launch_idx__ = optixGetLaunchIndex();                          \
    if( launch_idx__.x == (x_) && launch_idx__.y == (y_) )                     \

#define PRINT_PIXEL( x_, y_, ... )                                             \
do                                                                             \
{                                                                              \
    const uint3 launch_idx = optixGetLaunchIndex();                            \
    if( launch_idx.x == (x_) && launch_idx.y == (y_) )                         \
    {                                                                          \
         printf( __VA_ARGS__ );                                                \
    }                                                                          \
} while(0);
#else
#define IF_PIXEL( x_, y_ )
#define PRINT_PIXEL( x_, y_, ... )
#endif