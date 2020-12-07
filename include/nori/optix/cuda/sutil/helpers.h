#pragma once

/**
 * Various helpers for cuda.
 * Based on OptiX samples and NanoVDB
 */

#include <cuda_runtime.h>
#include <optix_device.h>
#include <vector_functions.h>
#include <vector_types.h>
#include <nanovdb/NanoVDB.h>
#include "vec_math.h"

static __forceinline__ __device__ void* unpackPointer(unsigned int i0, unsigned int i1) {
    const unsigned long long uptr = static_cast<unsigned long long>( i0 ) << 32 | i1;
    void* ptr = reinterpret_cast<void*>( uptr );
    return ptr;
}

static __forceinline__ __device__ void packPointer(void* ptr, unsigned int& i0, unsigned int& i1) {
    const unsigned long long uptr = reinterpret_cast<unsigned long long>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

#define float3_as_args(u) \
    reinterpret_cast<uint32_t&>((u).x), \
        reinterpret_cast<uint32_t&>((u).y), \
        reinterpret_cast<uint32_t&>((u).z)

#define array3_as_args(u) \
    reinterpret_cast<uint32_t&>((u)[0]), \
        reinterpret_cast<uint32_t&>((u)[1]), \
        reinterpret_cast<uint32_t&>((u)[2])

__forceinline__ __device__ float3 make_float3(const nanovdb::Vec3f& v)
{
	return make_float3(v[0], v[1], v[2]);
}

__forceinline__ __device__ float3 make_float3(const nanovdb::Vec3R& v)
{
	return make_float3(v[0], v[1], v[2]);
}

// Orthonormal basis
struct Onb
{
    __forceinline__ __device__ Onb(const float3& normal)
    {
        m_normal = normal;

        if(fabs(m_normal.x) > fabs(m_normal.z))
        {
            m_binormal.x = -m_normal.y;
            m_binormal.y = m_normal.x;
            m_binormal.z = 0;
        }
        else
        {
            m_binormal.x = 0;
            m_binormal.y = -m_normal.z;
            m_binormal.z = m_normal.y;
        }

        m_binormal = normalize(m_binormal);
        m_tangent = cross(m_binormal, m_normal);
    }

    __forceinline__ __device__ void inverse_transform(float3& p) const
    {
        p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
    }

    float3 m_tangent, m_binormal, m_normal;
};
