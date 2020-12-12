#pragma once

/**
 * Various helpers for cuda.
 * Based on OptiX samples and NanoVDB
 */

#include <cuda_runtime.h>
#include <optix_device.h>
#include <vector_functions.h>
#include <vector_types.h>
#include "stdint.h"
#include <nanovdb/NanoVDB.h>
#include "vec_math.h"

#define EPSILON 1e-4f
#define INFINITY 1e10f

static __forceinline__ __device__ void *unpackPointer(unsigned int i0, unsigned int i1)
{
	const unsigned long long uptr = static_cast<unsigned long long>( i0 ) << 32 | i1;
	void                     *ptr = reinterpret_cast<void *>( uptr );
	return ptr;
}

static __forceinline__ __device__ void packPointer(void *ptr, unsigned int &i0, unsigned int &i1)
{
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

__forceinline__ __device__ float3 make_float3(const nanovdb::Vec3f &v)
{
	return make_float3(v[0], v[1], v[2]);
}

__forceinline__ __device__ float3 make_float3(const nanovdb::Vec3R &v)
{
	return make_float3(v[0], v[1], v[2]);
}

// Orthonormal basis
struct Frame
{
	__forceinline__ __device__ Frame(const float3 &normal) : m_normal(normal)
	{
		if (fabs(m_normal.x) > fabs(m_normal.z))
		{
			m_bitangent.x = -m_normal.y;
			m_bitangent.y = m_normal.x;
			m_bitangent.z = 0;
		}
		else
		{
			m_bitangent.x = 0;
			m_bitangent.y = -m_normal.z;
			m_bitangent.z = m_normal.y;
		}

		m_bitangent = normalize(m_bitangent);
		m_tangent   = cross(m_bitangent, m_normal);
	}

	__forceinline__ __device__ Frame(const float3 &tangent, const float3 &bitangent, const float3 &normal) :
			m_tangent(tangent), m_bitangent(bitangent), m_normal(normal) {}

	__forceinline__ __device__ float3 toLocal(const float3 &v) const
	{
		return make_float3(dot(v, m_tangent), dot(v, m_bitangent), dot(v, m_normal));
	}

	__forceinline__ __device__ float3 toWorld(const float3 &p) const
	{
		return p.x * m_tangent + p.y * m_bitangent + p.z * m_normal;
	}

	float3 m_tangent, m_bitangent, m_normal;
};

/**
 * Normalizes the value and writes it into the output buffer with the correct weighting
 */
__forceinline__ void
interpolateAndApplyToBuffer(const uint32_t &sampleIndex, const uint32_t &samplesPerLaunch, const uint32_t &pixelIdx, float4 *const outputBuffer,
                            float3 value)
{
	value /= static_cast<float>(samplesPerLaunch);

	if (sampleIndex > 1)
	{
		const float3 prevPixel = make_float3(outputBuffer[pixelIdx]);

		const float a = samplesPerLaunch / static_cast<float>(sampleIndex + samplesPerLaunch);
		value = lerp(prevPixel, value, a);
	}

	outputBuffer[pixelIdx] = make_float4(value, 1.f);
}
