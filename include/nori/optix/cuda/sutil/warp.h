//
// Created by roger on 09/12/2020.
//

#pragma once

#include "vec_math.h"

__forceinline__ __device__ float2 squareToUniformDisk(const float2& sample)
{
	float rho   = sqrtf(sample.x);
	float theta = sample.y * 2.0f * M_PIf;
	return make_float2(rho * cos(theta), rho * sin(theta));
}

__forceinline__ __device__ float3 squareToCosineHemisphere(const float2& sample)
{
	// Uniformly sample disk.
	const float r   = sqrtf(sample.x);
	const float phi = 2.0f * M_PIf * sample.y;
	float3      p;

	p.x = r * cosf(phi);
	p.y = r * sinf(phi);

	// Project up to hemisphere.
	p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));

	return p;
}
