//
// Created by roger on 08/12/2020.
//
#pragma once

#include <nori/color.h>
#include <cuda_runtime.h>
#include <vector_types.h>

float3 make_float3(const nori::Color3f& v)
{
	return make_float3(v[0], v[1], v[2]);
}

float3 make_float3(const nori::Color4f& v)
{
	return make_float3(v[0], v[1], v[2]);
}

float4 make_float4(const nori::Color3f& v, const float alpha)
{
	return make_float4(v[0], v[1], v[2], alpha);
}

float4 make_float4(const nori::Color4f& v)
{
	return make_float4(v[0], v[1], v[2], v[3]);
}

float3 make_float3(const Eigen::Vector3f& v)
{
	return make_float3(v(0), v(1), v(2));
}

float4 make_float4(const Eigen::Vector4f& v)
{
	return make_float4(v(0), v(1), v(2), v(3));
}
