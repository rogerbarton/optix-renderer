//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#pragma once

#include <cuda_runtime.h>
#include <optix_device.h>

#include "vec_math.h"
#include "../../cuda_shared/GeometryData.h"

struct LocalGeometry
{
	float3 p;
	float3 n;
	float3 ng;   // global normal
	float2 uv;
	float3 dndu;
	float3 dndv;
	float3 dpdu; // tangent
	float3 dpdv; // bitangent
};


__host__ __device__ __forceinline__ LocalGeometry
getLocalGeometry(const GeometryData &geometryData)
{
	LocalGeometry lgeom{};
	switch (geometryData.type)
	{
		case GeometryData::TRIANGLE_MESH:
		{
			const GeometryData::TriangleMesh &meshData = geometryData.triangleMesh;

			const unsigned int prim_idx = optixGetPrimitiveIndex();
			const float2       barys    = optixGetTriangleBarycentrics();

			uint3 tri = meshData.F[prim_idx];

			const float3 p0 = meshData.V[tri.x];
			const float3 p1 = meshData.V[tri.y];
			const float3 p2 = meshData.V[tri.z];
			lgeom.p = (1.0f - barys.x - barys.y) * p0 + barys.x * p1 + barys.y * p2;
			lgeom.p = optixTransformPointFromObjectToWorldSpace(lgeom.p);

			float2 uv0, uv1, uv2;
			if (meshData.UV)
			{
				uv0 = meshData.UV[tri.x];
				uv1 = meshData.UV[tri.y];
				uv2 = meshData.UV[tri.z];
				lgeom.uv = (1.0f - barys.x - barys.y) * uv0 + barys.x * uv1 + barys.y * uv2;
			}
			else
			{
				uv0 = make_float2(0.0f, 0.0f);
				uv1 = make_float2(0.0f, 1.0f);
				uv2 = make_float2(1.0f, 0.0f);
				lgeom.uv = barys;
			}

			lgeom.ng = normalize(cross(p1 - p0, p2 - p0));
			lgeom.ng = optixTransformNormalFromObjectToWorldSpace(lgeom.ng);

			float3 n0, n1, n2;
			if (meshData.N)
			{
				n0 = meshData.N[tri.x];
				n1 = meshData.N[tri.y];
				n2 = meshData.N[tri.z];
				lgeom.n = (1.0f - barys.x - barys.y) * n0 + barys.x * n1 + barys.y * n2;
				lgeom.n = normalize(optixTransformNormalFromObjectToWorldSpace(lgeom.n));
			}
			else
			{
				lgeom.n = n0 = n1 = n2 = lgeom.ng;
			}

			const float du1 = uv0.x - uv2.x;
			const float du2 = uv1.x - uv2.x;
			const float dv1 = uv0.y - uv2.y;
			const float dv2 = uv1.y - uv2.y;

			const float3 dp1 = p0 - p2;
			const float3 dp2 = p1 - p2;

			const float3 dn1 = n0 - n2;
			const float3 dn2 = n1 - n2;

			const float det = du1 * dv2 - dv1 * du2;

			const float invdet = 1.f / det;
			lgeom.dpdu = (dv2 * dp1 - dv1 * dp2) * invdet;
			lgeom.dpdv = (-du2 * dp1 + du1 * dp2) * invdet;
			lgeom.dndu = (dv2 * dn1 - dv1 * dn2) * invdet;
			lgeom.dndv = (-du2 * dn1 + du1 * dn2) * invdet;

			break;
		}
		case GeometryData::SPHERE:
		{
			const float hitTime = int_as_float(optixGetAttribute_0());
			lgeom.p = optixGetWorldRayOrigin() + hitTime * optixGetWorldRayDirection();
			lgeom.n = make_float3(
					int_as_float(optixGetAttribute_1()),
					int_as_float(optixGetAttribute_2()),
					int_as_float(optixGetAttribute_3()));
			lgeom.ng = lgeom.n; // No transformations on spheres currently

			lgeom.uv = make_float2(
					-atan2f(lgeom.n.y, lgeom.n.x) / M_PIf / 2,
					acosf(lgeom.n.z)) / M_PIf;

			lgeom.dpdu = cross(make_float3(0, 0, 1), -lgeom.n);
			lgeom.dpdv = cross(lgeom.n, lgeom.dpdu);

			// Note: dndu and dndv not set

			break;
		}
		default:
			break;
	}


	return lgeom;
}


