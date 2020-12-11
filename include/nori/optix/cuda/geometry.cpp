
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <cuda_runtime.h>
#include <vector_functions.h>
#include <vector_types.h>
#include <optix.h>
#include <optix_types.h>

#include "sutil/helpers.h"
#include "../cuda_shared/LaunchParams.h"
#include "../cuda_shared/GeometryData.h"
#include "../cuda_shared/RayParams.h"

#include "sutil/stdint.h"
#include <nanovdb/util/Ray.h>
#include <nanovdb/util/HDDA.h>

//#define OPTIX_PERF_USE_LEAF_DDA

extern "C" {
__constant__ LaunchParams constantParams;
}

inline __device__ nanovdb::Vec3f
rayBoxIntersect(nanovdb::Vec3f rpos, nanovdb::Vec3f rdir, nanovdb::Vec3f vmin, nanovdb::Vec3f vmax)
{
	float ht[8];
	ht[0] = (vmin[0] - rpos[0]) / rdir[0];
	ht[1] = (vmax[0] - rpos[0]) / rdir[0];
	ht[2] = (vmin[1] - rpos[1]) / rdir[1];
	ht[3] = (vmax[1] - rpos[1]) / rdir[1];
	ht[4] = (vmin[2] - rpos[2]) / rdir[2];
	ht[5] = (vmax[2] - rpos[2]) / rdir[2];
	ht[6] = fmax(fmax(fmin(ht[0], ht[1]), fmin(ht[2], ht[3])), fmin(ht[4], ht[5]));
	ht[7] = fmin(fmin(fmax(ht[0], ht[1]), fmax(ht[2], ht[3])), fmax(ht[4], ht[5]));
	ht[6] = (ht[6] < 0) ? 0.0f : ht[6];
	return nanovdb::Vec3f(ht[6], ht[7], (ht[7] < ht[6] || ht[7] <= 0 || ht[6] <= 0) ? -1.0f : 0.0f);
}

// -----------------------------------------------------------------------------
// FogVolume render method
//
extern "C" __global__ void __intersection__nanovdb_fogvolume()
{
	const auto *sbt_data = reinterpret_cast<const HitGroupParams *>(optixGetSbtDataPointer());
	const auto *grid     = reinterpret_cast<const nanovdb::FloatGrid *>(sbt_data->geometry.volumeNvdb.grid);

	const float3              ray_orig = optixGetWorldRayOrigin();
	const float3              ray_dir  = optixGetWorldRayDirection();
	const nanovdb::Ray<float> wRay(reinterpret_cast<const nanovdb::Vec3f &>(ray_orig),
	                               reinterpret_cast<const nanovdb::Vec3f &>(ray_dir));

	auto iRay = wRay.worldToIndexF(*grid);
	auto bbox = grid->tree().bbox().asReal<float>();
	auto hit  = rayBoxIntersect(iRay.eye(), iRay.dir(), bbox.min(), bbox.max());
	if (hit[2] != -1)
	{
		float voxelUniformSize = float(grid->voxelSize()[0]);
		optixReportIntersection(hit[0] * voxelUniformSize, 0, int(hit[1] * voxelUniformSize));
	}
}
