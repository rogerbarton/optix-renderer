/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob, Romain Prévost

    Nori is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Nori is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <nori/mesh.h>
#include <nori/bbox.h>
#include <nori/bsdf.h>
#include <nori/emitter.h>
#include <nori/warp.h>
#include <Eigen/Geometry>

NORI_NAMESPACE_BEGIN

void Mesh::update(const NoriObject *guiObject)
{
    const auto *gui = static_cast<const Mesh *>(guiObject);
    if (gui->geometryTouched)
    {
        m_pdf = DiscretePDF();
        m_pdf.reserve(getPrimitiveCount());
        for (uint32_t i = 0; i < getPrimitiveCount(); ++i)
            m_pdf.append(surfaceArea(i));
        m_pdf.normalize();
    }

    Shape::update(guiObject);
}

void Mesh::sampleSurface(ShapeQueryRecord &sRec, const Point2f &sample) const
{
    Point2f s = sample;
    size_t idT = m_pdf.sampleReuse(s.x());

    Vector3f bc = Warp::squareToUniformTriangle(s);

    sRec.p = getInterpolatedVertex(static_cast<uint32_t>(idT), bc);
    if (m_N.size() > 0)
    {
        sRec.n = getInterpolatedNormal(static_cast<uint32_t>(idT), bc);
    }
    else
    {
        Point3f p0 = m_V.col(m_F(0, idT));
        Point3f p1 = m_V.col(m_F(1, idT));
        Point3f p2 = m_V.col(m_F(2, idT));
        Normal3f n = (p1 - p0).cross(p2 - p0).normalized();
        sRec.n = n;
    }
    sRec.pdf = m_pdf.getNormalization();
}
float Mesh::pdfSurface(const ShapeQueryRecord &sRec) const
{
    return m_pdf.getNormalization();
}

Point3f Mesh::getInterpolatedVertex(uint32_t index, const Vector3f &bc) const
{
    return (bc.x() * m_V.col(m_F(0, index)) +
            bc.y() * m_V.col(m_F(1, index)) +
            bc.z() * m_V.col(m_F(2, index)));
}

Normal3f Mesh::getInterpolatedNormal(uint32_t index, const Vector3f &bc) const
{
    return (bc.x() * m_N.col(m_F(0, index)) +
            bc.y() * m_N.col(m_F(1, index)) +
            bc.z() * m_N.col(m_F(2, index)))
        .normalized();
}

float Mesh::surfaceArea(uint32_t index) const
{
    uint32_t i0 = m_F(0, index), i1 = m_F(1, index), i2 = m_F(2, index);

    const Point3f p0 = m_V.col(i0), p1 = m_V.col(i1), p2 = m_V.col(i2);

    return 0.5f * Vector3f((p1 - p0).cross(p2 - p0)).norm();
}

bool Mesh::rayIntersect(uint32_t index, const Ray3f &ray, float &u, float &v, float &t) const
{
    uint32_t i0 = m_F(0, index), i1 = m_F(1, index), i2 = m_F(2, index);
    const Point3f p0 = m_V.col(i0), p1 = m_V.col(i1), p2 = m_V.col(i2);

    /* Find vectors for two edges sharing v[0] */
    Vector3f edge1 = p1 - p0, edge2 = p2 - p0;

    /* Begin calculating determinant - also used to calculate U parameter */
    Vector3f pvec = ray.d.cross(edge2);

    /* If determinant is near zero, ray lies in plane of triangle */
    float det = edge1.dot(pvec);

    if (det > -1e-8f && det < 1e-8f)
        return false;
    float inv_det = 1.0f / det;

    /* Calculate distance from v[0] to ray origin */
    Vector3f tvec = ray.o - p0;

    /* Calculate U parameter and test bounds */
    u = tvec.dot(pvec) * inv_det;
    if (u < 0.0 || u > 1.0)
        return false;

    /* Prepare to test V parameter */
    Vector3f qvec = tvec.cross(edge1);

    /* Calculate V parameter and test bounds */
    v = ray.d.dot(qvec) * inv_det;
    if (v < 0.0 || u + v > 1.0)
        return false;

    /* Ray intersects triangle -> compute t */
    t = edge2.dot(qvec) * inv_det;

    return t >= ray.mint && t <= ray.maxt;
}

void Mesh::setHitInformation(uint32_t index, const Ray3f &ray, Intersection &its) const
{
    /* Find the barycentric coordinates */
    Vector3f bary;
    bary << 1 - its.uv.sum(), its.uv;

    /* Vertex indices of the triangle */
    uint32_t idx0 = m_F(0, index), idx1 = m_F(1, index), idx2 = m_F(2, index);

    Point3f p0 = m_V.col(idx0), p1 = m_V.col(idx1), p2 = m_V.col(idx2);

    /* Compute the intersection positon accurately
       using barycentric coordinates */
    its.p = bary.x() * p0 + bary.y() * p1 + bary.z() * p2;

    /* Compute proper texture coordinates if provided by the mesh */
    if (m_UV.size() > 0)
        its.uv = bary.x() * m_UV.col(idx0) +
                 bary.y() * m_UV.col(idx1) +
                 bary.z() * m_UV.col(idx2);

    /* Compute the geometry frame */
    its.geoFrame = Frame((p1 - p0).cross(p2 - p0).normalized());

    if (m_N.size() > 0)
    {
        /* Compute the shading frame. Note that for simplicity,
           the current implementation doesn't attempt to provide
           tangents that are continuous across the surface. That
           means that this code will need to be modified to be able
           use anisotropic BRDFs, which need tangent continuity */

        its.shFrame = Frame(
            (bary.x() * m_N.col(idx0) +
             bary.y() * m_N.col(idx1) +
             bary.z() * m_N.col(idx2))
                .normalized());
    }
    else
    {
        its.shFrame = its.geoFrame;
    }
}

BoundingBox3f Mesh::getBoundingBox(uint32_t index) const
{
    BoundingBox3f result(m_V.col(m_F(0, index)));
    result.expandBy(m_V.col(m_F(1, index)));
    result.expandBy(m_V.col(m_F(2, index)));
    return result;
}

Point3f Mesh::getCentroid(uint32_t index) const
{
    return (1.0f / 3.0f) *
           (m_V.col(m_F(0, index)) +
            m_V.col(m_F(1, index)) +
            m_V.col(m_F(2, index)));
}

std::string Mesh::toString() const
{
    return tfm::format(
        "Mesh[\n"
        "  name = \"%s\",\n"
        "  vertexCount = %i,\n"
        "  triangleCount = %i,\n"
        "  bsdf = %s,\n"
        "  emitter = %s\n"
        "]",
        m_name,
        m_V.cols(),
        m_F.cols(),
        m_bsdf ? indent(m_bsdf->toString()) : std::string("null"),
        m_emitter ? indent(m_emitter->toString()) : std::string("null"));
}

#ifdef NORI_USE_IMGUI
bool Mesh::getImGuiNodes()
{
    ImGui::PushID(EShape);
    touched |= Shape::getImGuiNodes();
    ImGui::PopID();
    return touched;
}
#endif

#ifdef NORI_USE_OPTIX
	void Mesh::getOptixHitgroupRecords(OptixState &state, std::vector<HitGroupRecord> &hitgroupRecords)
	{
		copyMeshDataToDevice();

		HitGroupRecord rec = {};
		OPTIX_CHECK(optixSbtRecordPackHeader(state.m_hitgroup_prog_group[RAY_TYPE_RADIANCE], &rec));
		rec.data.geometry.type                   = GeometryData::TRIANGLE_MESH;
		rec.data.geometry.triangleMesh.positions = reinterpret_cast<float3 *>(d_V);
		rec.data.geometry.triangleMesh.normals   = reinterpret_cast<float3 *>(d_N);
		rec.data.geometry.triangleMesh.texcoords = reinterpret_cast<float2 *>(d_UV);
		rec.data.geometry.triangleMesh.indices   = reinterpret_cast<uint3 *>(d_F);

		Shape::getOptixHitgroupRecordsShape(rec);

		hitgroupRecords.push_back(rec);

		OPTIX_CHECK(optixSbtRecordPackHeader(state.m_hitgroup_prog_group[RAY_TYPE_SHADOWRAY], &rec));
		hitgroupRecords.push_back(rec);
	}

	void Mesh::copyMeshDataToDevice()
	{
		if (d_V != 0) return;

		{
			size_t bytes = m_V.size() * sizeof(float);
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_V), bytes));
			CUDA_CHECK(cudaMemcpy(
					reinterpret_cast<void *>(d_V),
					m_V.data(),
					bytes,
					cudaMemcpyHostToDevice
			));
		}

		{
			size_t bytes = m_N.size() * sizeof(float);
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_N), bytes));
			CUDA_CHECK(cudaMemcpy(
					reinterpret_cast<void *>(d_N),
					m_N.data(),
					bytes,
					cudaMemcpyHostToDevice
			));
		}

		{
			size_t bytes = m_UV.size() * sizeof(float);
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_UV), bytes));
			CUDA_CHECK(cudaMemcpy(
					reinterpret_cast<void *>(d_UV),
					m_UV.data(),
					bytes,
					cudaMemcpyHostToDevice
			));
		}

		{
			size_t bytes = m_F.size() * sizeof(uint32_t);
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_F), bytes));
			CUDA_CHECK(cudaMemcpy(
					reinterpret_cast<void *>(d_F),
					m_F.data(),
					bytes,
					cudaMemcpyHostToDevice
			));
		}
	}
#endif

	Mesh::~Mesh()
	{
#ifdef NORI_USE_OPTIX
		CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_V)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_N)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_UV)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_F)));
#endif
	}

NORI_NAMESPACE_END
