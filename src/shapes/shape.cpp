/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Romain Prévost

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

#include <nori/shape.h>
#include <nori/bsdf.h>
#include <nori/emitter.h>
#include <Eigen/Dense>
//#include <nori/warp.h>
//#include <Eigen/Geometry>

NORI_NAMESPACE_BEGIN

void Shape::cloneAndInit(Shape *clone)
{
	// If no material or medium was assigned, instantiate a diffuse BRDF
	if (!m_bsdf && !m_medium)
		m_bsdf = static_cast<BSDF *>(NoriObjectFactory::createInstance("diffuse"));
	if(m_bsdf)
		clone->m_bsdf = static_cast<BSDF *>(m_bsdf->cloneAndInit());

	if (m_medium)
	{
		if(clone->m_medium->getEmitter())
			m_medium->getEmitter()->setShape(this);
		clone->m_medium = static_cast<Medium *>(m_medium->cloneAndInit());
		if(clone->m_medium->getEmitter())
			clone->m_medium->getEmitter()->setShape(clone);
	}

	if (m_normalMap)
		clone->m_normalMap = static_cast<Texture<Normal3f> *>(m_normalMap->cloneAndInit());

	if (m_emitter)
	{
		clone->m_emitter = static_cast<Emitter *>(m_emitter->cloneAndInit());
		clone->m_emitter->setShape(clone);
	}
}

void Shape::update(const NoriObject *guiObject)
{
	const auto *gui = static_cast<const Shape *>(guiObject);
	if(m_bsdf)
		m_bsdf->update(gui->m_bsdf);

	if (m_medium)
		m_medium->update(gui->m_medium);

	if (m_normalMap)
		m_normalMap->update(gui->m_normalMap);

	if (m_emitter)
		m_emitter->update(gui->m_emitter);

	// Update volume
	if(geometryTouched)
		updateVolume();
}

Shape::~Shape()
{
	delete m_bsdf;
	delete m_normalMap;
	delete m_medium;
	delete m_emitter;
}


void Shape::applyNormalMap(Intersection &its) const
{
	if (!m_normalMap) return;

	// note: normal map already normalized
	const Normal3f nmap = m_normalMap->eval(its.uv);
	its.shFrame = Frame(its.toWorld(nmap));
}

void Shape::sampleVolume(ShapeQueryRecord &sRec, const Point3f &sample) const
{
	sRec.p   = m_bbox.min + m_bbox.getExtents().cwiseProduct(sample);
	sRec.pdf = pdfVolume(sRec);
}

float Shape::pdfVolume(const ShapeQueryRecord &sRec) const
{
	return 1 / m_volume;
}

void Shape::updateVolume()
{
	m_volume = m_bbox.getVolume();
}

void Shape::addChild(NoriObject *obj)
{
    switch (obj->getClassType())
    {
    case EBSDF:
        if (m_bsdf)
            throw NoriException(
                "Shape: tried to register multiple BSDF instances!");
        m_bsdf = static_cast<BSDF *>(obj);
        break;

    case EEmitter:
        if (m_emitter)
            throw NoriException(
                "Shape: tried to register multiple Emitter instances!");
        m_emitter = static_cast<Emitter *>(obj);
        m_emitter->setShape(static_cast<Shape *>(this));
        break;

    case EMedium:
	    if (m_medium)
		    throw NoriException("Shape: tried to register multiple Medium instances.");
	    m_medium = static_cast<Medium *>(obj);
	    break;

    case ETexture:
	    if (obj->getIdName() == "normal")
	    {
		    if (m_normalMap)
			    throw NoriException("There is already a normal map defined!");
		    m_normalMap = static_cast<Texture<Normal3f> *>(obj);
	    }
	    else
		    throw NoriException("Shape does not have a texture with name: %s", obj->getIdName());
	    break;

    default:
        throw NoriException("Shape::addChild(<%s>) is not supported!",
                            classTypeName(obj->getClassType()));
    }
}

std::string Intersection::toString() const
{
    if (!shape)
        return "Intersection[invalid]";

    return tfm::format(
        "Intersection[\n"
        "  p = %s,\n"
        "  t = %f,\n"
        "  uv = %s,\n"
        "  shFrame = %s,\n"
        "  geoFrame = %s,\n"
        "  mesh = %s\n"
        "]",
        p.toString(),
        t,
        uv.toString(),
        indent(shFrame.toString()),
        indent(geoFrame.toString()),
        shape ? shape->toString() : std::string("null"));
}

#ifdef NORI_USE_IMGUI
bool Shape::getImGuiNodes()
{
    ImGui::PushID(EShape);
    if (m_bsdf)
    {
        bool nodeOpen = ImGui::TreeNode("BSDF");
        ImGui::NextColumn();
        ImGui::AlignTextToFramePadding();

        ImGui::Text(m_bsdf->getImGuiName().c_str());
        ImGui::NextColumn();
        if (nodeOpen)
        {
            touched |= m_bsdf->getImGuiNodes();
            ImGui::TreePop();
        }
    }

    if (m_emitter)
    {
        bool nodeOpen = ImGui::TreeNode("Emitter");
        ImGui::NextColumn();
        ImGui::AlignTextToFramePadding();

        ImGui::Text(m_emitter->getImGuiName().c_str());
        ImGui::NextColumn();
        if (nodeOpen)
        {
            touched |= m_emitter->getImGuiNodes();
            ImGui::TreePop();
        }
    }

    if(m_normalMap)
    {
	    bool nodeOpen = ImGui::TreeNode("Normal Map");
	    ImGui::NextColumn();
	    ImGui::AlignTextToFramePadding();

	    ImGui::Text(m_normalMap->getImGuiName().c_str());
	    ImGui::NextColumn();
	    if (nodeOpen)
	    {
		    touched |= m_normalMap->getImGuiNodes();
		    ImGui::TreePop();
	    }
    }

	if (m_medium)
	{
		bool nodeOpen = ImGui::TreeNode("Medium");
		ImGui::NextColumn();
		ImGui::AlignTextToFramePadding();

		ImGui::Text(m_medium->getImGuiName().c_str());
		ImGui::NextColumn();
		if (nodeOpen)
		{
			touched |= m_medium->getImGuiNodes();
			ImGui::TreePop();
		}
	}

    ImGui::PopID();
    return touched;
}
#endif

#ifdef NORI_USE_OPTIX
	/**
	 * Default is to use bbox with custom intersection
	 */
	OptixBuildInput Shape::getOptixBuildInput()
	{
		// AABB build input
		OptixAabb aabb = {m_bbox.min.x(), m_bbox.min.y(), m_bbox.min.z(),
		                  m_bbox.max.x(), m_bbox.max.y(), m_bbox.max.z()};

		// TODO: delete this
		CUdeviceptr d_aabb_buffer;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>( &d_aabb_buffer ), sizeof(OptixAabb)));
		CUDA_CHECK(cudaMemcpy(
				reinterpret_cast<void *>( d_aabb_buffer ),
				&aabb,
				sizeof(OptixAabb),
				cudaMemcpyHostToDevice
		));

		uint32_t aabb_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};

		OptixBuildInput buildInput = {};
		buildInput.type                               = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
		buildInput.customPrimitiveArray.aabbBuffers   = &d_aabb_buffer;
		buildInput.customPrimitiveArray.numPrimitives = 1;
		buildInput.customPrimitiveArray.flags         = aabb_input_flags;
		buildInput.customPrimitiveArray.numSbtRecords = 1;

		return buildInput;
	}

	void Shape::getOptixHitgroupRecordsShape(HitGroupRecord &rec)
	{
		// Copy shape specifics to the record
		rec.data.geometry.volume = m_volume;
		if (m_normalMap)
		{
			float3 constNormalDummy;
			m_normalMap->getOptixTexture(constNormalDummy, rec.data.bsdf.normalTex);
		}

		if (m_bsdf)
			m_bsdf->getOptixMaterialData(rec.data.bsdf);

		if (m_medium)
			m_medium->getOptixMediumData(rec.data.medium);

		if (m_emitter)
			m_emitter->getOptixEmitterData(rec.data.emitter);
	}
#endif

NORI_NAMESPACE_END
