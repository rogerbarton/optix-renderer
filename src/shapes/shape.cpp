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
		m_bsdf = static_cast<BSDF *>(NoriObjectFactory::createInstance("diffuse", PropertyList()));
	clone->m_bsdf = static_cast<BSDF *>(m_bsdf->cloneAndInit());

	if (m_medium)
		clone->m_medium = static_cast<Medium *>(m_medium->cloneAndInit());

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
	m_bsdf->update(gui->m_bsdf);

	if (m_medium)
		m_medium->update(gui->m_medium);

	if (m_normalMap)
		m_normalMap->update(gui->m_normalMap);

	// Note: Emitter updated by scene
	// if(m_emitter)
	// 	m_emitter->update(gui->m_emitter);
}

Shape::~Shape()
{
	delete m_bsdf;
	delete m_normalMap;
	//delete m_emitter; // Emitter is parent of Shape
}


void Shape::applyNormalMap(Intersection &its) const
{
	if (!m_normalMap) return;

	// note: normal map already normalized
	const Normal3f nmap = m_normalMap->eval(its.uv);

	// For validation, use global normals first and then check that they match when using the existing shading frame
	// Used for normals-identity-global
	// its.shFrame = Frame(Normal3f(n.x(), n.y(), n.z()));

	// its.shFrame = Frame(its.shFrame.toWorld(n));

	auto& f = its.shFrame;
	Vector3f s2 = f.toWorld(f.s);
	Vector3f t2 = f.toWorld(f.t);
	Eigen::Matrix3f tbn;
	tbn << f.s, f.t, f.n;
	Vector3f n2 = (tbn * nmap).normalized();

	f.s = s2;
	f.t = t2;
	f.n = n2;

	f = Frame(n2);
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
        bool node_open_bsdf = ImGui::TreeNode("BSDF");
        ImGui::NextColumn();
        ImGui::AlignTextToFramePadding();

        ImGui::Text(m_bsdf->getImGuiName().c_str());
        ImGui::NextColumn();
        if (node_open_bsdf)
        {
            touched |= m_bsdf->getImGuiNodes();
            ImGui::TreePop();
        }
    }

    if (m_emitter)
    {
        bool node_open_emitter = ImGui::TreeNode("Emitter");
        ImGui::NextColumn();
        ImGui::AlignTextToFramePadding();

        ImGui::Text(m_emitter->getImGuiName().c_str());
        ImGui::NextColumn();
        if (node_open_emitter)
        {
            touched |= m_emitter->getImGuiNodes();
            ImGui::TreePop();
        }
    }

	bool normalMapOpen = ImGui::TreeNode("Normal Map");
	ImGui::NextColumn();
	ImGui::AlignTextToFramePadding();

	ImGui::Text(m_normalMap ? m_normalMap->getImGuiName().c_str() : "None");
	ImGui::NextColumn();
	if (normalMapOpen)
	{
		if(m_normalMap)
			touched |= m_normalMap->getImGuiNodes();
		ImGui::TreePop();
	}

    ImGui::PopID();
    return touched;
}
#endif

NORI_NAMESPACE_END
