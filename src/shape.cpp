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
//#include <nori/warp.h>
//#include <Eigen/Geometry>

NORI_NAMESPACE_BEGIN

void Shape::cloneAndInit(Shape *clone)
{
	// If no material was assigned, instantiate a diffuse BRDF
	if (!m_bsdf)
		m_bsdf = static_cast<BSDF *>(NoriObjectFactory::createInstance("diffuse", PropertyList()));

	clone->m_bsdf = static_cast<BSDF *>(m_bsdf->cloneAndInit());

	if(m_emitter)
	{
		clone->m_emitter = static_cast<Emitter *>(m_emitter->cloneAndInit());
		clone->m_emitter->setShape(clone);
	}
}

void Shape::update(const NoriObject *guiObject)
{
	const auto *gui = static_cast<const Shape *>(guiObject);
	m_bsdf->update(gui->m_bsdf);
	if(m_emitter)
		m_emitter->update(gui->m_emitter);
    // Note: Emitter updated by scene
}

Shape::~Shape()
{
	delete m_bsdf;
	//delete m_emitter; // Emitter is parent of Shape
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

    default:
        throw NoriException("Shape::addChild(<%s>) is not supported!",
                            classTypeName(obj->getClassType()));
    }
}

std::string Intersection::toString() const
{
    if (!mesh)
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
        mesh ? mesh->toString() : std::string("null"));
}
#ifndef NORI_USE_NANOGUI
bool Shape::getImGuiNodes()
{
    ImGui::PushID(EShape);
    if (m_bsdf)
    {
        bool node_open_bsdf = ImGui::TreeNode("BSDF");
        ImGui::NextColumn();
        ImGui::AlignTextToFramePadding();

        ImGui::Text(m_bsdf->getImGuiName());
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

        ImGui::Text(m_emitter->getImGuiName());
        ImGui::NextColumn();
        if (node_open_emitter)
        {
            touched |= m_emitter->getImGuiNodes();
            ImGui::TreePop();
        }
    }
    ImGui::PopID();
    return touched;
}
#endif

NORI_NAMESPACE_END
