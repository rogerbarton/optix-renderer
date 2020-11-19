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
#include <nori/warp.h>
#include <Eigen/Geometry>

NORI_NAMESPACE_BEGIN

class Sphere : public Shape
{
public:
    Sphere(const PropertyList &propList)
    {
        m_position = propList.getPoint3("center", Point3f());
        m_radius = propList.getFloat("radius", 1.f);

        m_bbox.expandBy(m_position - Vector3f(m_radius));
        m_bbox.expandBy(m_position + Vector3f(m_radius));
    }

    virtual BoundingBox3f getBoundingBox(uint32_t index) const override { return m_bbox; }

    virtual Point3f getCentroid(uint32_t index) const override { return m_position; }

    virtual bool rayIntersect(uint32_t index, const Ray3f &ray, float &u, float &v, float &t) const override
    {
        Vector3f L = ray.o - m_position;
        float a = ray.d.dot(ray.d);               // D^2
        float b = 2.f * ray.d.dot(L);             // 2OD
        float c = L.dot(L) - m_radius * m_radius; // O^2 - R^2

        // solve quadratic at^2 + bt + c = 0
        float discr = b * b - 4.f * a * c;
        if (discr < 0.f)
        {
            return false;
        }
        const float tmin = (-b - sqrt(discr)) / 2 / a;
        const float tmax = (-b + sqrt(discr)) / 2 / a;

        if (ray.mint <= tmin && ray.maxt >= tmin)
        {
            t = tmin;
            return true;
        }
        if (ray.mint <= tmax && ray.maxt >= tmax)
        {
            t = tmax;
            return true;
        }
        return false;
    }

    virtual void setHitInformation(uint32_t index, const Ray3f &ray, Intersection &its) const override
    {
        its.p = ray(its.t);
        Vector3f dir = (its.p - m_position).normalized();

        its.shFrame = Frame(dir);
        its.geoFrame = Frame(dir);

        Point2f uv_coords = sphericalCoordinates(-dir);

        // switch coordinates and map to [0,1]
        its.uv.x() = uv_coords.y() / (2.f * M_PI);
        its.uv.y() = uv_coords.x() / M_PI;
    }

    virtual void sampleSurface(ShapeQueryRecord &sRec, const Point2f &sample) const override
    {
        Vector3f q = Warp::squareToUniformSphere(sample);
        sRec.p = m_position + m_radius * q;
        sRec.n = q;
        sRec.pdf = std::pow(1.f / m_radius, 2) * Warp::squareToUniformSpherePdf(Vector3f(0.0f, 0.0f, 1.0f));
    }
    virtual float pdfSurface(const ShapeQueryRecord &sRec) const override
    {
        return std::pow(1.f / m_radius, 2) * Warp::squareToUniformSpherePdf(Vector3f(0.0f, 0.0f, 1.0f));
    }

    virtual std::string toString() const override
    {
        return tfm::format(
            "Sphere[\n"
            "  center = %s,\n"
            "  radius = %f,\n"
            "  bsdf = %s,\n"
            "  emitter = %s\n"
            "]",
            m_position.toString(),
            m_radius,
            m_bsdf ? indent(m_bsdf->toString()) : std::string("null"),
            m_emitter ? indent(m_emitter->toString()) : std::string("null"));
    }
#ifndef NORI_USE_NANOGUI
    virtual const char *getImGuiName() const override { return "Sphere"; }
    virtual void getImGuiNodes() override
    {
        // get ImGuiNodes for all children and own
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen |
                                   ImGuiTreeNodeFlags_Bullet;

        ImGui::AlignTextToFramePadding();
        ImGui::TreeNodeEx("center", flags, "Center");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragPoint3f("##value", &m_position);
        ImGui::NextColumn();

        ImGui::AlignTextToFramePadding();
        ImGui::TreeNodeEx("radius", flags, "Radius");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragFloat("##value", &m_radius, 0.1, 0, SLIDER_MAX_FLOAT, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::NextColumn();

        if (m_bsdf)
        {
            bool node_open_bsdf = ImGui::TreeNode("BSDF");
            ImGui::NextColumn();
            ImGui::AlignTextToFramePadding();

            ImGui::Text(m_bsdf->getImGuiName());
            ImGui::NextColumn();
            if (node_open_bsdf)
            {
                m_bsdf->getImGuiNodes();
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
                m_emitter->getImGuiNodes();
                ImGui::TreePop();
            }
        }
    }
    #endif

protected:
    Point3f m_position;
    float m_radius;
};

NORI_REGISTER_CLASS(Sphere, "sphere");
NORI_NAMESPACE_END
