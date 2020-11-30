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
    explicit Sphere(const PropertyList &propList)
    {
        m_position = propList.getPoint3("center", Point3f());
        m_radius = propList.getFloat("radius", 1.f);

        m_bbox.expandBy(m_position - Vector3f(m_radius));
        m_bbox.expandBy(m_position + Vector3f(m_radius));
    }

	NoriObject *cloneAndInit() override
	{
		auto clone = new Sphere(*this);
		Shape::cloneAndInit(clone);
		return clone;
	}

	void update(const NoriObject *guiObject) override
	{
		const auto *gui = static_cast<const Sphere *>(guiObject);
		if (!gui->touched)return;
		gui->touched = false;

		m_position = gui->m_position;
		m_radius   = gui->m_radius;

		Shape::update(guiObject);

		if(gui->geometryTouched || gui->transformTouched)
		{
			m_bbox.expandBy(m_position - Vector3f(m_radius));
			m_bbox.expandBy(m_position + Vector3f(m_radius));
		}

		gui->geometryTouched = false;
		gui->transformTouched = false;
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
#ifdef NORI_USE_IMGUI
	NORI_OBJECT_IMGUI_NAME("Sphere");
    virtual bool getImGuiNodes() override
    {
    	touched |= Shape::getImGuiNodes();

        ImGui::AlignTextToFramePadding();
        ImGui::TreeNodeEx("center", ImGuiLeafNodeFlags, "Center");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
	    transformTouched |= ImGui::DragPoint3f("##value", &m_position, 0.02f);
        ImGui::NextColumn();

        ImGui::AlignTextToFramePadding();
        ImGui::TreeNodeEx("radius", ImGuiLeafNodeFlags, "Radius");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
	    geometryTouched |= ImGui::DragFloat("##value", &m_radius, 0.1f, 0, SLIDER_MAX_FLOAT, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::NextColumn();

        touched |= geometryTouched | transformTouched;
        return touched;
    }
    #endif

protected:
    Point3f m_position;
    float m_radius;
};

NORI_REGISTER_CLASS(Sphere, "sphere");
NORI_NAMESPACE_END
