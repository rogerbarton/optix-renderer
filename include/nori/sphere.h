//
// Created by roger on 07/12/2020.
//

#pragma once

#include <nori/shape.h>
#include <nori/bsdf.h>
#include <nori/emitter.h>
#include <nori/warp.h>
#include <Eigen/Geometry>

NORI_NAMESPACE_BEGIN

	class Sphere : public Shape
	{
	public:
		explicit Sphere(const PropertyList &propList);

		NoriObject *cloneAndInit() override;
		void update(const NoriObject *guiObject) override;

		virtual BoundingBox3f getBoundingBox(uint32_t index) const override { return m_bbox; }
		virtual Point3f getCentroid(uint32_t index) const override { return m_position; }

		bool rayIntersect(uint32_t index, const Ray3f &ray, float &u, float &v, float &t) const override;
		void setHitInformation(uint32_t index, const Ray3f &ray, Intersection &its) const override;

		void sampleSurface(ShapeQueryRecord &sRec, const Point2f &sample) const override;
		float pdfSurface(const ShapeQueryRecord &sRec) const override;

		void sampleVolume(ShapeQueryRecord &sRec, const Point3f &sample) const override;

		std::string toString() const override;
#ifdef NORI_USE_IMGUI
		NORI_OBJECT_IMGUI_NAME("Sphere");
		bool getImGuiNodes() override;
#endif

	protected:
		void updateVolume() override { m_volume = 4.f / 3.f * M_PI * std::pow(m_radius, 3); }

		Point3f m_position;
		float   m_radius;
	};

	NORI_REGISTER_CLASS(Sphere, "sphere");
NORI_NAMESPACE_END
