//
// Created by roger on 01/12/2020.
//

#pragma once

#include <nori/object.h>

#include <utility>

NORI_NAMESPACE_BEGIN

	struct PhaseQueryRecord
	{
		// Incident direction (in the local frame)
		Vector3f wi;

		// Outgoing direction (in the local frame)
		Vector3f wo;

		PhaseQueryRecord(Vector3f wi) : wi(std::move(wi)) {}
		PhaseQueryRecord(Vector3f wi, Vector3f wo) : wi(std::move(wi)), wo(std::move(wo)) {}
	};

	/**
	 * \brief Superclass of all Phase functions, based on BSDF
	 */
	struct PhaseFunction : public NoriObject
	{
		virtual Color3f sample(PhaseQueryRecord &pRec, const Point2f &sample) const = 0;

		virtual float pdf(const PhaseQueryRecord &bRec) const = 0;

		virtual EClassType getClassType() const override { return EBSDF; }

#ifdef NORI_USE_IMGUI
		NORI_OBJECT_IMGUI_NAME("Phase Function Base");
		virtual bool getImGuiNodes() override { return false; }
#endif
	};

NORI_NAMESPACE_END
