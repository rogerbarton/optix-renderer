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
		static Vector3f wi() { return Vector3f::UnitZ(); };

		// Outgoing direction (in the local frame) with wi=(0,0,1)
		Vector3f wo;

		PhaseQueryRecord() = default;
		PhaseQueryRecord(Vector3f wo) : wo(std::move(wo)) {}
	};

	/**
	 * \brief Superclass of all Phase functions, based on BSDF
	 */
	struct PhaseFunction : public NoriObject
	{
		virtual void sample(PhaseQueryRecord &pRec, const Point2f &sample) const = 0;

		virtual float pdf(const PhaseQueryRecord &bRec) const = 0;

		virtual EClassType getClassType() const override { return EPhaseFunction; }

#ifdef NORI_USE_IMGUI
		NORI_OBJECT_IMGUI_NAME("Phase Function Base");
		virtual bool getImGuiNodes() override { return false; }
#endif
	};

NORI_NAMESPACE_END
