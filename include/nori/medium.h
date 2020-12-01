//
// Created by roger on 01/12/2020.
//

#pragma once

#include <nori/object.h>
#include <nori/phase.h>

#include <utility>

NORI_NAMESPACE_BEGIN

	struct MediumQueryRecord
	{
		// Incident direction (in the local frame)
		Vector3f wi;

		// Outgoing direction (in the local frame)
		Vector3f wo;

		MediumQueryRecord(Vector3f wi) : wi(std::move(wi)) {}
		MediumQueryRecord(Vector3f wi, Vector3f wo) : wi(std::move(wi)), wo(std::move(wo)) {}
	};

	struct Medium : NoriObject
	{
		/**
		 * Samples the free flight distance to the next interaction
		 * @return the time until the next interaction
		 */
		virtual float sampleTr(MediumQueryRecord &mRec, const Point2f &sample) const = 0;

		virtual Color3f getTransmittance(const Vector3f &from, const Vector3f &to) const = 0;

		/**
		 * Finish initialization for other components
		 * @param clone The already created clone
		 */
		void cloneAndInit(Medium *clone);

		void update(const NoriObject *guiObject) override;

		void addChild(NoriObject *child) override;
		EClassType getClassType() const override { return EMedium; }

#ifdef NORI_USE_IMGUI
		NORI_OBJECT_IMGUI_NAME("Medium Base");
		virtual bool getImGuiNodes() override { return false; }
#endif

	protected:
		PhaseFunction *m_phase = nullptr;
	};

NORI_NAMESPACE_END