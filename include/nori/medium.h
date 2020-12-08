//
// Created by roger on 01/12/2020.
//

#pragma once

#include <nori/object.h>
#include <nori/phase.h>
#include <utility>

#ifdef NORI_USE_OPTIX
#include <nori/optix/cuda/MediumData.h>
#endif


NORI_NAMESPACE_BEGIN

	// TODO: obsolete for homog at the moment as the phase function sampling is done separately
	struct MediumQueryRecord
	{
		Ray3f ray;

		explicit MediumQueryRecord(const Ray3f &ray) : ray(ray) {}
	};

	struct Medium : NoriObject
	{
		/**
		 * Samples the free flight distance to the next interaction inside the medium.
		 * Note: intersections with the scene and medium boundary should be handled separately
		 * @return the time until the next interaction
		 */
		virtual float sampleFreePath(MediumQueryRecord &mRec, const Point1f &sample) const = 0;

		virtual Color3f getTransmittance(const Vector3f &from, const Vector3f &to) const = 0;

		const PhaseFunction *getPhase() const { return m_phase; }

		Emitter *getEmitter() { return m_emitter; }
		const Emitter *getEmitter() const { return m_emitter; }

		NoriObject *cloneAndInit() override = 0;

		/**
		 * Finish initialization for other components
		 * @param clone The already created clone
		 */
		void cloneAndInit(Medium *clone);

		void update(const NoriObject *guiObject) override;

		virtual ~Medium();

		void addChild(NoriObject *child) override;
		EClassType getClassType() const override { return EMedium; }

#ifdef NORI_USE_IMGUI
		NORI_OBJECT_IMGUI_NAME("Medium Base");
		virtual bool getImGuiNodes() override;
#endif

#ifdef NORI_USE_OPTIX
	virtual void getOptixMediumData(MediumData &sbtData);
#endif

	protected:
		PhaseFunction *m_phase   = nullptr;
		Emitter       *m_emitter = nullptr;
	};

NORI_NAMESPACE_END