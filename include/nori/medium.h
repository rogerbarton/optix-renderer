//
// Created by roger on 01/12/2020.
//

#pragma once

#include <nori/object.h>
#include <nori/phase.h>

NORI_NAMESPACE_BEGIN

	struct MediumQueryRecord
	{
		// Incident direction (in the local frame)
		Vector3f wi;

		// Outgoing direction (in the local frame)
		Vector3f wo;

		MediumQueryRecord(const Vector3f &wi) : wi(wi) {}
		MediumQueryRecord(const Vector3f &wi, const Vector3f &wo) : wi(wi), wo(wo) {}
	};

	struct Medium : NoriObject
	{
		/**
		 * Samples the free flight distance to the next interaction
		 * @return the time until the next interaction
		 */
		virtual float sampleTr(MediumQueryRecord &mRec, const Point2f &sample) const = 0;

		virtual float getTransmittance(const Vector3f& from, const Vector3f& to) const = 0;

		PhaseFunction *phase = nullptr;
	};

NORI_NAMESPACE_END