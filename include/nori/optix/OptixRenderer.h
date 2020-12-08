//
// Created by roger on 06/12/2020.
//

#pragma once
#ifdef NORI_USE_OPTIX

#include "OptixState.h"
#include <nori/object.h>

NORI_NAMESPACE_BEGIN

	/**
	 * Stores properties related to optix for gui interface
	 * Does not use optix itself.
	 * Everything related to optix is inside the OptixState!
	 */
	struct OptixRenderer : public NoriObject
	{
		explicit OptixRenderer(const PropertyList &propList);

		NoriObject *cloneAndInit() override;
		void update(const NoriObject *guiObject) override;

		EClassType getClassType() const override { return ERenderer; }
		std::string toString() const override;

#ifdef NORI_USE_IMGUI
		NORI_OBJECT_IMGUI_NAME("OptixRenderer");
		bool getImGuiNodes() override;
#endif

		int  m_samplesPerLaunch;
		bool m_enableSpecialization; /// Set specific constants at compile time for better optimization
	protected:
	};

	NORI_REGISTER_CLASS(OptixRenderer, "optix");

NORI_NAMESPACE_END

#endif