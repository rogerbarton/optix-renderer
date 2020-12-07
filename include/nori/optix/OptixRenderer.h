//
// Created by roger on 06/12/2020.
//

#pragma once
#ifdef NORI_USE_OPTIX

#include "OptixState.h"
#include <nori/object.h>

NORI_NAMESPACE_BEGIN

	/**
	 * The bridge between nori/gui and optix.
	 * Does not use optix itself.
	 * Everything related to optix is inside the OptixState
	 */
	struct OptixRenderer : public NoriObject
	{
		// -- Interface
		void renderOptixState();

		// -- Nori Object
		explicit OptixRenderer(const PropertyList &propList);
		~OptixRenderer();

		NoriObject *cloneAndInit() override;;
		void update(const NoriObject *guiObject) override;

		EClassType getClassType() const override { return ERenderer; }
		std::string toString() const override;

#ifdef NORI_USE_IMGUI
		NORI_OBJECT_IMGUI_NAME("OptixRenderer");
		bool getImGuiNodes() override;
#endif

	protected:
		int m_samplesPerLaunch;

		OptixState *m_optixState;
	};

	NORI_REGISTER_CLASS(OptixRenderer, "optix");

NORI_NAMESPACE_END

#endif