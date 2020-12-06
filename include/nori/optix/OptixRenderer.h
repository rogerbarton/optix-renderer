//
// Created by roger on 06/12/2020.
//

#pragma once
#ifdef NORI_USE_OPTIX

#include "OptixState.h"
#include <nori/object.h>

NORI_NAMESPACE_BEGIN

    struct OptixRenderer : public NoriObject {

        explicit OptixRenderer(const PropertyList& propList) {
            m_samplesPerLaunch = propList.getFloat("samplesPerLaunch", 16);
        }

        NORI_OBJECT_DEFAULT_CLONE(OptixRenderer);

        void update(const NoriObject* guiObject) override {
            const auto* gui = static_cast<const OptixRenderer*>(guiObject);
            if (!gui->touched)return;
            gui->touched = false;

            m_samplesPerLaunch = gui->m_samplesPerLaunch;
        }

        EClassType getClassType() const override { return ERenderer; }

        virtual std::string toString() const override {
            return tfm::format(
                    "OptixRenderer[\n"
                    "  samplesPerLaunch = %i\n"
                    "]",
                    m_samplesPerLaunch);
        }

#ifdef NORI_USE_IMGUI
        NORI_OBJECT_IMGUI_NAME("OptixRenderer");
        virtual bool getImGuiNodes() override { return false; }
#endif

        OptixState* createOptixState();

        void renderOptixState(OptixState* state);

    protected:
        int m_samplesPerLaunch;
    };

    NORI_REGISTER_CLASS(OptixRenderer, "optix");

NORI_NAMESPACE_END

#endif