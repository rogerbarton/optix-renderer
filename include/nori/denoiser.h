#pragma once

#include <nori/object.h>

NORI_NAMESPACE_BEGIN

class Denoiser : public NoriObject
{
public:
    virtual EClassType getClassType() const override
    {
        return EDenoiser;
    }
    virtual void denoise(ImageBlock* block) const = 0;
    virtual std::string toString() const override
    {
        return tfm::format("Denoiser[]");
    }
#ifdef NORI_USE_IMGUI
    NORI_OBJECT_IMGUI_NAME("Denoiser Base");
    virtual bool getImGuiNodes() override { return false; }
#endif
};

NORI_NAMESPACE_END