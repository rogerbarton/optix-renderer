#pragma once

#include <nori/object.h>
#include <nori/bitmap.h>

NORI_NAMESPACE_BEGIN

class Denoiser : public NoriObject
{
public:
    virtual EClassType getClassType() const override
    {
        return EDenoiser;
    }
    virtual Bitmap *denoise(const Bitmap *bitmap) const = 0;
    virtual std::string toString() const override
    {
        return tfm::format("Denoiser[]");
    }
#ifndef NORI_USE_NANOGUI
	NORI_OBJECT_IMGUI_NAME("Denoiser Base");
    virtual bool getImGuiNodes() override { return false; }
#endif
};

NORI_NAMESPACE_END