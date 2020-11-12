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
};

NORI_NAMESPACE_END