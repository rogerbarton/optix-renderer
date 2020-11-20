#pragma once

#include <nori/object.h>

NORI_NAMESPACE_BEGIN

class EnvironmentMap : public NoriObject
{
public:
    virtual EClassType getClassType() const override { return EEnvironmentMap; }
    virtual Color3f eval(const Vector3f &_wi) = 0;
    virtual std::string toString() const override
    {
        return tfm::format("EnvironmentMap[]");
    }
#ifndef NORI_USE_NANOGUI
    virtual const char *getImGuiName() const override
    {
        return "EnvironmentMap Base";
    }
    virtual bool getImGuiNodes() override { return false; }
#endif
};

NORI_NAMESPACE_END