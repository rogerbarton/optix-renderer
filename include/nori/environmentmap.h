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

    virtual const char *getImGuiName() const override { return "EnvironmentMap Base"; }
    virtual void getImGuiNodes() override {}
};

NORI_NAMESPACE_END