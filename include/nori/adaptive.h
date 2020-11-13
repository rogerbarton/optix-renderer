#pragma once

#include <nori/object.h>
#include <nori/sampler.h>

NORI_NAMESPACE_BEGIN

class AdaptiveSampler : public NoriObject
{
public:
    AdaptiveSampler(const PropertyList &props);

    std::string toString() const override;

    EClassType getClassType() const override;

private:
};

NORI_REGISTER_CLASS(AdaptiveSampler, "adaptive");

NORI_NAMESPACE_END