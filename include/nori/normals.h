#pragma once

#include <nori/integrator.h>
#include <nori/scene.h>

NORI_NAMESPACE_BEGIN

class NormalIntegrator : public Integrator
{
public:
    NormalIntegrator(const PropertyList &props)
    {
        /* No parameters this time */
    }

    Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const
    {
        /* Find the surface that is visible in the requested direction */
        Intersection its;
        if (!scene->rayIntersect(ray, its))
        {
            if (scene->getEnvMap())
            {
                return scene->getEnvMap()->eval(ray.d);
            }
            return Color3f(0.0f);
        }

        /* Return the component-wise absolute
           value of the shading normal as a color */
        Normal3f n = its.shFrame.n.cwiseAbs();
        return Color3f(n.x(), n.y(), n.z());
    }

    std::string toString() const
    {
        return "NormalIntegrator[]";
    }
#ifndef NORI_USE_NANOGUI
    virtual const char *getImGuiName() const override { return "Normals"; }
    virtual bool getImGuiNodes() override { return Integrator::getImGuiNodes(); }
#endif
};

NORI_NAMESPACE_END