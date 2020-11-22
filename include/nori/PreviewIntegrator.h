#pragma once

#include <nori/bsdf.h>
#include <nori/integrator.h>
#include <nori/mesh.h>
#include <nori/sampler.h>
#include <nori/scene.h>

NORI_NAMESPACE_BEGIN

class PreviewIntegrator : public Integrator
{
public:
	PreviewIntegrator(const PropertyList &props) {}
	NORI_OBJECT_DEFAULT_CLONE(PointLight)

    Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const
    {
        Intersection its;
        // if no collision at all, return black
        if (!scene->rayIntersect(ray, its))
        {
            if (scene->getEnvMap())
            {
                return scene->getEnvMap()->eval(ray.d);
            }
            return Color3f(0.f);
        }

        Color3f result; // final Color

        // get colliding object
        auto shape = its.mesh;
        auto bsdf = shape->getBSDF();
        // primary ray, pointing to camera
        Vector3f wo = its.toLocal((ray.o - its.p).normalized());

        const Emitter* l = scene->getRandomEmitter(sampler->next1D());

        EmitterQueryRecord rec(its.p);
        Color3f li = l->sample(rec, sampler->next2D()) * scene->getLights().size();

        Vector3f wi = its.toLocal(rec.wi);

        Intersection light_intersection;
        scene->rayIntersect(rec.shadowRay, light_intersection);

        BSDFQueryRecord bsdfRec(wi, wo, EMeasure::ESolidAngle);
        bsdfRec.uv = its.uv;                      // set the uv coordinates
        Color3f bsdf_color = bsdf->eval(bsdfRec); // eval the bsdf on the shape
        float cos = std::abs(rec.wi.dot(its.shFrame.n)) / rec.wi.norm();

        result += li * cos * bsdf_color;
        return result;
    }

	std::string toString() const
    {
        return "NormalIntegrator[]";
    }
#ifndef NORI_USE_NANOGUI
    virtual const char *getImGuiName() const override
    {
        return "Normals";
    }
    virtual bool getImGuiNodes() override { return Integrator::getImGuiNodes(); }
#endif
};

NORI_NAMESPACE_END