#include <nori/bsdf.h>
#include <nori/integrator.h>
#include <nori/mesh.h>
#include <nori/sampler.h>
#include <nori/scene.h>

NORI_NAMESPACE_BEGIN

class DirectIntegrator : public Integrator
{
public:
  DirectIntegrator(const PropertyList &propList) {}

  Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const
  {
    Intersection its;
    // if no collision at all, return black
    if (!scene->rayIntersect(ray, its))
    {
      if (scene->getEnvMap())
      {
        EmitterQueryRecord eqr;
        eqr.wi = ray.d;
        return scene->getEnvMap()->eval(eqr);
      }
      return Color3f(0.f);
    }

    Color3f result; // final Color

    // get colliding object
    auto shape = its.mesh;
    auto bsdf = shape->getBSDF();
    // primary ray, pointing to camera
    Vector3f wo = its.toLocal((ray.o - its.p).normalized());

    for (auto &l : scene->getLights())
    {
      // create the emitter query record with the origin point
      EmitterQueryRecord rec(its.p);
      // light i, with a sampled point
      Color3f li = l->sample(rec, sampler->next2D());

      // secondary ray
      Vector3f wi = its.toLocal(rec.wi);

      // the intersection for the shadow (secondary) ray
      Intersection light_intersection;
      if (!scene->rayIntersect(rec.shadowRay))
      {

        // create BSDF query record based on wi, wo and the measure
        BSDFQueryRecord bsdfRec(wi, wo, EMeasure::ESolidAngle);
        bsdfRec.uv = its.uv; // set the uv coordinates
        bsdfRec.p = its.p;
        Color3f bsdf_color = bsdf->eval(bsdfRec); // eval the bsdf on the shape

        // calculate the angle
        float cos = std::abs(rec.wi.dot(its.shFrame.n)) / rec.wi.norm();

        // add up the bsdf term together with the cos and the sampled light
        result += li * cos * bsdf_color;
      }
    }
    return result;
  }

  std::string toString() const { return tfm::format("DirectIntegrator[]"); }
#ifndef NORI_USE_NANOGUI
  virtual const char *getImGuiName() const override
  {
    return "Direct";
  }
  virtual bool getImGuiNodes() override { return Integrator::getImGuiNodes(); }
#endif
};

NORI_REGISTER_CLASS(DirectIntegrator, "direct");
NORI_NAMESPACE_END