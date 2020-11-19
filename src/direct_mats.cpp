#include <nori/bsdf.h>
#include <nori/integrator.h>
#include <nori/sampler.h>
#include <nori/scene.h>
#include <nori/warp.h>

NORI_NAMESPACE_BEGIN

class DirectMATSIntegrator : public Integrator
{
public:
  DirectMATSIntegrator(const PropertyList &propList) {}
  Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const
  {
    Intersection its;
    Color3f result(0.f); // final color

    if (!scene->rayIntersect(ray, its))
    {
      if (scene->getEnvMap())
      {
        result += scene->getEnvMap()->eval(ray.d);
      }

      return result;
    }

    // get colliding object and shape
    auto shape = its.mesh;
    auto bsdf = shape->getBSDF();

    // if shape is emitter, add eval to result
    if (shape->isEmitter())
    {
      auto emitter = shape->getEmitter();
      EmitterQueryRecord eqr(ray.o, its.p, its.shFrame.n);
      result += emitter->eval(eqr);
    }

    BSDFQueryRecord bRec(its.toLocal(-ray.d));
    bRec.uv = its.uv;
    bRec.p = its.p;
    bRec.measure = ESolidAngle;

    Color3f bsdf_col = bsdf->sample(bRec, sampler->next2D());

    if (bsdf_col.isZero(Epsilon))
    {
      return result;
    }

    // secondary ray (wo), check if hit emitter
    Intersection secondaryIts;
    Ray3f secondaryRay(its.p, its.toWorld(bRec.wo));

    if (!scene->rayIntersect(secondaryRay, secondaryIts))
    {
      // if the ray does not intersect again, it will intersect the Env map
      if (scene->getEnvMap())
      {
        result += scene->getEnvMap()->eval(secondaryRay.d) * bsdf_col;
      }

      return result;
    }

    // test if emitter was hit
    if (secondaryIts.mesh->isEmitter())
    {
      EmitterQueryRecord secondaryEQR(its.p, secondaryIts.p, secondaryIts.shFrame.n);

      result += secondaryIts.mesh->getEmitter()->eval(secondaryEQR) * bsdf_col;
    }

    return result;
  }
  std::string toString() const
  {
    return std::string("DirectMATSIntegrator[]");
  }

  virtual const char *getImGuiName() const override { return "Direct MATS"; }
  virtual void getImGuiNodes() override {}

protected:
};

NORI_REGISTER_CLASS(DirectMATSIntegrator, "direct_mats");
NORI_NAMESPACE_END