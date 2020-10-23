#include <nori/bsdf.h>
#include <nori/integrator.h>
#include <nori/sampler.h>
#include <nori/scene.h>

NORI_NAMESPACE_BEGIN

class DirectEMSIntegrator : public Integrator {
public:
  DirectEMSIntegrator(const PropertyList &propList) {}
  Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const {
    Intersection its;
    if (!scene->rayIntersect(ray, its)) {
      return Color3f(0.f);
    }
    Color3f result(0.f); // final color

    // get colliding object and shape
    auto shape = its.mesh;
    auto bsdf = shape->getBSDF();

    // if shape is emitter, add eval to result
    if (shape->isEmitter()) {
      auto emitter = shape->getEmitter();
      EmitterQueryRecord eqr(ray.o, its.p, its.shFrame.n);
      result += emitter->eval(eqr);
    }

    Vector3f wo = its.toLocal(-ray.d);
    for (auto &l : scene->getLights()) {
      EmitterQueryRecord eqr(its.p);
      Color3f li = l->sample(eqr, sampler->next2D());

      // check if 0 returned to prevent failed sampling, continue
      if (li.isZero(Epsilon)) {
        continue;
      }

      // compute shadow ray intersection
      // we have penumbra because we have hit another obstacle      
      if(scene->rayIntersect(eqr.shadowRay)) {
        continue;
      }

      // bsdf query
      BSDFQueryRecord bqr(its.toLocal(eqr.wi), wo, EMeasure::ESolidAngle);
      bqr.uv = its.uv; // set uv coordinates
      bqr.p = its.p; // set point p

      Color3f bsdf_col = bsdf->eval(bqr);

      // calc cos and add together
      result += li * std::abs(its.shFrame.cosTheta(its.toLocal(eqr.wi))) * bsdf_col;
    }
    return result;
  }
  std::string toString() const {
    return std::string("DirectEMSIntegrator[]\n");
  }

protected:
};

NORI_REGISTER_CLASS(DirectEMSIntegrator, "direct_ems");
NORI_NAMESPACE_END