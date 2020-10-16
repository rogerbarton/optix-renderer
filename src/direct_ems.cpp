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
    Color3f result; // final color

    // get colliding object and shape
    auto shape = its.mesh;
    auto bsdf = shape->getBSDF();

    // if shape is emitter, add eval to result
    if (shape->isEmitter()) {
      auto emitter = shape->getEmitter();
      EmitterQueryRecord eqr(ray.o, its.p, its.shFrame.n);
      result += emitter->eval(eqr);
    }

    Vector3f wo = its.toLocal((ray.o - its.p).normalized());
    for (auto &l : scene->getLights()) {
      EmitterQueryRecord eqr(its.p);
      Color3f li = l->sample(eqr, sampler->next2D());

      // check if 0 returned to prevent failed sampling, continue
      if (std::abs(li.maxCoeff() - 0.f) < Epsilon) {
        continue;
      }

      // compute wi, wo was already computed above
      Vector3f wi = its.toLocal(eqr.wi);

      // compute shadow ray intersection
      Intersection shadRayIts;
      scene->rayIntersect(eqr.shadowRay, shadRayIts);

      if ((eqr.p - shadRayIts.p).norm() > Epsilon) {
        if (std::abs(l->pdf(eqr) - 1.0f) > Epsilon) {
          // we have an intersection on this point --> penumbra
          continue;
        }
      }

      // bsdf query
      BSDFQueryRecord bqr(wi, wo, EMeasure::ESolidAngle);
      bqr.uv = its.uv; // set uv coordinates
      Color3f bsdf_col = bsdf->eval(bqr);

      // cos and add together
      float cos = std::abs(eqr.wi.dot(its.shFrame.n)) / eqr.wi.norm();
      result += li * cos * bsdf_col;
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