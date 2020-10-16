#include <nori/bsdf.h>
#include <nori/integrator.h>
#include <nori/sampler.h>
#include <nori/scene.h>

NORI_NAMESPACE_BEGIN

class DirectMATSIntegrator : public Integrator {
public:
  DirectMATSIntegrator(const PropertyList &propList) {}
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

    Vector3f wi = its.toLocal((ray.o - its.p).normalized());

    // bsdf query
    BSDFQueryRecord bqr(wi);
    bqr.uv = its.uv; // set uv coordinates
    Color3f bsdf_color = bsdf->sample(bqr, sampler->next2D());

    if (bsdf_color.isZero()) {
      return result;
    }

    Ray3f shadowray(its.p, its.toWorld(bqr.wo));
    // compute sampled ray interaction point
    Intersection its2;
    if (!scene->rayIntersect(shadowray, its2))
      return result;
    // if not encounter emitter return radiance

    if (its2.mesh->isEmitter()) {
      EmitterQueryRecord EQR(its.p, its2.p, its2.shFrame.n);
      auto ems = its2.mesh->getEmitter();
      result += bsdf_color * ems->eval(EQR);
    }
    return result;
  }
  std::string toString() const {
    return std::string("DirectMATSIntegrator[]\n");
  }

protected:
};

NORI_REGISTER_CLASS(DirectMATSIntegrator, "direct_mats");
NORI_NAMESPACE_END