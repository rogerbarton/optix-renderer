#include <nori/bsdf.h>
#include <nori/integrator.h>
#include <nori/sampler.h>
#include <nori/scene.h>

NORI_NAMESPACE_BEGIN

class DirectMISIntegrator : public Integrator {
public:
  DirectMISIntegrator(const PropertyList &propList) {}
  Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const {
    Intersection its;
    if (!scene->rayIntersect(ray, its)) {
      return Color3f(0.f);
    }
    Color3f result(0.f);
    Color3f result_ems(0.f);
    Color3f result_mats(0.f);
    float w_ems = 0.f;
    float w_mat = 0.f;
    float pdf_ems = 0.f;
    float pdf_mat = 0.f;

    // get colliding object and shape
    auto shape = its.mesh;
    auto bsdf = shape->getBSDF();

    // if shape is emitter, add eval to result
    if (shape->isEmitter()) {
      auto emitter = shape->getEmitter();
      EmitterQueryRecord eqr(ray.o, its.p, its.shFrame.n);
      result += emitter->eval(eqr);
    }

    // sampling the emitter (EMS)
    auto l_ems = scene->getRandomEmitter(sampler->next1D());
    EmitterQueryRecord eqr(its.p);
    Color3f li = l_ems->sample(eqr, sampler->next2D());

    // check if 0 returned to prevent failed sampling
    if (!li.isZero(Epsilon)) {
      // normalize over all lights
      pdf_ems = l_ems->pdf(eqr) / scene->getLights().size();
      Vector3f wo_ems = its.toLocal(eqr.wi);
      Vector3f wi_ems = its.toLocal((ray.o - its.p).normalized());
      Intersection shadRayIts;
      scene->rayIntersect(eqr.shadowRay, shadRayIts);
      if ((eqr.p - shadRayIts.p).norm() < Epsilon &&
          std::abs(pdf_ems) >= Epsilon) {
        // bsdf query
        BSDFQueryRecord bqr_ems(wi_ems, wo_ems, EMeasure::ESolidAngle);
        bqr_ems.uv = its.uv; // set uv coordinates
        Color3f bsdf_col_ems = bsdf->eval(bqr_ems);

        // cos and add together
        float cos = std::abs(eqr.wi.dot(its.shFrame.n)) / eqr.wi.norm();
        // renormalize by using count of lights
        result_ems = li * cos * bsdf_col_ems * scene->getLights().size();
        w_ems = pdf_ems / (pdf_ems + bsdf->pdf(bqr_ems));
      }
    }

    // Sampling the BSDF (mats)
    Vector3f wi_mat = its.toLocal((ray.o - its.p).normalized());

    // bsdf query
    BSDFQueryRecord bqr_mat(wi_mat);
    bqr_mat.uv = its.uv; // set uv coordinates
    Color3f bsdf_color = bsdf->sample(bqr_mat, sampler->next2D());

    if (!bsdf_color.isZero(Epsilon)) {
      Ray3f shadowray(its.p, its.toWorld(bqr_mat.wo));
      // compute sampled ray interaction point
      pdf_mat = bsdf->pdf(bqr_mat);
      Intersection its2;
      if (scene->rayIntersect(shadowray, its2) &&
          std::abs(pdf_mat) >= Epsilon) {
        if (its2.mesh->isEmitter()) {
          EmitterQueryRecord EQR(its.p, its2.p, its2.shFrame.n);
          auto ems = its2.mesh->getEmitter();
          result_mats = bsdf_color * ems->eval(EQR);
          w_mat =
              pdf_mat / (pdf_mat + ems->pdf(EQR) / scene->getLights().size());
        }
      }
    }

    // COMBINE BOTH
    return result + w_ems * result_ems + w_mat * result_mats;
  }
  std::string toString() const {
    return std::string("DirectMISIntegrator[]\n");
  }

protected:
};

NORI_REGISTER_CLASS(DirectMISIntegrator, "direct_mis");
NORI_NAMESPACE_END