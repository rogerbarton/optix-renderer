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

      Intersection shadRayIts;

      // if we don't intersect, calculate the final color
      // initially check pdf_ems to reduce workload
      if(!scene->rayIntersect(eqr.shadowRay, shadRayIts)){
        // bsdf query
        BSDFQueryRecord bqr_ems(its.toLocal(eqr.wi), its.toLocal(-ray.d), EMeasure::ESolidAngle);
        bqr_ems.uv = its.uv; // set uv coordinates
        Color3f bsdf_col_ems = bsdf->eval(bqr_ems);

        // cos and add together
        float cos = std::abs(its.shFrame.cosTheta(its.toLocal(eqr.wi)));
        // renormalize by using count of lights
        result_ems = li * cos * bsdf_col_ems * scene->getLights().size();
        w_ems = pdf_ems / (pdf_ems + bsdf->pdf(bqr_ems));
      }
    }
    // Sampling the BSDF (mats)
  
    // bsdf query
    BSDFQueryRecord bqr_mat(its.toLocal(-ray.d));
    bqr_mat.uv = its.uv; // set uv coordinates
    bqr_mat.measure = ESolidAngle;
    Color3f bsdf_color = bsdf->sample(bqr_mat, sampler->next2D());

    if (!bsdf_color.isZero(Epsilon)) {
      Ray3f shadowRay(its.p, its.toWorld(bqr_mat.wo));
      // compute sampled ray interaction point
      pdf_mat = bsdf->pdf(bqr_mat);
      Intersection secondaryIts;
      if (scene->rayIntersect(shadowRay, secondaryIts) && secondaryIts.mesh->isEmitter()) {
        EmitterQueryRecord secondaryEQR(its.p, secondaryIts.p, secondaryIts.shFrame.n);
        auto ems = secondaryIts.mesh->getEmitter();
        result_mats = bsdf_color * ems->eval(secondaryEQR);
        w_mat = pdf_mat / (pdf_mat + ems->pdf(secondaryEQR) / scene->getLights().size());
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