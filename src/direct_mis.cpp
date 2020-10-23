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

      // if we don't intersect, calculate the final color
      if(!scene->rayIntersect(eqr.shadowRay)) {
        // bsdf query
        // flip wi and wo
        BSDFQueryRecord bqr_ems(its.toLocal(-ray.d), its.toLocal(eqr.wi), EMeasure::ESolidAngle);
        bqr_ems.uv = its.uv; // set uv coordinates
        bqr_ems.p = its.p;
        Color3f bsdf_col_ems = bsdf->eval(bqr_ems);

        float cos = its.shFrame.cosTheta(its.shFrame.toLocal(eqr.wi));

        // normalize over all lights, because we only use a random one
        float pdf_ems = l_ems->pdf(eqr) / scene->getLights().size();
        float pdf_mat = bsdf->pdf(bqr_ems);

        result_ems = li * cos * bsdf_col_ems * scene->getLights().size();
        if(pdf_ems + pdf_mat > Epsilon) {
          w_ems = pdf_ems / (pdf_ems + pdf_mat);
        }
      }
    }

    // Sampling the BSDF (mats)
  
    // bsdf query
    BSDFQueryRecord bqr_mat(its.toLocal(-ray.d));
    bqr_mat.uv = its.uv; // set uv coordinates
    bqr_mat.p = its.p;
    bqr_mat.measure = ESolidAngle;
    Color3f bsdf_color = bsdf->sample(bqr_mat, sampler->next2D());

    // check if failed sampling
    if (!bsdf_color.isZero(Epsilon)) {
      Ray3f shadowRay(its.p, its.toWorld(bqr_mat.wo));

      // compute sampled ray interaction point
      Intersection secondaryIts;

      if (scene->rayIntersect(shadowRay, secondaryIts) && secondaryIts.mesh->isEmitter()) {
        EmitterQueryRecord secondaryEQR(its.p, secondaryIts.p, secondaryIts.shFrame.n);
        auto ems = secondaryIts.mesh->getEmitter();

        result_mats = bsdf_color * ems->eval(secondaryEQR);
        
        float pdf_mat = bsdf->pdf(bqr_mat);
        float pdf_ems = ems->pdf(secondaryEQR);

        if(pdf_mat + pdf_ems > Epsilon) {
          w_mat = pdf_mat / (pdf_mat + pdf_ems);
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