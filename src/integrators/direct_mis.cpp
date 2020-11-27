#include <nori/bsdf.h>
#include <nori/integrator.h>
#include <nori/sampler.h>
#include <nori/scene.h>

NORI_NAMESPACE_BEGIN

class DirectMISIntegrator : public Integrator
{
public:
  explicit DirectMISIntegrator(const PropertyList &propList) {}
  NORI_OBJECT_DEFAULT_CLONE(DirectMISIntegrator)
  NORI_OBJECT_DEFAULT_UPDATE(DirectMISIntegrator)

  Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const
  {
    Intersection its;
    if (!scene->rayIntersect(ray, its))
    {
      if (scene->getEnvMap())
      {
        EmitterQueryRecord eqr;
        eqr.wi = ray.d;
        return scene->getEnvMap()->eval(eqr);
      }
      else
      {
        return Color3f(0.f);
      }
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
    if (shape->isEmitter())
    {
      auto emitter = shape->getEmitter();
      EmitterQueryRecord eqr(ray.o, its.p, its.shFrame.n);
      result += emitter->eval(eqr);
    }

    // sampling the emitter (EMS)
    auto l_ems = scene->getRandomEmitter(sampler->next1D());
    EmitterQueryRecord eqr(its.p);
    Color3f li = l_ems->sample(eqr, sampler->next2D());

    // check if 0 returned to prevent failed sampling
    if (!li.isZero(Epsilon))
    {

      // if we don't intersect, calculate the final color
      if (!scene->rayIntersect(eqr.shadowRay))
      {
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
        if (pdf_ems + pdf_mat > Epsilon)
        {
          w_ems = pdf_ems / (pdf_ems + pdf_mat);
        }
      }
      else
      {
        // add env map
        if (scene->getEnvMap())
        {
          EmitterQueryRecord eqr2;
          eqr2.wi = eqr.shadowRay.d;
          result_ems = li * scene->getEnvMap()->eval(eqr2);
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
    if (!bsdf_color.isZero(Epsilon))
    {
      Ray3f shadowRay(its.p, its.toWorld(bqr_mat.wo));

      // compute sampled ray interaction point
      Intersection secondaryIts;

      if (scene->rayIntersect(shadowRay, secondaryIts) && secondaryIts.mesh->isEmitter())
      {
        EmitterQueryRecord secondaryEQR(its.p, secondaryIts.p, secondaryIts.shFrame.n);
        auto ems = secondaryIts.mesh->getEmitter();

        result_mats = bsdf_color * ems->eval(secondaryEQR); // / scene->getLights().size();

        float pdf_mat = bsdf->pdf(bqr_mat);
        float pdf_ems = ems->pdf(secondaryEQR) / scene->getLights().size();

        if (pdf_mat + pdf_ems > Epsilon)
        {
          w_mat = pdf_mat / (pdf_mat + pdf_ems);
        }
      }
      else
      {
        // add env map
        
        if (scene->getEnvMap())
        {
          EmitterQueryRecord eqr;
          eqr.wi = shadowRay.d;
          result_mats = bsdf_color * scene->getEnvMap()->eval(eqr);
        }
      }
    }

    // COMBINE BOTH
    return result + w_ems * result_ems + w_mat * result_mats;
  }
  std::string toString() const
  {
    return std::string("DirectMISIntegrator[]");
  }
#ifdef NORI_USE_IMGUI
	NORI_OBJECT_IMGUI_NAME("Direct MIS");
	virtual bool getImGuiNodes() override { return Integrator::getImGuiNodes(); }
#endif
};

NORI_REGISTER_CLASS(DirectMISIntegrator, "direct_mis");
NORI_NAMESPACE_END