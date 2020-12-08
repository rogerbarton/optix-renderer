#include <nori/bsdf.h>
#include <nori/integrator.h>
#include <nori/sampler.h>
#include <nori/scene.h>
#include <nori/warp.h>

NORI_NAMESPACE_BEGIN

class PathMISIntegrator : public Integrator
{
public:
    explicit PathMISIntegrator(const PropertyList &propList) {}
	NORI_OBJECT_DEFAULT_CLONE(PathMISIntegrator)
	NORI_OBJECT_DEFAULT_UPDATE(PathMISIntegrator)

    Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray, Color3f &albedo, Color3f &normal) const
    {
        Color3f Li = Color3f(0.f); // initial radiance
        Color3f t = Color3f(1.0);  // initial throughput

        Ray3f traceRay(ray);
        int counter = 0;
        float w_mats = 1.f;
        float w_ems = 0.f;

        while (true)
        {
            Color3f Li_EMS = Color3f(0.f);
            float pdfems = 0.f, pdfmat = 0.f;
            float pdfems_mats = 0.f;
            float pdfmat_ems = 0.f;

            Intersection its;
            if (!scene->rayIntersect(traceRay, its))
            {

                if (scene->getEnvMap())
                {
                    EmitterQueryRecord eqr;
                    eqr.wi = traceRay.d;
                    Li += t * scene->getEnvMap()->eval(eqr);
                }
                break;
            }

            // get colliding object and shape
            const Shape *shape = its.shape;
            const BSDF *bsdf = shape->getBSDF();

            // if shape is emitter, add eval to result
	        if (shape->getEmitter())
            {
                auto emitter = shape->getEmitter();
                EmitterQueryRecord eqr(traceRay.o, its.p, its.shFrame.n);
                Li += w_mats * t * emitter->eval(eqr);
            }

            float succ_prob = std::min(t.maxCoeff(), 0.99f);
            succ_prob = std::max(succ_prob, Epsilon);
            if (counter < 0)
            {
                counter++;
            }
            else if (sampler->next1D() > succ_prob)
            {
                break;
            }
            else
            {
                t /= succ_prob;
            }

            // WE NEED probs seen from its.p
            // ========== EMS =============
            const Emitter *emitter = scene->getRandomEmitter(sampler->next1D());
            if (!emitter)
            {
                throw NoriException("No Emitter in scene!");
            }

            EmitterQueryRecord eqr_ems(its.p);
            // sample the emitter
            Color3f ems_col = emitter->sample(eqr_ems, sampler->next2D());

            Vector3f we = its.toLocal(eqr_ems.wi);

            if (!ems_col.isZero(Epsilon))
            {
                if (!scene->rayIntersect(eqr_ems.shadowRay))
                {
                    BSDFQueryRecord bqr_ems(its.toLocal(-traceRay.d), we, EMeasure::ESolidAngle);
                    bqr_ems.uv = its.uv;
                    bqr_ems.p = its.p;
                    Color3f bsdf_col_ems = bsdf->eval(bqr_ems);

                    float cos = Frame::cosTheta(we);

                    Li_EMS = ems_col * cos * bsdf_col_ems * scene->getLights().size();
                    pdfems_mats = bsdf->pdf(bqr_ems);
                    pdfems = emitter->pdf(eqr_ems) / scene->getLights().size();
                }
            }
            if ((pdfems_mats + pdfems) > Epsilon)
            {
                w_ems = pdfems / (pdfems_mats + pdfems);
            }

            // ========== MATS =============
            BSDFQueryRecord bRec_MATS(its.toLocal(-traceRay.d));
            bRec_MATS.uv = its.uv;
            bRec_MATS.p = its.p;
            // Sample BSDF
            Color3f bsdf_col = bsdf->sample(bRec_MATS, sampler->next2D());

            if (!bsdf_col.isZero(Epsilon))
            {
                Ray3f shadowray(its.p, its.toWorld(bRec_MATS.wo));
                Intersection itsS;
                if (scene->rayIntersect(shadowray, itsS))
                {
	                if (itsS.shape->getEmitter())
                    {
                        EmitterQueryRecord eqr_mats(its.p, itsS.p, itsS.shFrame.n);
                        pdfmat = bsdf->pdf(bRec_MATS);
                        pdfmat_ems = itsS.shape->getEmitter()->pdf(eqr_mats) / scene->getLights().size();

                        if ((pdfmat + pdfmat_ems) > Epsilon)
                        {
                            w_mats = pdfmat / (pdfmat + pdfmat_ems);
                        }
                    }
                }
            }

            // check for discrete bsdf
            if (bRec_MATS.measure == EDiscrete)
            {
                w_ems = 0.f;
                w_mats = 1.f;
            }

            Li += w_ems * t * Li_EMS;
            t = t * bsdf_col;

            // create next ray to trace
            traceRay = Ray3f(its.p, its.toWorld(bRec_MATS.wo));
        }

        return Li;
    }
    std::string toString() const
    {
        return "PathMISIntegrator[]";
    }
#ifdef NORI_USE_IMGUI
	NORI_OBJECT_IMGUI_NAME("Path MIS");
    virtual bool getImGuiNodes() override { return Integrator::getImGuiNodes(); }
#endif
protected:
};

NORI_REGISTER_CLASS(PathMISIntegrator, "path_mis");

NORI_NAMESPACE_END