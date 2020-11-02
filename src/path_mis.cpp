#include <nori/bsdf.h>
#include <nori/integrator.h>
#include <nori/sampler.h>
#include <nori/scene.h>
#include <nori/warp.h>

NORI_NAMESPACE_BEGIN

class PathMISIntegrator : public Integrator
{
public:
    PathMISIntegrator(const PropertyList &propList) {}
    Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const
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
            float pdfems_bsdf = 0.f;
            float pdfmat_ems = 0.f;

            Intersection its;
            if (!scene->rayIntersect(traceRay, its))
            {
                break;
            }

            // get colliding object and shape
            const Shape *shape = its.mesh;
            const BSDF *bsdf = shape->getBSDF();

            // if shape is emitter, add eval to result
            if (shape->isEmitter())
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

            BSDFQueryRecord bRec_MATS(its.toLocal(-traceRay.d));
            // Sample BSDF
            Color3f bsdf_col = bsdf->sample(bRec_MATS, sampler->next2D());

            if (bRec_MATS.measure == EMeasure::EDiscrete)
            {
                w_mats = 1.0f;
                w_ems = 0.f; // do not have to do EMS
            }
            else
            {
                // ========== EMS =============
                const Emitter *emitter = scene->getRandomEmitter(sampler->next1D());

                EmitterQueryRecord eqr_ems(its.p);
                // sample the emitter
                Color3f ems_col = emitter->sample(eqr_ems, sampler->next2D());

                Vector3f we = its.toLocal(eqr_ems.wi);

                if (!ems_col.isZero(Epsilon))
                {
                    // set PDF only if a valid sample was done
                    pdfems = emitter->pdf(eqr_ems) / scene->getLights().size();
                    if (std::abs(pdfems) > Epsilon)
                    {
                        Ray3f shadowRay_ems(eqr_ems.shadowRay, Epsilon, (eqr_ems.p - its.p).norm() - Epsilon);
                        if (!scene->rayIntersect(shadowRay_ems))
                        {
                            BSDFQueryRecord bqr_ems(its.toLocal(-traceRay.d), we, EMeasure::ESolidAngle);
                            bqr_ems.uv = its.uv;
                            Color3f bsdf_col_ems = bsdf->eval(bqr_ems);
                            pdfems_bsdf = bsdf->pdf(bqr_ems);
                            float cos = std::abs(Frame::cosTheta(we));
                            Li_EMS = ems_col * cos * bsdf_col_ems * scene->getLights().size();
                        }
                        else
                        {
                            pdfems = 0.f;
                        }
                    }
                }

                // ========== MATS =============

                if (!bsdf_col.isZero(Epsilon))
                {

                    pdfmat = bsdf->pdf(bRec_MATS);
                    if (std::abs(pdfmat) < Epsilon)
                    {
                        // dieletric?
                        break;
                    }

                    Ray3f shadowray(its.p, its.toWorld(bRec_MATS.wo));
                    Intersection itsS;
                    if (scene->rayIntersect(shadowray, itsS))
                    {
                        if (itsS.mesh->isEmitter())
                        {
                            pdfmat_ems = itsS.mesh->getEmitter()->pdf(EmitterQueryRecord(its.p, itsS.p, itsS.shFrame.n));
                        }
                    }
                }
                else
                {
                    break;
                }

                // calculate the weights
                if ((pdfmat + pdfmat_ems) > Epsilon)
                {
                    w_mats = pdfmat / (pdfmat + pdfmat_ems);
                }
                if ((pdfems_bsdf + pdfems) > Epsilon)
                {
                    w_ems = pdfems / (pdfems_bsdf + pdfems);
                }
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
        return std::string("PathMISIntegrator[]\n");
    }

protected:
};

NORI_REGISTER_CLASS(PathMISIntegrator, "path_mis");

NORI_NAMESPACE_END