#include <nori/bsdf.h>
#include <nori/integrator.h>
#include <nori/sampler.h>
#include <nori/scene.h>
#include <nori/warp.h>

NORI_NAMESPACE_BEGIN

class PathMATSIntegrator : public Integrator
{
public:
    PathMATSIntegrator(const PropertyList &propList) {}
    
    Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const
    {
        Color3f Li = Color3f(0.f); // initial radiance§
        Color3f t = Color3f(1.0);  // initial throughput
        Ray3f traceRay(ray);
        int counter = 0;
        while (true)
        {
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
                Li = Li + t * emitter->eval(eqr);
            }

            float succ_prob = std::min(t.maxCoeff(), 0.99f);
            if (counter < 3)
            {
                counter++;
            }
            else if (sampler->next1D() > succ_prob)
            {
                break;
            }
            else
            {
                t = t / succ_prob;
            }

            BSDFQueryRecord bRec(its.toLocal(-traceRay.d));

            // Sample BSDF
            Color3f bsdf_col = bsdf->sample(bRec, sampler->next2D());

            t = t * bsdf_col;

            // create next ray to trace
            traceRay = Ray3f(its.p, its.toWorld(bRec.wo));
        }

        return Li;
    }

    std::string toString() const
    {
        return std::string("PathMATSIntegrator[]\n");
    }

protected:
};

NORI_REGISTER_CLASS(PathMATSIntegrator, "path_mats");
NORI_NAMESPACE_END