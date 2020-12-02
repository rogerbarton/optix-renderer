#include <nori/bsdf.h>
#include <nori/integrator.h>
#include <nori/sampler.h>
#include <nori/scene.h>
#include <nori/warp.h>

NORI_NAMESPACE_BEGIN

/**
 * path_mats but with support for media/volumes
 */
class PathVolMATSIntegrator : public Integrator
{
public:
    explicit PathVolMATSIntegrator(const PropertyList &propList) {}
	NORI_OBJECT_DEFAULT_CLONE(PathVolMATSIntegrator)
	NORI_OBJECT_DEFAULT_UPDATE(PathVolMATSIntegrator)

    Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &_ray) const override
    {
	    Color3f Li = Color3f(0.f);
	    Color3f throughput = Color3f(1.f);

	    Ray3f ray = _ray;
	    int bounces = 0;
	    while (true) {
		    Intersection its;
		    if (!scene->rayIntersect(ray, its))
			    break; // miss scene

		    if (its.shape->isEmitter()) {
			    EmitterQueryRecord lRec{ray.o, its.p, its.shFrame.n};
			    Li += throughput * its.shape->getEmitter()->eval(lRec);
		    }

		    // Russian roulette with success probability
		    {
			    const float rouletteSuccess = std::min(throughput.maxCoeff(), 0.99f);

			    if (sampler->next1D() > rouletteSuccess || rouletteSuccess < Epsilon)
				    break;
			    // Adjust throughput in case of survival
			    throughput /= rouletteSuccess;
		    }

		    auto bsdf = its.shape->getBSDF();
		    BSDFQueryRecord bRec{its.toLocal(-ray.d)};
		    bRec.uv = its.uv;

		    throughput *= bsdf->sample(bRec, sampler->next2D());
		    // Note w_i . n is the cosTheta term included in the sample function

		    ray = Ray3f(its.p, its.toWorld(bRec.wo));

		    bounces++;
	    }

	    return Li;
    }

    std::string toString() const
    {
        return std::string("PathVolMATSIntegrator[]");
    }
#ifdef NORI_USE_IMGUI
	NORI_OBJECT_IMGUI_NAME("Path MATS w/ Volumes");
    virtual bool getImGuiNodes() override { return Integrator::getImGuiNodes(); }
#endif
protected:
};

NORI_REGISTER_CLASS(PathVolMATSIntegrator, "path_vol_mats");
NORI_NAMESPACE_END