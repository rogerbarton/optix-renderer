#include <nori/bsdf.h>
#include <nori/integrator.h>
#include <nori/sampler.h>
#include <nori/scene.h>
#include <nori/medium.h>
#include <nori/phase.h>

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
			Color3f Li         = Color3f(0.f);
			Color3f throughput = Color3f(1.f);

			const Medium *medium = nullptr;
			Ray3f  ray     = _ray;

			int bounces = 0;
			while (true)
			{
				Intersection its;
				if (!scene->rayIntersect(ray, its))
					break; // miss scene

				if (medium)
				{
					MediumQueryRecord mRec(ray);
					const float tm = medium->sampleFreePath(mRec, sampler->next1D());
					if(tm < its.t)
					{
						// medium interaction
					}
					else
					{
						// surface interaction + medium transmittance
						if (its.shape->getBSDF())
						{

						}
						else // its.shape->getMedium()
						{
							// Change in medium, ray direction const
						}
					}

					Color3f Tr = medium->getTransmittance(ray.o, its.p);
				}
				else
				{
					// Vacuum
					if (its.shape->getBSDF())
					{

					}
					else // its.shape->getMedium()
					{
						// Change in medium, ray direction const
					}
				}

				if (its.shape->isEmitter())
				{
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

				// Sample interaction for next ray
				Vector3f p  = its.p;
				Vector3f wo = ray.d; // default is unperturbed ray
				if (its.shape->getBSDF())
				{
					const BSDF      *bsdf = its.shape->getBSDF();
					BSDFQueryRecord bRec{its.toLocal(-ray.d)};
					bRec.uv = its.uv;

					throughput *= bsdf->sample(bRec, sampler->next2D());
					// Note w_i . n is the cosTheta term included in the sample function
					wo      = its.toWorld(bRec.wo);
				}
				else
				{
					// Use medium
					PhaseQueryRecord pRec(its.toLocal(-ray.d));
					// TODO: how to adjust throughput?
					throughput *= its.shape->getMedium()->getPhase()->sample(pRec, sampler->next2D());
					wo = its.toWorld(pRec.wo);

					// Update current medium, if entering or leaving shape (note: overlaps unhandled)
					medium = pRec.wi.dot(its.geoFrame.n) < 0.f ? nullptr : its.shape->getMedium();
				}

				ray = Ray3f(its.p, wo);

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