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

		Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &_ray, Color3f &albedo, Color3f &normal) const override
		{
			Color3f Li         = Color3f(0.f);
			Color3f throughput = Color3f(1.f);

			// Start in ambient scene medium for now
			const Medium *medium = scene->getAmbientMedium();
			Ray3f        ray     = _ray;

			int bounces = 0;
			while (true)
			{
				Intersection its;
				if (!scene->rayIntersect(ray, its)) {
					if(scene->getEnvMap()) {
						EmitterQueryRecord eqr(ray.d);
						Li += throughput * scene->getEnvMap()->eval(eqr);
					}
					break; // miss scene

				}

				// -- Find next interaction point
				MediumQueryRecord mRec(ray);
				const float       tm                  = medium->sampleFreePath(mRec, sampler->next1D());
				const bool        isMediumInteraction = tm < its.t;

				Vector3f p = isMediumInteraction ? ray(tm) : its.p;

				Color3f Tr = medium->getTransmittance(ray.o, p);
				throughput *= Tr;

				// -- Apply captured emission
				if(isMediumInteraction)
				{
					if(medium->getEmitter())
					{
						EmitterQueryRecord lRec{ray.o, its.p, 0.f};
						Li += throughput * medium->getEmitter()->eval(lRec);
					}
				}
				else
				{
					if (its.shape->getEmitter())
					{
						EmitterQueryRecord lRec{ray.o, its.p, its.shFrame.n};
						Li += throughput * its.shape->getEmitter()->eval(lRec);
					}
				}

				// -- Russian roulette with success probability
				{
					const float rouletteSuccess = std::min(throughput.maxCoeff(), 0.99f);

					if (sampler->next1D() > rouletteSuccess || rouletteSuccess < Epsilon)
						break;
					// Adjust throughput in case of survival
					throughput /= rouletteSuccess;
				}

				// -- Determine next ray
				Vector3f wo = ray.d; // default is unperturbed ray
				if (isMediumInteraction)
				{
					// Next interaction is in medium => Sample phase
					PhaseQueryRecord pRec(its.toLocal(-ray.d));
					// TODO: how to adjust throughput?
					throughput *= medium->getPhase()->sample(pRec, sampler->next2D());
					wo = its.toWorld(pRec.wo);
				}
				else
				{
					// Next interaction is on a surface => sample surface
					if (its.shape->getBSDF())
					{
						// Surface interaction
						const BSDF      *bsdf = its.shape->getBSDF();
						BSDFQueryRecord bRec{its.toLocal(-ray.d)};
						bRec.uv = its.uv;

						throughput *= bsdf->sample(bRec, sampler->next2D());
						// Note w_i . n is the cosTheta term included in the sample function
						wo      = its.toWorld(bRec.wo);
					}

					// Update current medium, if entering or leaving shape (note: overlaps unhandled)
					if(ray.d.dot(wo) > 0) // change in shape, else reflected
						medium = wo.dot(its.geoFrame.n) < 0.f && its.shape->getMedium() ?
					         its.shape->getMedium() :     // entering
					         scene->getAmbientMedium();   // leaving
				}

				ray = Ray3f(p, wo);

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