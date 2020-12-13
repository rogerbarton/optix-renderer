#include <nori/bsdf.h>
#include <nori/integrator.h>
#include <nori/sampler.h>
#include <nori/scene.h>
#include <nori/warp.h>
#include <nori/phase.h>

NORI_NAMESPACE_BEGIN

	/**
	 * path_vol_mats but with importance sampling
	 */
	class PathVolMISIntegrator : public Integrator
	{
	public:
		explicit PathVolMISIntegrator(const PropertyList &propList) {}
		NORI_OBJECT_DEFAULT_CLONE(PathVolMISIntegrator)
		NORI_OBJECT_DEFAULT_UPDATE(PathVolMISIntegrator)

		Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &_ray, Color3f &albedo, Color3f &normal) const
		{
			Color3f Li         = Color3f(0.f);
			Color3f throughput = Color3f(1.f);

			// Start in ambient scene medium for now
			const Medium *medium = scene->getAmbientMedium();
			Ray3f        ray     = _ray;
			const auto   &lights = scene->getLights();

			float    pdf_mat         = 1.f; // pdf from last intersection, camera is like delta bsdf
			EMeasure pdf_mat_measure = EDiscrete;

			int bounces = 0;
			while (true)
			{
				Intersection its;
				if (!scene->rayIntersect(ray, its))
					break; // miss scene

				// -- Find next interaction point
				MediumQueryRecord mRec(ray);
				const float       tm                  = medium->sampleFreePath(mRec, sampler->next1D());
				const bool        isMediumInteraction = tm < its.t;

				Vector3f p = isMediumInteraction ? ray(tm) : its.p;

				Color3f Tr = medium->getTransmittance(ray.o, p, isMediumInteraction);
				throughput *= Tr;

				// -- Apply captured emission
				if (isMediumInteraction)
				{
					if (medium->getEmitter())
					{
						EmitterQueryRecord lRec{ray.o, its.p, 0.f};

						const float pdf_mat_em = medium->getEmitter()->pdf(lRec) / lights.size();
						// TODO: use pdf_medium
						const float w_mat      = pdf_mat > Epsilon ? pdf_mat / (pdf_mat + pdf_mat_em) : 0.f;

						Li += w_mat * throughput * medium->getEmitter()->eval(lRec);
					}
				}
				else
				{
					if (its.shape->getEmitter())
					{
						EmitterQueryRecord lRec{ray.o, its.p, its.shFrame.n};

						const float pdf_mat_em = its.shape->getEmitter()->pdf(lRec) / lights.size();
						const float w_mat      = pdf_mat_measure == EDiscrete ? 1.f :
						                         pdf_mat > Epsilon ? pdf_mat / (pdf_mat + pdf_mat_em) : 0.f;

						Li += w_mat * throughput * its.shape->getEmitter()->eval(lRec);
					}
				}

				// -- Russian roulette with success probability
				if (bounces >= 3 && (isMediumInteraction || its.shape->getBSDF()))
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
					medium->getPhase()->sample(pRec, sampler->next2D());
					pdf_mat = medium->getPhase()->pdf(pRec);
					wo      = its.toWorld(pRec.wo);
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

						const Color3f bsdfSample = bsdf->sample(bRec, sampler->next2D());
						// Note w_i . n is the cosTheta term included in the sample function
						wo = its.toWorld(bRec.wo);

						// -- Emitter sampling
						pdf_mat         = bsdf->pdf(bRec);
						pdf_mat_measure = bRec.measure;

						if (bRec.measure == ESolidAngle) // skip delta bsdf's
						{
							// sample random emitter and calculate its pdf for the weight
							const auto emitter        = scene->getRandomEmitter(sampler->next1D());

							EmitterQueryRecord lRec{its.p};
							const Color3f      Le     = emitter->sample(lRec, sampler->next2D()) * lights.size();
							const float        pdf_em = lRec.pdf / lights.size();

							// shadowRay data
							Ray3f        shadowRay = lRec.shadowRay;
							Color3f      shadowTr  = 1.f;
							bool         occluded  = false;
							const Medium *shadowMedium;
							if (ray.d.dot(shadowRay.d) > 0) // change in shape, else reflected
								shadowMedium = shadowRay.d.dot(its.geoFrame.n) < 0.f && its.shape->getMedium() ?
								               its.shape->getMedium() :     // entering
								               scene->getAmbientMedium();   // leaving
							else
								shadowMedium = medium;

							// Propagate shadowray through media until we hit a surface with a bsdf
							while (true)
							{
								Intersection shadowIts;
								if (!scene->rayIntersect(shadowRay, shadowIts))
									break;
								if (shadowIts.shape->getBSDF())
								{
									occluded = true;
									break;
								}
								shadowTr *= shadowMedium->getTransmittance(shadowRay.o, shadowIts.p, false);
								shadowMedium =
										shadowRay.d.dot(shadowIts.geoFrame.n) < 0.f && shadowIts.shape->getMedium() ?
										shadowIts.shape->getMedium() :     // entering
										scene->getAmbientMedium();   // leaving

								shadowRay.o = shadowIts.p; // Note: no normal maps applied to keep ray.d constant
								shadowRay.maxt -= shadowIts.t;
							}

							if (!occluded)
							{
								// Use same measure as the other bRec we sampled (its the same bsdf)
								BSDFQueryRecord bRecEm{its.toLocal(-ray.d), its.toLocal(shadowRay.d), bRec.measure};
								bRecEm.uv = its.uv;

								const float pdf_em_mat = bsdf->pdf(bRecEm);
								const float w_em       = pdf_em > Epsilon ? pdf_em / (pdf_em + pdf_em_mat) : 0.f;

								if (w_em > Epsilon)
									Li += w_em * shadowTr * throughput *
									      abs(lRec.wi.dot(its.shFrame.n)) * bsdf->eval(bRecEm) * Le;
							}
						} // End Emitter Sampling

					}

					// Update current medium, if entering or leaving shape (note: overlaps unhandled)
					if (ray.d.dot(wo) > 0) // change in shape, else reflected
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
			return "PathVolMISIntegrator[]";
		}
#ifdef NORI_USE_IMGUI
		NORI_OBJECT_IMGUI_NAME("Path MIS w/ Volumes");
		virtual bool getImGuiNodes() override { return Integrator::getImGuiNodes(); }
#endif

#ifdef NORI_USE_OPTIX
		IntegratorType getOptixIntegratorType() const override { return INTEGRATOR_TYPE_PATH_MIS; }
#endif
	};

	NORI_REGISTER_CLASS(PathVolMISIntegrator, "path_vol_mis");

NORI_NAMESPACE_END