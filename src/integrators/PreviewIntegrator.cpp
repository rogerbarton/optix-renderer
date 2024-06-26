
#include <nori/bsdf.h>
#include <nori/integrator.h>
#include <nori/mesh.h>
#include <nori/sampler.h>
#include <nori/scene.h>

NORI_NAMESPACE_BEGIN

class PreviewIntegrator : public Integrator
{
public:
	explicit PreviewIntegrator(const PropertyList &props) {}
	NORI_OBJECT_DEFAULT_CLONE(PreviewIntegrator)
	NORI_OBJECT_DEFAULT_UPDATE(PreviewIntegrator)

	int getSupportedLayers() const override
	{
		// Preview integrator needs to support all layers
		return ERenderLayer::Composite | ERenderLayer::Albedo | ERenderLayer::Normal;
	}

	Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray, Color3f &albedo, Color3f &normal) const
	{
		Intersection its;
		// if no collision at all, return black
		if (!scene->rayIntersect(ray, its))
		{
			if (scene->getEnvMap())
			{
				EmitterQueryRecord eqr;
				eqr.wi = ray.d;
				return scene->getEnvMap()->eval(eqr);
			}
			return Color3f(0.f);
		}
		normal = Color3f(its.shFrame.n.x(), its.shFrame.n.y(), its.shFrame.n.z());

		Color3f result; // final Color

		// get colliding object
		auto shape = its.shape;
		auto bsdf = shape->getBSDF();
		// TODO: set albedo if the bsdf has it

		// primary ray, pointing to camera
		Vector3f wo = its.toLocal((ray.o - its.p).normalized());

		const Emitter *light = scene->getRandomEmitter(sampler->next1D());
		if (light)
		{
			EmitterQueryRecord rec(its.p);
			Color3f li = light->sample(rec, sampler->next2D()) * scene->getLights().size();
			Vector3f wi = its.toLocal(rec.wi);

			if (!scene->rayIntersect(rec.shadowRay))
			{
				BSDFQueryRecord bsdfRec(wi, wo, EMeasure::ESolidAngle);
				bsdfRec.uv = its.uv;
				bsdfRec.p = its.p;
				Color3f bsdf_color = bsdf->eval(bsdfRec); // eval the bsdf on the shape
				float cos = std::abs(rec.wi.dot(its.shFrame.n)) / rec.wi.norm();

				result = li * cos * bsdf_color;
			}
		}
		else
		{
			Normal3f n = its.shFrame.n.cwiseAbs();
			result = Color3f(n.x(), n.y(), n.z());
		}

		return result;
	}

	std::string toString() const override
	{
		return "PreviewIntegrator[]";
	}
#ifdef NORI_USE_IMGUI
	NORI_OBJECT_IMGUI_NAME("Preview");
	virtual bool getImGuiNodes() override { return Integrator::getImGuiNodes(); }
#endif
};

NORI_REGISTER_CLASS(PreviewIntegrator, "preview");
NORI_NAMESPACE_END