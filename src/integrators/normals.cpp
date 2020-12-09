
#include <nori/integrator.h>
#include <nori/scene.h>

NORI_NAMESPACE_BEGIN

	class NormalIntegrator : public Integrator
	{
	public:
		explicit NormalIntegrator(const PropertyList &props) {
			direction = props.getPoint3("direction", Point3f(0, 0, 1));
		}
		NORI_OBJECT_DEFAULT_CLONE(NormalIntegrator)
		NORI_OBJECT_DEFAULT_UPDATE(NormalIntegrator)

		Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray, Color3f &albedo, Color3f &normal) const
		{
			/* Find the surface that is visible in the requested direction */
			Intersection its;
			if (!scene->rayIntersect(ray, its))
			{
				if (scene->getEnvMap())
				{
					EmitterQueryRecord eqr;
					eqr.wi = ray.d;
					return scene->getEnvMap()->eval(eqr);
				}
				return Color3f(0.0f);
			}

			/* Return the component-wise absolute
			   value of the shading normal as a color */
			// Normal3f n = its.shFrame.n.cwiseAbs();
			Normal3f n = its.shFrame.toWorld(direction).cwiseAbs();
			return Color3f(n.x(), n.y(), n.z());
		}

		std::string toString() const
		{
			return "NormalIntegrator[]";
		}
#ifdef NORI_USE_IMGUI
		NORI_OBJECT_IMGUI_NAME("Normal");
		virtual bool getImGuiNodes() override {
			touched |= Integrator::getImGuiNodes();

			ImGui::AlignTextToFramePadding();
			ImGui::TreeNodeEx("direction", ImGuiLeafNodeFlags, "Direction");
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(-1);
			touched |= ImGui::DragFloat3("##value", direction.data(), 0.01f);
			ImGui::NextColumn();

			if (touched)
				direction.normalize();

			return touched;
		}
#endif
	protected:
		Normal3f direction;
	};

	NORI_REGISTER_CLASS(NormalIntegrator, "normals");
NORI_NAMESPACE_END
