#include <nori/bsdf.h>
#include <nori/frame.h>
#include <nori/warp.h>

NORI_NAMESPACE_BEGIN

	/**
	 * Anisotropic phase function for volumes based on Henyey-Greenstein.
	 */
	class AnisoPhase : public BSDF
	{
	public:
		explicit AnisoPhase(const PropertyList &propList)
		{
			m_g = propList.getFloat("g", 0);
		}
		NORI_OBJECT_DEFAULT_CLONE(AnisoPhase)
		NORI_OBJECT_DEFAULT_UPDATE(AnisoPhase)


		/// Evaluate the BRDF for the given pair of directions
		Color3f eval(const BSDFQueryRecord &bRec) const override
		{
			return Color3f(1.f);
		}

		/// Evaluate the sampling density of \ref sample() wrt. solid angles
		float pdf(const BSDFQueryRecord &bRec) const override
		{
			const auto cosTheta = Frame::cosTheta(Frame(bRec.wi).toLocal(bRec.wo));
			const auto g2 = m_g * m_g;
			return 0.25f / M_PI * (1 - g2) / std::pow(1 + g2 - 2 * m_g * cosTheta, 1.5f);
		}

		/// Sample the BRDF
		Color3f sample(BSDFQueryRecord &bRec,
		                       const Point2f &sample) const override
		{
			// TODO: sample from pdf
			bRec.wo = Warp::squareToUniformSphere(sample);

			return eval(bRec) / pdf(bRec) * Frame::cosTheta(bRec.wo);
		}

		std::string toString() const override
		{
			return tfm::format("AnisoPhase[\n"
			                   "  g = %f,\n"
			                   "]",
			                   m_g);
		}
#ifdef NORI_USE_IMGUI
		NORI_OBJECT_IMGUI_NAME("Anisotropic Phase");
		bool getImGuiNodes() override
		{
			touched |= BSDF::getImGuiNodes();

			ImGui::AlignTextToFramePadding();
			ImGui::PushID(1);
			ImGui::TreeNodeEx("g", ImGuiLeafNodeFlags, "g");
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(-1);
			touched |= ImGui::DragFloat("##value", &m_g, 0.001f, -1, 1, "%f", ImGuiSliderFlags_AlwaysClamp);
			ImGui::NextColumn();
			ImGui::PopID();

			return touched;
		}
#endif

	private:
		float m_g;
	};

	NORI_REGISTER_CLASS(AnisoPhase, "anisophase");
NORI_NAMESPACE_END
