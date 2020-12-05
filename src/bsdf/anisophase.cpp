//
// Created by roger on 01/12/2020.
//

#include <nori/phase.h>
#include <nori/frame.h>
#include <nori/warp.h>

NORI_NAMESPACE_BEGIN

	/**
	 * Anisotropic phase function for volumes based on Henyey-Greenstein.
	 */
	struct AnisoPhase : public PhaseFunction
	{
		explicit AnisoPhase(const PropertyList &propList)
		{
			m_g = propList.getFloat("g", 0);
		}
		NORI_OBJECT_DEFAULT_CLONE(AnisoPhase)
		NORI_OBJECT_DEFAULT_UPDATE(AnisoPhase)

		/// Evaluate the sampling density of \ref sample() wrt. solid angles
		float pdf(const PhaseQueryRecord &bRec) const override
		{
			return Warp::squareToHenyeyGreensteinPdf(Frame(bRec.wi).toLocal(bRec.wo), m_g);
		}

		/// Sample the BRDF
		Color3f sample(PhaseQueryRecord &bRec, const Point2f &sample) const override
		{
			bRec.wo = Warp::squareToHenyeyGreenstein(sample, m_g);

			return 1.f / pdf(bRec) * abs(bRec.wi.dot(bRec.wo));
		}

		std::string toString() const override
		{
			return tfm::format("AnisoPhase[\n"
			                   "  g = %f,\n"
			                   "]",
			                   m_g);
		}
#ifdef NORI_USE_IMGUI
		NORI_OBJECT_IMGUI_NAME("Henyey-Greenstein (Anisotropic)");
		bool getImGuiNodes() override
		{
			touched |= PhaseFunction::getImGuiNodes();

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
