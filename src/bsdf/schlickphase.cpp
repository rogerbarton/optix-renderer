//
// Created by roger on 03/12/2020.
//

#include <nori/phase.h>
#include <nori/frame.h>
#include <nori/warp.h>

NORI_NAMESPACE_BEGIN

	/**
	 * Anisotropic phase function for volumes based on Schlick, a fast approximation of Henyey-Greenstein.
	 */
	struct SchlickPhase : public PhaseFunction
	{
		explicit SchlickPhase(const PropertyList &propList)
		{
			m_g = propList.getFloat("g", 0);
			m_k = 1.55f * m_g - 0.55f * std::pow(m_g, 3);
		}
		NORI_OBJECT_DEFAULT_CLONE(SchlickPhase)
		NORI_OBJECT_DEFAULT_UPDATE(SchlickPhase)

		/// Evaluate the sampling density of \ref sample() wrt. solid angles
		float pdf(const PhaseQueryRecord &bRec) const override
		{
			return Warp::squareToSchlickPdf(bRec.wo, m_k);
		}

		/// Sample the BRDF
		void sample(PhaseQueryRecord &bRec, const Point2f &sample) const override
		{
			bRec.wo = Warp::squareToSchlick(sample, m_k);
		}

		std::string toString() const override
		{
			return tfm::format("SchlickPhase[\n"
			                   "  g = %f,\n"
			                   "]",
			                   m_g);
		}

#ifdef NORI_USE_IMGUI
		NORI_OBJECT_IMGUI_NAME("Schlick (Anisotropic)");
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

			// Update and display derived properties
			if(touched)
				m_k = 1.55f * m_g - 0.55f * std::pow(m_g, 3);

			ImGui::AlignTextToFramePadding();
			ImGui::TreeNodeEx("k", ImGuiLeafNodeFlags, "k");
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(-1);
			ImGui::Text("%f", m_k);
			ImGui::NextColumn();

			return touched;
		}
#endif

	private:
		float m_g;

		// derived properties
		float m_k;
	};

	NORI_REGISTER_CLASS(SchlickPhase, "schlick");
NORI_NAMESPACE_END
