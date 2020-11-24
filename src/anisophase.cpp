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
		virtual Color3f eval(const BSDFQueryRecord &bRec) const override
		{
			throw NoriException("Not implemented.");
		}

		/// Evaluate the sampling density of \ref sample() wrt. solid angles
		virtual float pdf(const BSDFQueryRecord &bRec) const override
		{
			throw NoriException("Not implemented.");
		}

		/// Sample the BRDF
		virtual Color3f sample(BSDFQueryRecord &bRec,
		                       const Point2f &_sample) const override
		{
			throw NoriException("Not implemented.");
		}

		virtual std::string toString() const override
		{
			return tfm::format("AnisoPhase[\n"
			                   "  g = %f,\n"
			                   "]",
			                   m_g);
		}
#ifndef NORI_USE_NANOGUI
		NORI_OBJECT_IMGUI_NAME("Anisotropic Phase");
		virtual bool getImGuiNodes() override
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
