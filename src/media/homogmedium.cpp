//
// Created by roger on 01/12/2020.
//

#include <nori/medium.h>

NORI_NAMESPACE_BEGIN

	struct HomogeneousMedium : public Medium
	{
		explicit HomogeneousMedium(const PropertyList &propList)
		{
			m_sigma_a = propList.getFloat("sigma_a", 0.5f);
			m_sigma_s = propList.getFloat("sigma_s", 0); // Beer's law medium as default
		}

		NoriObject *cloneAndInit() override
		{
			auto clone = new HomogeneousMedium(*this);
			Medium::cloneAndInit(clone);
			return clone;
		}

		void update(const NoriObject *guiObject) override
		{
			const auto *gui = static_cast<const HomogeneousMedium *>(guiObject);
			if (!gui->touched)return;
			gui->touched = false;

			m_sigma_a = gui->m_sigma_a;
			m_sigma_s = gui->m_sigma_s;

			Medium::update(guiObject);

			// Update derived properties
			m_sigma_t = m_sigma_a + m_sigma_s;
			m_albedo  = m_sigma_t > Epsilon ? m_sigma_s / m_sigma_t : 0.f;
		}

		float sampleFreePath(MediumQueryRecord &mRec, const Point1f &sample) const override
		{
			// Sample proportional to transmittance
			// TODO: check this
			return sample.x() < Epsilon ? INFINITY :
			       -std::log(sample.x()) / m_sigma_t;
		}

		float getTransmittance(const Vector3f &from, const Vector3f &to) const override
		{
			return std::exp(-m_sigma_t * (from - to).norm());
		}

		std::string toString() const override
		{
			return tfm::format(
					"HomogeneousMedium[\n"
					"  sigma_a = %f,\n"
					"  sigma_s = %f,\n"
					"]",
					m_sigma_a,
					m_sigma_s);
		}

#ifdef NORI_USE_IMGUI
		NORI_OBJECT_IMGUI_NAME("Homogeneous");
		virtual bool getImGuiNodes() override
		{
			touched |= Medium::getImGuiNodes();

			ImGui::AlignTextToFramePadding();
			ImGui::TreeNodeEx("sigma_a", ImGuiLeafNodeFlags, "Sigma_a (Absorption)");
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(-1);
			touched |= ImGui::DragFloat("##sigma_a", &m_sigma_a, 0.01f, 0, SLIDER_MAX_FLOAT, "%.3f",
			                            ImGuiSliderFlags_AlwaysClamp);
			ImGui::NextColumn();

			ImGui::AlignTextToFramePadding();
			ImGui::TreeNodeEx("sigma_s", ImGuiLeafNodeFlags, "Sigma_s (Scattering)");
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(-1);
			touched |= ImGui::DragFloat("##sigma_s", &m_sigma_s, 0.01f, 0, SLIDER_MAX_FLOAT, "%.3f",
			                            ImGuiSliderFlags_AlwaysClamp);
			ImGui::NextColumn();

			// -- Display derived properties
			m_sigma_t = m_sigma_a + m_sigma_s;
			m_albedo  = m_sigma_t > Epsilon ? m_sigma_s / m_sigma_t : 0.f;

			ImGui::AlignTextToFramePadding();
			ImGui::TreeNodeEx("sigma_t", ImGuiLeafNodeFlags, "Sigma_t (Extinction)");
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(-1);
			ImGui::Text("%f", m_sigma_t);
			ImGui::NextColumn();

			ImGui::AlignTextToFramePadding();
			ImGui::TreeNodeEx("freepath", ImGuiLeafNodeFlags, "Mean free path");
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(-1);
			ImGui::Text("%f", 1.f / m_sigma_t);
			ImGui::NextColumn();

			ImGui::AlignTextToFramePadding();
			ImGui::TreeNodeEx("albedo", ImGuiLeafNodeFlags, "Albedo");
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(-1);
			ImGui::Text("%f", m_albedo);
			ImGui::NextColumn();

			return touched;
		}
#endif

		float m_sigma_a;
		float m_sigma_s;

		// Derived properties
		float m_sigma_t;
		float m_albedo;
	};

	NORI_REGISTER_CLASS(HomogeneousMedium, "homog");
NORI_NAMESPACE_END