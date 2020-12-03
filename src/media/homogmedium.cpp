//
// Created by roger on 01/12/2020.
//

#include <nori/medium.h>

NORI_NAMESPACE_BEGIN

	struct HomogeneousMedium : public Medium
	{
		explicit HomogeneousMedium(const PropertyList &propList)
		{
			// Beer's law medium as default, sigma_s = 0
			m_sigma_a_normalized = propList.getColor("sigma_a", 0.5f);
			m_sigma_s_normalized = propList.getColor("sigma_s", 0);
			m_sigma_a_intensity  = propList.getFloat("sigma_a_intensity", 1.f);
			m_sigma_s_intensity  = propList.getFloat("sigma_s_intensity", 1.f);
		}

		NoriObject *cloneAndInit() override
		{
			// Calculate derived properties for the first time
			m_sigma_a  = m_sigma_a_normalized * m_sigma_a_intensity;
			m_sigma_s  = m_sigma_s_normalized * m_sigma_s_intensity;
			m_sigma_t  = m_sigma_a + m_sigma_s;
			for (int i = 0; i < 3; ++i)
				m_albedo(i) = m_sigma_t(i) > Epsilon ? m_sigma_s(i) / m_sigma_t(i) : 0.f;

			auto clone = new HomogeneousMedium(*this);
			Medium::cloneAndInit(clone);
			return clone;
		}

		void update(const NoriObject *guiObject) override
		{
			const auto *gui = static_cast<const HomogeneousMedium *>(guiObject);
			if (!gui->touched)return;
			gui->touched = false;

			// Only copy relevant properties
			m_sigma_a = gui->m_sigma_a;
			m_sigma_s = gui->m_sigma_s;
			m_sigma_t = gui->m_sigma_t;
			m_albedo  = gui->m_albedo;

			Medium::update(guiObject);
		}

		float sampleFreePath(MediumQueryRecord &mRec, const Point1f &sample) const override
		{
			// Sample proportional to transmittance
			// TODO: check this
			return sample.x() < Epsilon ? INFINITY :
			       -std::log(sample.x()) / m_sigma_t.maxCoeff();
		}

		Color3f getTransmittance(const Vector3f &from, const Vector3f &to) const override
		{
			return (-m_sigma_t * (from - to).norm()).array().exp();
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
			ImGui::PushItemWidth(ImGui::CalcItemWidth() * 0.75f);
			touched |= ImGui::ColorPicker3("##sigma_a", reinterpret_cast<float *>(&m_sigma_a_normalized),
			                               ImGuiClampColorEditFlags);
			ImGui::NextColumn();
			ImGui::TreeNodeEx("sigma_a_intensity", ImGuiLeafNodeFlags, "Intensity");
			ImGui::Text("");
			ImGui::NextColumn();
			touched |= ImGui::DragFloat("##sigma_a_intensity", &m_sigma_a_intensity, 0.01f, 0.f, SLIDER_MAX_FLOAT);
			ImGui::NextColumn();

			ImGui::AlignTextToFramePadding();
			ImGui::TreeNodeEx("sigma_s", ImGuiLeafNodeFlags, "Sigma_s (Scattering)");
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(-1);
			ImGui::PushItemWidth(ImGui::CalcItemWidth() * 0.75f);
			touched |= ImGui::ColorPicker3("##sigma_s", reinterpret_cast<float *>(&m_sigma_s_normalized),
			                               ImGuiClampColorEditFlags);
			ImGui::NextColumn();
			ImGui::TreeNodeEx("sigma_s_intensity", ImGuiLeafNodeFlags, "Intensity");
			ImGui::NextColumn();
			touched |= ImGui::DragFloat("##sigma_s_intensity", &m_sigma_s_intensity, 0.01f, 0.f, SLIDER_MAX_FLOAT);
			ImGui::NextColumn();

			// -- Update and display derived properties
			if (touched)
			{
				m_sigma_a = m_sigma_a_normalized * m_sigma_a_intensity;
				m_sigma_s = m_sigma_s_normalized * m_sigma_s_intensity;
				m_sigma_t = m_sigma_a + m_sigma_s;
				for (int i = 0; i < 3; ++i)
					m_albedo(i) = m_sigma_t(i) > Epsilon ? m_sigma_s(i) / m_sigma_t(i) : 0.f;
			}
			const Color3f meanFreePath = m_sigma_t.cwiseInverse();

			ImGui::AlignTextToFramePadding();
			ImGui::TreeNodeEx("sigma_t", ImGuiLeafNodeFlags, "Sigma_t (Extinction)");
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(-1);
			ImGui::Text("%s", m_sigma_t.toString().c_str());
			ImGui::NextColumn();

			ImGui::AlignTextToFramePadding();
			ImGui::TreeNodeEx("freepath", ImGuiLeafNodeFlags, "Mean free path");
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(-1);
			ImGui::Text("%s", meanFreePath.toString().c_str());
			ImGui::NextColumn();

			ImGui::AlignTextToFramePadding();
			ImGui::TreeNodeEx("albedo", ImGuiLeafNodeFlags, "Albedo");
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(-1);
			ImGui::TextColored(ImVec4(m_albedo.x(), m_albedo.y(), m_albedo.z(), 1.f), "%s", m_albedo.toString().c_str());
			ImGui::NextColumn();

			return touched;
		}
#endif

		Color3f m_sigma_a;
		Color3f m_sigma_s;

		// Gui-only properties
		Color3f m_sigma_a_normalized;
		Color3f m_sigma_s_normalized;
		float   m_sigma_a_intensity;
		float   m_sigma_s_intensity;

		// Derived properties
		Color3f m_sigma_t;
		Color3f m_albedo;
	};

	NORI_REGISTER_CLASS(HomogeneousMedium, "homog");
NORI_NAMESPACE_END