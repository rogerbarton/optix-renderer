//
// Created by roger on 01/12/2020.
//

#include <nori/medium.h>

#ifdef NORI_USE_OPTIX
#include <nori/optix/sutil/host_vec_math.h>
#endif

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
			m_density            = propList.getFloat("density", 1.f);
		}

		NoriObject *cloneAndInit() override
		{
			// Calculate derived properties for the first time
			updateDerivedProperties();

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
			m_mu_a      = gui->m_mu_a;
			m_mu_s      = gui->m_mu_s;
			m_mu_t      = gui->m_mu_t;
			m_density   = gui->m_density;
			m_albedo    = gui->m_albedo;
			m_albedoInv = gui->m_albedoInv;

			Medium::update(guiObject);
		}

		void updateDerivedProperties()
		{
			m_mu_a     = m_density * m_sigma_a_normalized * m_sigma_a_intensity;
			m_mu_s     = m_density * m_sigma_s_normalized * m_sigma_s_intensity;
			m_mu_t     = m_mu_a + m_mu_s;
			for (int i = 0; i < 3; ++i)
				m_albedo(i) = m_mu_t(i) > Epsilon ? m_mu_s(i) / m_mu_t(i) : 0.f;
			m_albedoInv    = 1.f - m_albedo;
			m_mu_t_invNorm = (1 - m_mu_t).matrix().normalized().array();
		}

		float sampleFreePath(MediumQueryRecord &mRec, const Point1f &sample) const override
		{
			// Sample proportional to transmittance
			// TODO: check this
			return m_mu_t.maxCoeff() < Epsilon ? INFINITY :
			       -std::log(sample.x()) / m_mu_t.maxCoeff();
		}

		Color3f getTransmittance(const Vector3f &from, const Vector3f &to) const override
		{
			return (-m_mu_t * (from - to).norm()).array().exp();
		}

		std::string toString() const override
		{
			return tfm::format(
					"HomogeneousMedium[\n"
					"  sigma_a = %f,\n"
					"  sigma_s = %f,\n"
					"]",
					m_mu_a,
					m_mu_s);
		}

#ifdef NORI_USE_IMGUI
		NORI_OBJECT_IMGUI_NAME("Homogeneous");
		virtual bool getImGuiNodes() override
		{
			touched |= Medium::getImGuiNodes();

			ImGui::AlignTextToFramePadding();
			ImGui::TreeNodeEx("sigma_a", ImGuiLeafNodeFlags, "Absorption (sigma_a)");
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(-1);
			ImGui::PushItemWidth(ImGui::CalcItemWidth() * 0.75f);
			touched |= ImGui::ColorPicker3("##sigma_a", reinterpret_cast<float *>(&m_sigma_a_normalized),
			                               ImGuiClampColorEditFlags);
			ImGui::NextColumn();
			ImGui::TreeNodeEx("sigma_a_intensity", ImGuiLeafNodeFlags, "Intensity");
			ImGui::Text("");
			ImGui::NextColumn();
			touched |= ImGui::DragFloat("##sigma_a_intensity", &m_sigma_a_intensity, 0.001f, 0.f, SLIDER_MAX_FLOAT);
			ImGui::NextColumn();

			ImGui::AlignTextToFramePadding();
			ImGui::TreeNodeEx("sigma_s", ImGuiLeafNodeFlags, "Scattering (sigma_s)");
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(-1);
			ImGui::PushItemWidth(ImGui::CalcItemWidth() * 0.75f);
			touched |= ImGui::ColorPicker3("##sigma_s", reinterpret_cast<float *>(&m_sigma_s_normalized),
			                               ImGuiClampColorEditFlags);
			ImGui::NextColumn();
			ImGui::TreeNodeEx("sigma_s_intensity", ImGuiLeafNodeFlags, "Intensity");
			ImGui::NextColumn();
			touched |= ImGui::DragFloat("##sigma_s_intensity", &m_sigma_s_intensity, 0.001f, 0.f, SLIDER_MAX_FLOAT);
			ImGui::NextColumn();

			ImGui::TreeNodeEx("density", ImGuiLeafNodeFlags, "Density");
			ImGui::NextColumn();
			touched |= ImGui::DragFloat("##density", &m_density, 0.001f, 0.f, SLIDER_MAX_FLOAT);
			ImGui::NextColumn();

			// -- Update and display derived properties
			if (touched)
				updateDerivedProperties();
			const Color3f meanFreePath = m_mu_t.cwiseInverse();

			ImGui::AlignTextToFramePadding();
			ImGui::TreeNodeEx("mu_t", ImGuiLeafNodeFlags, "Extinction (mu_t)");
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(-1);
			ImGui::Text("%s", m_mu_t.toString().c_str());
			ImGui::NextColumn();

			ImGui::AlignTextToFramePadding();
			ImGui::TreeNodeEx("freepath", ImGuiLeafNodeFlags, "Mean free path");
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(-1);
			ImGui::Text("%s", meanFreePath.toString().c_str());
			ImGui::NextColumn();

			ImGui::AlignTextToFramePadding();
			ImGui::TreeNodeEx("albedo", ImGuiLeafNodeFlags, "Albedo (Scatter ratio)");
			ImGui::SameLine();
			ImGui::HelpMarker("Fraction of photons that scatter (vs are absorbed)");
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(-1);
			ImGui::TextColored(ImVec4(m_albedo.x(), m_albedo.y(), m_albedo.z(), 1.f), "%s",
			                   m_albedo.toString().c_str());
			ImGui::NextColumn();

			ImGui::AlignTextToFramePadding();
			ImGui::TreeNodeEx("albedo_inv", ImGuiLeafNodeFlags, "Albedo Inverse (Absorb ratio)");
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(-1);
			ImGui::TextColored(ImVec4(m_albedoInv.x(), m_albedoInv.y(), m_albedoInv.z(), 1.f), "%s",
			                   m_albedoInv.toString().c_str());
			ImGui::NextColumn();

			ImGui::AlignTextToFramePadding();
			ImGui::TreeNodeEx("m_mu_t_invNorm", ImGuiLeafNodeFlags, "Color (mu_t inverse normalized)");
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(-1);
			ImGui::TextColored(ImVec4(m_mu_t_invNorm.x(), m_mu_t_invNorm.y(), m_mu_t_invNorm.z(), 1.f), "%s",
			                   m_mu_t_invNorm.toString().c_str());
			ImGui::NextColumn();

			return touched;
		}
#endif

#ifdef NORI_USE_OPTIX
		void getOptixMediumData(MediumData &sbtData) override
		{
			sbtData.type          = MediumData::HOMOG;
			sbtData.homog.mu_a    = make_float3(m_mu_a);
			sbtData.homog.mu_s    = make_float3(m_mu_s);
			sbtData.homog.mu_t    = make_float3(m_mu_t);
			sbtData.homog.density = m_density;

			Medium::getOptixMediumData(sbtData);
		}
#endif

		Color3f m_mu_a;                 // absorption coefficient (pdf) [m^-1]
		Color3f m_mu_s;                 // scattering coefficient (pdf) [m^-1]
		float   m_density;              // rho [m^-3]

		// Gui-only properties
		Color3f m_sigma_a_normalized;   // cross-sectional absorption area [m^2]
		Color3f m_sigma_s_normalized;   // cross-sectional scattering area [m^2]
		float   m_sigma_a_intensity;    // scaling factor
		float   m_sigma_s_intensity;    // scaling factor
		Color3f m_mu_t_invNorm;         // 1 - mu_t normalized, apparent color
		Color3f m_albedo;               // fraction of photons that scattered (vs are absorbed) []
		Color3f m_albedoInv;            // fraction of photons that are absorbed (vs scattered), 1 - albedo []

		// Derived properties
		Color3f m_mu_t;                 // extinction coefficient (pdf) [m^-1]
	};

	NORI_REGISTER_CLASS(HomogeneousMedium, "homog");
NORI_NAMESPACE_END