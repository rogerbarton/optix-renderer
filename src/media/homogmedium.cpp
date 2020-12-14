//
// Created by roger on 01/12/2020.
//

#include <nori/medium.h>
#include <nori/sampler.h>

#ifdef NORI_USE_OPTIX
#include <nori/optix/sutil/host_vec_math.h>
#endif

NORI_NAMESPACE_BEGIN

	struct HomogeneousMedium : public Medium
	{
		explicit HomogeneousMedium(const PropertyList &propList) : Medium(propList)
		{
			m_density = propList.getFloat("density", 1.f);
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

			Medium::update(guiObject);

			// Only copy relevant properties
			m_mu_a      = gui->m_mu_a;
			m_mu_s      = gui->m_mu_s;
			m_mu_t      = gui->m_mu_t;
			m_density   = gui->m_density;
			m_albedo    = gui->m_albedo;
			m_albedoInv = gui->m_albedoInv;
		}

		void updateDerivedProperties() override
		{
			Medium::updateDerivedProperties();

			m_mu_a     = m_density * m_sigma_a;
			m_mu_s     = m_density * m_sigma_s;
			m_mu_t     = m_mu_a + m_mu_s;
			for (int i = 0; i < 3; ++i)
				m_albedo(i) = m_mu_t(i) > Epsilon ? m_mu_s(i) / m_mu_t(i) : 0.f;
			m_albedoInv    = 1.f - m_albedo;
			m_mu_t_invNorm = (1 - m_mu_t).matrix().normalized().array();
		}

		float sampleFreePath(MediumQueryRecord &mRec, Sampler &sampler) const override
		{
			// Sample proportional to transmittance, sample a random channel uniformly
			const int sampledChannel = (int) (3 * sampler.next1D());
			return m_mu_t(sampledChannel) < Epsilon ? INFINITY :
			       -std::log(sampler.next1D()) / m_mu_t(sampledChannel);
		}

		Color3f
		getTransmittance(const Vector3f &from, const Vector3f &to, const bool &scattered, Sampler &sampler) const override
		{
			return (scattered ? m_mu_t : 1.f) *
			       (-m_mu_t * (from - to).norm()).array().exp();
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
			bool newlyTouched = false;
			ImGui::PushID(EMedium);
			touched |= Medium::getImGuiNodes();

			ImGui::TreeNodeEx("density", ImGuiLeafNodeFlags, "Density");
			ImGui::NextColumn();
			newlyTouched |= ImGui::DragFloat("##density", &m_density, 0.001f, 0.f, SLIDER_MAX_FLOAT);
			ImGui::NextColumn();

			// -- Update and display derived properties
			if (newlyTouched)
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

			ImGui::PopID();

			touched |= newlyTouched;
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
		Color3f m_mu_t;                 // extinction coefficient (pdf) [m^-1]
		float   m_density;              // rho [m^-3]

		Color3f m_albedo;               // fraction of photons that scattered (vs are absorbed) []
		Color3f m_albedoInv;            // fraction of photons that are absorbed (vs scattered), 1 - albedo []
		Color3f m_mu_t_invNorm;         // 1 - mu_t normalized, apparent color
	};

	NORI_REGISTER_CLASS(HomogeneousMedium, "homog");
NORI_NAMESPACE_END