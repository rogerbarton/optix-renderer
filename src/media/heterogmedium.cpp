//
// Created by roger on 01/12/2020.
//

#include <nori/medium.h>
#include <nori/NvdbVolume.h>
#include <nanovdb/NanoVDB.h>
#include <nori/sampler.h>

#ifdef NORI_USE_OPTIX
#include <nori/optix/sutil/host_vec_math.h>
#endif

NORI_NAMESPACE_BEGIN

	struct HeterogeneousMedium : public Medium
	{
		explicit HeterogeneousMedium(const PropertyList &propList) : Medium(propList)
		{
			m_densityScale     = propList.getFloat("densityScale", 1.f);
			m_temperatureScale = propList.getFloat("temperatureScale", 0.f); // disable by default
		}

		NoriObject *cloneAndInit() override
		{
			if (!m_volume)
				throw NoriException("HeterogeneousMedium requires a nvdb volume.");

			updateDerivedProperties();

			auto clone = new HeterogeneousMedium(*this);
			clone->m_volume = static_cast<NvdbVolume *>(m_volume->cloneAndInit());
			Medium::cloneAndInit(clone);
			return clone;
		}

		void update(const NoriObject *guiObject) override
		{
			const auto *gui = static_cast<const HeterogeneousMedium *>(guiObject);
			if (!gui->touched)return;
			gui->touched = false;

			m_densityScale     = gui->m_densityScale;
			m_temperatureScale = gui->m_temperatureScale;
			m_volume->update(gui->m_volume);

			Medium::update(guiObject);

			m_densityMaxUnscaled    = m_volume->densityGrid->tree().root().valueMax();
			m_densityMaxInvUnscaled = 1.f / m_densityMaxUnscaled;
			m_densityMax            = std::max(0.001f, m_densityScale * m_densityMaxUnscaled);
			m_densityMaxInv         = 1.f / m_densityMax;
		}

		~HeterogeneousMedium() override
		{
			delete m_volume;
		}

		void addChild(NoriObject *obj) override
		{
			switch (obj->getClassType())
			{
				case EVolume:
					if (m_volume)
						throw NoriException("There is already an volume defined!");
					m_volume = static_cast<NvdbVolume *>(obj);
					break;
				default:
					throw NoriException("HeterogeneousMedium::addChild(<%s>) is not supported!",
					                    classTypeName(obj->getClassType()));
			}
		}

		float sampleFreePath(MediumQueryRecord &mRec, Sampler &sampler) const override
		{
			// NanoVDB: RenderFogVolumeUtils.h
			float t = mRec.ray.mint;
			do
			{
				t += -std::log(sampler.next1D()) * m_densityMaxInv;
			} while (t < mRec.ray.maxt &&
			         m_volume->getDensity(mRec.ray(t)) * m_densityMaxInvUnscaled < sampler.next1D());

			return t;
		}

		Color3f	getTransmittance(const Vector3f &from, const Vector3f &to, const bool &scattered,
		                 Sampler &sampler) const override
		{
			// NanoVDB: RenderFogVolumeUtils.h
			// delta tracking.
			// faster due to earlier termination, but we need multiple samples to reduce variance.
			const int nSamples      = 2;
			float     transmittance = 0.f;
			Ray3f     ray(from, (to - from).normalized(), Epsilon, (to - from).norm());

			for (int n = 0; n < nSamples; n++)
			{
				float t = ray.mint;
				while (true)
				{
					t -= std::log(sampler.next1D()) * m_densityMaxInv;
					if (t >= ray.maxt)
					{
						transmittance += 1.0f;
						break;
					}
					if (m_volume->getDensity(ray(t)) * m_densityMaxInvUnscaled >= sampler.next1D())
						break;
				}
			}

			// TODO: calculate a color
			return transmittance / nSamples;
		}

		std::string toString() const override
		{
			return tfm::format("HeterogeneousMedium[...]");
		}

#ifdef NORI_USE_IMGUI
		NORI_OBJECT_IMGUI_NAME("Heterogeneous");
		virtual bool getImGuiNodes() override
		{
			ImGui::PushID(EMedium);
			touched |= Medium::getImGuiNodes();

			NORI_IMGUI_CHILD_OBJECT(m_volume, "NanoVDB Volume")

			ImGui::PopID();

			return touched;
		}
#endif

#ifdef NORI_USE_OPTIX
		void getOptixMediumData(MediumData &sbtData) override
		{
			// TODO
			sbtData.type          = MediumData::HETEROG;
			sbtData.heterog.densityGrid = m_volume->densityGrid;
			sbtData.heterog.temperatureGrid = m_volume->temperatureGrid;
			Medium::getOptixMediumData(sbtData);
			throw NoriException("Heterog::getOptixMediumData not implemented!");
		}
#endif

		NvdbVolume *m_volume = nullptr;

		float m_densityScale;
		float m_temperatureScale;
		float m_densityMaxUnscaled;
		float m_densityMaxInvUnscaled;
		float m_densityMax;
		float m_densityMaxInv;
	};

	NORI_REGISTER_CLASS(HeterogeneousMedium, "heterog");
NORI_NAMESPACE_END