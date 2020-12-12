//
// Created by roger on 01/12/2020.
//

#include <nori/medium.h>
#include <nori/NvdbVolume.h>

#ifdef NORI_USE_OPTIX
#include <nori/optix/sutil/host_vec_math.h>
#endif

NORI_NAMESPACE_BEGIN

	struct HeterogeneousMedium : public Medium
	{
		explicit HeterogeneousMedium(const PropertyList &propList) : Medium(propList)
		{
			m_densityScale = propList.getFloat("densityScale", 1.f);
		}

		NoriObject *cloneAndInit() override
		{
			auto clone = new HeterogeneousMedium(*this);
			Medium::cloneAndInit(clone);
			return clone;
		}

		void update(const NoriObject *guiObject) override
		{
			const auto *gui = static_cast<const HeterogeneousMedium *>(guiObject);
			if (!gui->touched)return;
			gui->touched = false;

			Medium::update(guiObject);
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

		float sampleFreePath(MediumQueryRecord &mRec, const Point2f &sample) const override
		{
			return 1.f;
		}

		Color3f getTransmittance(const Vector3f &from, const Vector3f &to) const override
		{
			return 0.f;
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
			TODO
			sbtData.type          = MediumData::HETEROG;

			Medium::getOptixMediumData(sbtData);
		}
#endif

		NvdbVolume *m_volume;

		float m_densityScale;
		float m_invMaxDensity;
	};

	NORI_REGISTER_CLASS(HeterogeneousMedium, "heterog");
NORI_NAMESPACE_END