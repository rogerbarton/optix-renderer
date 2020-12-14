//
// Created by roger on 01/12/2020.
//

#include <nori/medium.h>
#include <nori/sampler.h>
#include <nori/NvdbVolume.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/Ray.h>

#ifdef NORI_USE_OPTIX
#include <nori/optix/sutil/host_vec_math.h>
#endif

NORI_NAMESPACE_BEGIN
	inline Color3f make_color(const nanovdb::Vec3f &v)
	{
		return Color3f(v[0], v[2], v[3]);
	}

	inline Point3f make_point(const nanovdb::Vec3f &v)
	{
		return Point3f(v[0], v[2], v[3]);
	}

	inline nanovdb::Vec3f make_nvec(const Point3f &v)
	{
		return nanovdb::Vec3f(v.x(), v.y(), v.z());
	}

	inline nanovdb::Vec3f make_nvec(const Color3f &v)
	{
		return nanovdb::Vec3f(v.x(), v.y(), v.z());
	}

	template<typename ValueT>
	inline Color3f colorFromTemperature(const ValueT &v, float scale)
	{
		// NanoVDB: RenderFogVolumeUtils.h
		float r = v;
		float g = r * r;
		float b = g * g;
		return scale * Color3f(r * r * r, g * g * g, b * b * b);
	}

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
			// Convert ray from world to index and skip to bbox
			nanovdb::Ray<float> wRay(make_nvec(mRec.ray.o), make_nvec(mRec.ray.d), mRec.ray.mint, mRec.ray.maxt);
			float               mint, maxt;
			if (!wRay.intersects(m_volume->densityGrid->worldBBox(), mint, maxt))
				return INFINITY;

			if (mint > mRec.ray.mint)
				wRay.setMinTime(mint);
			if (maxt < mRec.ray.maxt)
				wRay.setMaxTime(maxt);
			nanovdb::Ray<float> iRay = wRay.worldToIndexF(*m_volume->densityGrid);

			float t = iRay.t0();
			do
			{
				// step to next null collision in homogeneous medium
				t += -std::log(sampler.next1D()) * m_densityMaxInv / m_sigma_t.maxCoeff();
			} while (t < iRay.t1() &&
			         // Check if this is a real collision
			         m_volume->getDensity(iRay(t)) * m_densityMaxInvUnscaled < sampler.next1D());

			return t;
		}

		Color3f getTransmittance(const Vector3f &from, const Vector3f &to, const bool &scattered,
		                         Sampler &sampler) const override
		{
			// NanoVDB: RenderFogVolumeUtils.h
			// Convert ray from world to index
			Ray3f               noriRay(from, (to - from).normalized(), Epsilon, (to - from).norm());
			nanovdb::Ray<float> wRay(make_nvec(noriRay.o), make_nvec(noriRay.d), noriRay.mint, noriRay.maxt);
			float               mint, maxt;
			if (!wRay.intersects(m_volume->densityGrid->worldBBox(), mint, maxt))
				return INFINITY;

			if (mint > noriRay.mint)
				wRay.setMinTime(mint);
			if (maxt < noriRay.maxt)
				wRay.setMaxTime(maxt);
			nanovdb::Ray<float> iRay = wRay.worldToIndexF(*m_volume->densityGrid);

			assert(iRay.clip(m_volume->densityGrid->worldBBox()));

			Color3f transmittance = 0.f;
			float   t             = iRay.t0();
			nanovdb::Vec3f p;

			do
			{
				// step to next null collision in homogeneous medium
				t += -std::log(sampler.next1D()) * m_densityMaxInv / m_sigma_t.maxCoeff();

				// Use multiplicative transmittance
				p = iRay(t);
				transmittance *= 1.f - m_volume->getDensity(p) * m_densityMaxInv;

				// emission
				if (m_temperatureScale > Epsilon)
					transmittance *= colorFromTemperature(m_volume->getTemperature(p), m_temperatureScale);
			} while (t < iRay.t1() &&
			         // Check if this is a real collision
			         m_volume->getDensity(p) * m_densityMaxInvUnscaled < sampler.next1D());


			return transmittance;
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