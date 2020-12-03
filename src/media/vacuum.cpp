//
// Created by roger on 02/12/2020.
//

#include <nori/medium.h>

NORI_NAMESPACE_BEGIN

	struct VacuumMedium : public Medium
	{
		explicit VacuumMedium(const PropertyList &propList) {}

		NoriObject *cloneAndInit() override
		{
			auto clone = new VacuumMedium(*this);
			Medium::cloneAndInit(clone);
			return clone;
		}

		void update(const NoriObject *guiObject) override
		{
			const auto *gui = static_cast<const VacuumMedium *>(guiObject);
			if (!gui->touched)return;
			gui->touched = false;

			Medium::update(guiObject);
		}

		float sampleFreePath(MediumQueryRecord &mRec, const Point1f &sample) const override
		{
			return INFINITY;
		}

		Color3f getTransmittance(const Vector3f &from, const Vector3f &to) const override
		{
			return 1.f;
		}

		std::string toString() const override { return "VacuumMedium[]"; }

#ifdef NORI_USE_IMGUI
		NORI_OBJECT_IMGUI_NAME("Vacuum");
		virtual bool getImGuiNodes() override
		{
			touched |= Medium::getImGuiNodes();
			return touched;
		}
#endif
	};

	NORI_REGISTER_CLASS(VacuumMedium, "vacuum");
NORI_NAMESPACE_END