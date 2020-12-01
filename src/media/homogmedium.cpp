//
// Created by roger on 01/12/2020.
//

#include <nori/medium.h>

NORI_NAMESPACE_BEGIN

	struct HomogeneousMedium : public Medium
	{
		explicit HomogeneousMedium(const PropertyList &propList)
		{
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

			// TODO: copy properties from gui, e.g.:
			//  m_position = gui->m_position;

			Medium::update(guiObject);
		}

		float sampleTr(MediumQueryRecord &mRec, const Point2f &sample) const override
		{
			return 0;
		}

		Color3f getTransmittance(const Vector3f &from, const Vector3f &to) const override
		{
			return 1;
		}

		std::string toString() const override { return "HomogeneousMedium[]"; }

#ifdef NORI_USE_IMGUI
		NORI_OBJECT_IMGUI_NAME("Homogeneous");
		virtual bool getImGuiNodes() override
		{
			touched |= Medium::getImGuiNodes();
			// TODO:

			return touched;
		}
#endif
	};

	NORI_REGISTER_CLASS(HomogeneousMedium, "homog");
NORI_NAMESPACE_END