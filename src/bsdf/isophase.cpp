//
// Created by roger on 01/12/2020.
//

#include <nori/phase.h>
#include <nori/frame.h>
#include <nori/warp.h>

NORI_NAMESPACE_BEGIN

	/**
	 * Simple isotropic phase function for volumes. Scatters uniformly in all directions.
	 */
	struct IsoPhase : public PhaseFunction
	{
		explicit IsoPhase(const PropertyList &propList) {}
		NORI_OBJECT_DEFAULT_CLONE(IsoPhase)
		NORI_OBJECT_DEFAULT_UPDATE(IsoPhase)

		float pdf(const PhaseQueryRecord &bRec) const override
		{
			return 0.25f / M_PI;
		}

		Color3f sample(PhaseQueryRecord &bRec, const Point2f &sample) const override
		{
			bRec.wo = Warp::squareToUniformSphere(sample);

			// TODO: do we need cosine term?
			return 1.f / pdf(bRec) * Frame::cosTheta(bRec.wo);
		}

		std::string toString() const override
		{
			return tfm::format("IsoPhase[]");
		}
#ifdef NORI_USE_IMGUI
		NORI_OBJECT_IMGUI_NAME("Isotropic Phase");
		virtual bool getImGuiNodes() override {
			return PhaseFunction::getImGuiNodes();
		}
#endif
	};

	NORI_REGISTER_CLASS(IsoPhase, "isophase");
NORI_NAMESPACE_END
