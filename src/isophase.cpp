#include <nori/bsdf.h>
#include <nori/frame.h>
#include <nori/warp.h>

NORI_NAMESPACE_BEGIN

	/**
	 * Simple isotropic phase function for volumes. Scatters uniformly in all directions.
	 */
	class IsoPhase : public BSDF
	{
	public:
		explicit IsoPhase(const PropertyList &propList) {}
		NORI_OBJECT_DEFAULT_CLONE(IsoPhase)
		NORI_OBJECT_DEFAULT_UPDATE(IsoPhase)

		virtual Color3f eval(const BSDFQueryRecord &bRec) const override
		{
			return Color3f(1.f);
		}

		virtual float pdf(const BSDFQueryRecord &bRec) const override
		{
			return 0.25f / M_PI;
		}

		virtual Color3f sample(BSDFQueryRecord &bRec, const Point2f &sample) const override
		{
			bRec.wo = Warp::squareToUniformSphere(sample);

			return eval(bRec) / pdf(bRec) * Frame::cosTheta(bRec.wo);
		}

		virtual std::string toString() const override
		{
			return tfm::format("IsoPhase[]");
		}
#ifdef NORI_USE_IMGUI
		NORI_OBJECT_IMGUI_NAME("Isotropic Phase");
#endif
	};

	NORI_REGISTER_CLASS(IsoPhase, "isophase");
NORI_NAMESPACE_END
