#pragma once

#include <nori/emitter.h>
#include <nori/texture.h>
#include <Eigen/Geometry>
#include <nori/warp.h>
#include <nori/dpdf.h>

NORI_NAMESPACE_BEGIN

	class EnvMap : public Emitter
	{
	public:
		explicit EnvMap(const PropertyList &props);
		~EnvMap() override;
		NoriObject *cloneAndInit() override;
		void update(const NoriObject *guiObject) override;
		void addChild(NoriObject *obj) override;

		Color3f sample(EmitterQueryRecord &lRec,
		               const Point2f &sample) const override;

		float pdf(const EmitterQueryRecord &lRec) const override;

		Color3f eval(const EmitterQueryRecord &lRec) const override;

		Color3f samplePhoton(Ray3f &ray, const Point2f &sample1, const Point2f &sample2) const override;

		bool isEnvMap() const override { return true; }

		std::string toString() const override;
#ifdef NORI_USE_IMGUI
		NORI_OBJECT_IMGUI_NAME("Environment Map");
		bool getImGuiNodes() override;
#endif


#ifdef NORI_USE_OPTIX
		void getOptixEmitterData(EmitterData &sbtData) override;
#endif
	private:
		void calculateProbs();

		Texture<Color3f> *m_map = nullptr;
		DiscretePDF      dpdf;
	};
NORI_NAMESPACE_END
