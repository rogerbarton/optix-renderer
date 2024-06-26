//
// Created by roger on 05/12/2020.
//

#include <nori/emitter.h>
#include <nori/shape.h>
#include <nori/warp.h>

NORI_NAMESPACE_BEGIN

	class VolumeEmitter : public Emitter
	{
	public:
		explicit VolumeEmitter(const PropertyList &props)
		{
			m_radiance = props.getColor("radiance");
			lightProb = props.getFloat("lightWeight", 1.f);
		}

		NoriObject *cloneAndInit() override {
			if (!m_shape || !m_shape->getMedium())
				throw NoriException("There is no shape or medium attached to this volume light.");

			auto clone = new VolumeEmitter(*this);
			Emitter::cloneAndInit(clone);
			return clone;
		}

		void update(const NoriObject *guiObject) override
		{
			const auto* gui = static_cast<const VolumeEmitter *>(guiObject);
			if (!gui->touched)return;
			gui->touched = false;

			m_radiance = gui->m_radiance;
			Emitter::update(guiObject);
		}

		virtual std::string toString() const override
		{
			return tfm::format("VolumeEmitter[\n"
			                   "  radiance = %s,\n"
			                   "]",
			                   m_radiance.toString());
		}

		virtual Color3f eval(const EmitterQueryRecord &lRec) const override
		{
			return m_radiance;
		}

		virtual Color3f sample(EmitterQueryRecord &lRec,
		                       const Point2f &sample2) const override
		{
			static Sampler *const sampler = static_cast<Sampler *>(
					NoriObjectFactory::createInstance("independent"));
			Point3f sample = Point3f(sample2.x(), sample2.y(), sampler->next1D());

			// Sample a point on the mesh from the reference point
			ShapeQueryRecord sRec{lRec.ref};
			m_shape->sampleVolume(sRec, sample);

			lRec.p = sRec.p;
			lRec.wi = (lRec.p - lRec.ref).normalized();
			lRec.n = -lRec.wi;
			lRec.shadowRay = Ray3f(lRec.ref, lRec.wi, Epsilon, (lRec.ref - lRec.p).norm() - Epsilon);

			lRec.pdf = pdf(lRec);

			return lRec.pdf == 0 ? Color3f(0.f) : (eval(lRec) / lRec.pdf).eval();
		}

		virtual float pdf(const EmitterQueryRecord &lRec) const override
		{
			ShapeQueryRecord sRec{lRec.ref, lRec.p};
			return m_shape->pdfVolume(sRec) * (lRec.p - lRec.ref).squaredNorm();
		}

		virtual Color3f samplePhoton(Ray3f &ray, const Point2f &sample1,
		                             const Point2f &sample2) const override
		{
			throw NoriException("Not implemented.");
		}

#ifdef NORI_USE_IMGUI
		NORI_OBJECT_IMGUI_NAME("Volume Light");
		virtual bool getImGuiNodes() override
		{
			touched |= Emitter::getImGuiNodes();
			return touched;
		}
#endif

#ifdef NORI_USE_OPTIX
		void getOptixEmitterData(EmitterData &sbtData) override
		{
			sbtData.type = EmitterData::VOLUME;
			// No specific data
			Emitter::getOptixEmitterData(sbtData);
		}
#endif
	};

	NORI_REGISTER_CLASS(VolumeEmitter, "volumelight")
NORI_NAMESPACE_END