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
			// store the radiance
			m_radiance = props.getColor("radiance");
		}

		NoriObject *cloneAndInit() override {
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
			// we need a shape for the arealight to work
			if (!m_shape)
				throw NoriException("There is no shape attached to this Area light!");
			// check the normal if we are on the back
			// we use -lRec.wi because wi goes into the emitter (convention)
			if (lRec.n.dot(-lRec.wi) < 0.f)
			{
				return Color3f(0.f); // we are on the back, return black
			}
			else
			{
				return m_radiance; // we are on the front, return the radiance
			}
		}

		virtual Color3f sample(EmitterQueryRecord &lRec,
		                       const Point2f &sample) const override
		{
			if (!m_shape)
				throw NoriException("There is no shape attached to this Area light!");

			// sample the surface using a shapeQueryRecord
			ShapeQueryRecord sqr(lRec.ref);
			// TODO: sample volume
			m_shape->sampleSurface(sqr, sample);

			// create an emitter query
			// we create a new one because we do not want to change the existing one
			// until we actually should return a color
			lRec = EmitterQueryRecord(sqr.ref, sqr.p, sqr.n);
			lRec.shadowRay = Ray3f(lRec.p, -lRec.wi, Epsilon, (lRec.p - lRec.ref).norm() - Epsilon);

			// compute the pdf of this query
			float probs = pdf(lRec);
			lRec.pdf = probs;

			// check for it being near zero
			if (std::abs(probs) < Epsilon)
			{
				return Color3f(0.f);
			}

			// return radiance
			return eval(lRec) / probs;
		}

		virtual float pdf(const EmitterQueryRecord &lRec) const override
		{
			if (!m_shape)
				throw NoriException("There is no shape attached to this Area light!");

			// if we are on the back, return 0
			if (lRec.n.dot(-lRec.wi) < 0.f)
			{
				return 0.f;
			}
			// create a shape query record and get the pdf of the surface
			// create by reference and sampled point
			ShapeQueryRecord sqr(lRec.ref, lRec.p);
			// TODO: use pdf volume
			float            prob = m_shape->pdfSurface(sqr);

			// transform the probability to solid angle
			// where the first part is the distance and
			// the second part is the cosine (computed using the normal)
			// taken from the slides (converts pdf to solid anggles)
			return prob * (lRec.p - lRec.ref).squaredNorm() / std::abs(lRec.n.dot(-lRec.wi));
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

			ImGui::AlignTextToFramePadding();
			ImGui::TreeNodeEx("Radiance", ImGuiLeafNodeFlags, "Radiance");
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(-1);

			touched |= ImGui::DragColor3f("##value", &m_radiance, 0.1f, 0, SLIDER_MAX_FLOAT, "%.3f",
			                              ImGuiSliderFlags_AlwaysClamp);
			ImGui::NextColumn();
			return touched;
		}
#endif
	protected:
		Color3f m_radiance;
	};

	NORI_REGISTER_CLASS(VolumeEmitter, "volumelight")
NORI_NAMESPACE_END