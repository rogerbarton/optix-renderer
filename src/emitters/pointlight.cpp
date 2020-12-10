#include <nori/emitter.h>

#ifdef NORI_USE_OPTIX
#include <nori/optix/sutil/host_vec_math.h>
#endif

NORI_NAMESPACE_BEGIN

	/**
	 * http://www.pbr-book.org/3ed-2018/Light_Sources/Point_Lights.html
	 */
	class PointLight : public Emitter
	{
	public:
		explicit PointLight(const PropertyList &propList)
		{
			m_power    = propList.getColor("power"); // we store power, not intensity
			m_position = propList.getPoint3("position");
			lightProb = propList.getFloat("lightWeight", 1.f);
		}

		NoriObject *cloneAndInit() override
		{
			auto clone = new PointLight(*this);
			Emitter::cloneAndInit(clone);
			return clone;
		}

		void update(const NoriObject *guiObject) override
		{
			const auto *gui = static_cast<const PointLight *>(guiObject);
			if (!gui->touched)return;
			gui->touched = false;

			m_position = gui->m_position;
			m_power    = gui->m_power;
			m_radiance = m_power / (4 * M_PI);
			Emitter::update(guiObject);
		}

		Color3f sample(EmitterQueryRecord &lRec,
		               const Point2f &sample) const override
		{
			// store sample data inside the query record
			// shadow ray is ray from the light to the original first intersection
			// add a little bit to the initial position (move a bit in
			// the direction) because of collision with the object itself
			lRec.shadowRay = Ray3f(this->m_position, (lRec.ref - m_position).normalized(), Epsilon,
			                       (lRec.ref - m_position).norm() - Epsilon);
			lRec.wi        = (m_position - lRec.ref).normalized(); // pointing to light
			lRec.pdf       = pdf(lRec);                           // 1.0 in this case
			// don't store the normal, because this does not make sense for a point
			// light?

			// based on emitter.h comment on sample function
			return eval(lRec) / pdf(lRec);
		}

		Color3f eval(const EmitterQueryRecord &lRec) const override
		{
			// intensity divided by squared distance
			// since we store power, we need to convert to intensity (solid angles)
			Color3f Color =
					        m_radiance / (lRec.ref - m_position).squaredNorm();
			return Color3f(Color.x(), Color.y(), Color.z());
		}

		float pdf(const EmitterQueryRecord &lRec) const override
		{
			// http://www.pbr-book.org/3ed-2018/Light_Sources/Point_Lights.html
			return 1.f; // all directions uniform, taken from webpage of book
		}

		virtual std::string toString() const override
		{
			return tfm::format("PointLight[\n"
			                   "  power = %s,\n"
			                   "  position = %s,\n"
			                   "]",
			                   m_power.toString(), m_position.toString());
		}
#ifdef NORI_USE_IMGUI
		NORI_OBJECT_IMGUI_NAME("Pointlight");
		virtual bool getImGuiNodes() override
		{
			touched |= Emitter::getImGuiNodes();

			ImGui::PushID(EEmitter);

			ImGui::AlignTextToFramePadding();
			ImGui::TreeNodeEx("position", ImGuiLeafNodeFlags, "Position");
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(-1);
			touched |= ImGui::DragPoint3f("##position", &m_position, 0.1f);
			ImGui::NextColumn();
			ImGui::PopID();

			ImGui::AlignTextToFramePadding();
			ImGui::TreeNodeEx("power", ImGuiLeafNodeFlags, "Power");
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(-1);
			touched |= ImGui::DragColor3f("##power", &m_power, 1, 0, SLIDER_MAX_FLOAT, "%.3f",
			                          ImGuiSliderFlags_AlwaysClamp);
			ImGui::NextColumn();

			return touched;
		}
#endif

#ifdef NORI_USE_OPTIX
		void getOptixEmitterData(EmitterData &sbtData) override
		{
			sbtData.type = EmitterData::POINT;
			sbtData.point.position = make_float3(m_position);

			Emitter::getOptixEmitterData(sbtData);
		}
#endif

	protected:
		Point3f m_position;
		Color3f m_power;
	};

	NORI_REGISTER_CLASS(PointLight, "point");
NORI_NAMESPACE_END