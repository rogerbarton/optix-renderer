#include <nori/emitter.h>
#include <nori/warp.h>
#include <nori/shape.h>
#include <nori/mesh.h>

NORI_NAMESPACE_BEGIN
class SpotLight : public Emitter
{
public:
    SpotLight(const PropertyList &props)
    {
        m_position = props.getPoint3("position", Point3f());
        m_direction = props.getVector3("direction", Vector3f()).normalized();
        m_power = props.getColor("power", Color3f());
        cosFalloffStart = std::cos(degToRad(props.getFloat("falloffstart")));
        cosTotalWidth = std::cos(degToRad(props.getFloat("totalwidth")));
        m_coord = Frame(m_direction.normalized());
    }

    NoriObject *cloneAndInit() override
    {
        auto clone = new SpotLight(*this);
        Emitter::cloneAndInit(clone);
        return clone;
    }

    void update(const NoriObject *guiObject) override
    {
        const auto *gui = static_cast<const SpotLight *>(guiObject);
        if (!gui->touched)
            return;

        gui->touched = false;

        m_power = gui->m_power;
        m_direction = gui->m_direction;
        cosFalloffStart = gui->cosFalloffStart;
        cosTotalWidth = gui->cosTotalWidth;
        m_coord = gui->m_coord;

        Emitter::update(guiObject);
    }

    virtual Color3f sample(EmitterQueryRecord &lRec, const Point2f &sample) const override
    {
        lRec.wi = (m_position - lRec.ref).normalized();
        const float maxt = (m_position - lRec.ref).norm() - Epsilon;

        lRec.shadowRay = Ray3f(m_position, -lRec.wi, Epsilon, maxt);
        lRec.pdf = 1.f;
        lRec.p = m_position;

        return eval(lRec) / lRec.pdf;
    }

    virtual Color3f eval(const EmitterQueryRecord &lRec) const override
    {
        Color3f i = m_power / 2. / M_PI / (1.f - .5f * (cosTotalWidth + cosFalloffStart));
        Color3f color = i * falloff(-lRec.wi) / (lRec.ref - m_position).squaredNorm();
        return color;
    }

    virtual float pdf(const EmitterQueryRecord &lRec) const override
    {
        return 1.f;
    }

    virtual std::string toString() const override
    {
        return tfm::format(
            "Spotlight[\n"
            "  position = %s\n"
            "  power = %s\n"
            "  falloffStart = %f\n"
            "  totalwidth = %f\n"
            "]",
            m_position.toString(),
            m_power.toString(),
            cosFalloffStart,
            cosTotalWidth);
    };

#ifdef NORI_USE_IMGUI
    NORI_OBJECT_IMGUI_NAME("Spotlight");
    virtual bool getImGuiNodes() override
    {
        int counter = 0;

        touched |= Emitter::getImGuiNodes();

        ImGui::PushID(counter++);
        ImGui::AlignTextToFramePadding();
        ImGui::TreeNodeEx("power", ImGuiLeafNodeFlags, "Power");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);

        touched |= ImGui::DragColor3f("##value", &m_power, 0.1f, 0, SLIDER_MAX_FLOAT, "%.3f",
                                      ImGuiSliderFlags_AlwaysClamp);
        ImGui::NextColumn();
        ImGui::PopID();

        ImGui::PushID(counter++);
        ImGui::AlignTextToFramePadding();
        ImGui::TreeNodeEx("direction", ImGuiLeafNodeFlags, "Direction");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);

        bool dir_touched = ImGui::DragVector3f("##value", &m_direction, 0.1f, -1.f, 1.f, "%.3f",
                                                ImGuiSliderFlags_AlwaysClamp);
        touched |= dir_touched;
        if (dir_touched)
        {
            m_coord = Frame(m_direction.normalized());
        }
        ImGui::PopID();

        ImGui::NextColumn();

        ImGui::PushID(counter++);
        ImGui::AlignTextToFramePadding();
        ImGui::TreeNodeEx("cosTotalWidth", ImGuiLeafNodeFlags, "Cos Total Width");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);

        touched |= ImGui::DragFloat("##value", &cosTotalWidth, 0.1f, 0, SLIDER_MAX_FLOAT, "%.3f",
                                    ImGuiSliderFlags_AlwaysClamp);
        ImGui::NextColumn();
        ImGui::PopID();

        ImGui::PushID(counter++);
        ImGui::AlignTextToFramePadding();
        ImGui::TreeNodeEx("cosFalloffStart", ImGuiLeafNodeFlags, "Cos Falloff Start");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);

        touched |= ImGui::DragFloat("##value", &cosTotalWidth, 0.1f, 0, SLIDER_MAX_FLOAT, "%.3f",
                                    ImGuiSliderFlags_AlwaysClamp);
        ImGui::NextColumn();
        ImGui::PopID();

        return touched;
    }
#endif

private:
    Color3f m_power;
    Vector3f m_direction;
    float cosTotalWidth, cosFalloffStart;
    Frame m_coord;

    float falloff(const Vector3f &w) const
    {
        Vector3f wi = m_coord.toLocal(w);
        wi = wi.normalized();
        float cosTheta = wi.z();
        if (cosTheta < cosTotalWidth)
        {
            return 0;
        }
        else if (cosTheta >= cosFalloffStart)
        {
            return 1;
        }

        float delta = (cosTheta - cosTotalWidth) /
                      (cosFalloffStart - cosTotalWidth);
        return std::pow(delta, 4);
    }
};

NORI_REGISTER_CLASS(SpotLight, "spot");
NORI_NAMESPACE_END
