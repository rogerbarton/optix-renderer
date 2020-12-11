/*
    This file is part of an exercise from the Computer Graphics
    lecture at ETHZ

    Copyright (c) 2015 by Wenzel Jakob

    Nori is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Nori is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <nori/emitter.h>
#include <nori/warp.h>

NORI_NAMESPACE_BEGIN

class DirectionalLight : public Emitter
{
public:
    DirectionalLight(const PropertyList &props)
    {
        // could use lookat
        m_direction = props.getVector3("direction", Vector3f(0, 0, 1)).normalized();
        m_radiance = props.getColor("radiance", Color3f(0.f));
        m_angle = props.getFloat("angle", 1.f);
        m_position = props.getPoint3("position", Point3f(0.f));
        m_radius = props.getFloat("radius", 1.f);
        m_coord = Frame(m_direction);
    }

    NoriObject *cloneAndInit() override
    {
        auto clone = new DirectionalLight(*this);

        Emitter::cloneAndInit(clone);

        clone->m_position = m_position;
        clone->m_angle = m_angle;
        clone->m_direction = m_direction;
        clone->m_radius = m_radius;
        return clone;
    }

    void update(const NoriObject *guiObject) override
    {
        const auto *gui = static_cast<const DirectionalLight *>(guiObject);
        if (!gui->touched)
            return;
        gui->touched = false;

        m_position = gui->m_position;
        m_angle = gui->m_angle;
        m_direction = gui->m_direction;
        m_coord = gui->m_coord;
        m_radius = gui->m_radius;

        Emitter::update(guiObject);
    }

    virtual std::string toString() const override
    {
        return tfm::format(
            "DirectionalLight[\n"
            "  direction = %s,\n"
            "  radiance = %s,\n"
            "  angle = %f,\n"
            "  position = %s,\n"
            "  radius = %f\n"
            "]",
            m_direction.toString(),
            m_radiance.toString(),
            m_angle, m_position.toString(), m_radius);
    }

    /**
     * \brief Sample the emitter and return the importance weight (i.e. the
     * value of the Emitter divided by the probability density
     * of the sample with respect to solid angles).
     *
     * \param lRec    An emitter query record (only ref is needed)
     * \param sample  A uniformly distributed sample on \f$[0,1]^2\f$
     *
     * \return The emitter value divided by the probability density of the sample.
     *         A zero value means that sampling failed.
     */
    virtual Color3f sample(EmitterQueryRecord &lRec, const Point2f &sample) const override
    {

        // create shadow ray
        // distant point
        // instead of hardcoded it would be better more robust to have it as the
        // bounding sphere radius time 2 ?

        Vector3f offsetZ = Warp::squareToUniformSphereCap(sample, cos(degToRad(m_angle)));
        lRec.wi = -m_coord.toWorld(offsetZ).normalized();

        lRec.p = m_position + lRec.wi * m_radius; // directional light = one point

        const float maxt = (m_position - lRec.ref).norm() - Epsilon;
        lRec.shadowRay = Ray3f(lRec.p, (lRec.ref - lRec.p).normalized(), Epsilon, maxt);

        // calculate the pdf
        lRec.pdf = pdf(lRec);

        if (lRec.pdf < Epsilon)
        {
            return Color3f(0.f);
        }

        // return value of emitter divided by pdf
        return eval(lRec) / lRec.pdf;
    }

    /**
     * \brief Evaluate the emitter
     *
     * \param lRec
     *     A record with detailed information on the emitter query
     * \return
     *     The emitter value, evaluated for each color channel
     */
    virtual Color3f eval(const EmitterQueryRecord &lRec) const override
    {
        // TODO CHECK THIS
        float cosTotalWidth = std::cos(degToRad(m_angle));
        Color3f i = m_radiance / (1.f - .5f * (cosTotalWidth));
        Color3f color = i / (lRec.ref - m_position).squaredNorm();
        return color;
    }

    /**
     * \brief Compute the probability of sampling \c lRec.p.
     *
     * This method provides access to the probability density that
     * is realized by the \ref sample() method.
     *
     * \param lRec
     *     A record with detailed information on the emitter query
     *
     * \return
     *     A probability/density value
     */
    virtual float pdf(const EmitterQueryRecord &lRec) const override
    {
        // TODO CHECK THSI
        Vector3f localVector = m_coord.toLocal(-lRec.wi);
        float prob = Warp::squareToUniformSphereCapPdf(localVector, cos(degToRad(m_angle)));
        return prob * (lRec.p - lRec.ref).squaredNorm();
    }

    ~DirectionalLight(){};

    EClassType getClassType() const { return EEmitter; };

    NORI_OBJECT_IMGUI_NAME("DirectionalLight");
    virtual bool getImGuiNodes() override
    {
        touched |= Emitter::getImGuiNodes();
        int counter = 0;
        ImGui::PushID(EEmitter);

        ImGui::PushID(counter++);
        ImGui::AlignTextToFramePadding();
        ImGui::TreeNodeEx("position", ImGuiLeafNodeFlags, "Position");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        touched |= ImGui::DragPoint3f("##position", &m_position, 0.1f);
        ImGui::NextColumn();
        ImGui::PopID();

        ImGui::PushID(counter++);
        ImGui::AlignTextToFramePadding();
        ImGui::TreeNodeEx("direction", ImGuiLeafNodeFlags, "Direction");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        bool dir_touched = ImGui::DragVector3f("##direction", &m_direction, 0.01f, -1.f, 1.f, "%.3f",
                                               ImGuiSliderFlags_AlwaysClamp);
        touched |= dir_touched;
        if (dir_touched)
        {
            m_coord = Frame(m_direction.normalized());
        }
        ImGui::NextColumn();
        ImGui::PopID();

        ImGui::PushID(counter++);
        ImGui::AlignTextToFramePadding();
        ImGui::TreeNodeEx("angle", ImGuiLeafNodeFlags, "Angle");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        touched |= ImGui::DragFloat("##angle", &m_angle, 0.01f, 0.f, 360.f);
        ImGui::NextColumn();
        ImGui::PopID();

        ImGui::PushID(counter++);
        ImGui::AlignTextToFramePadding();
        ImGui::TreeNodeEx("radius", ImGuiLeafNodeFlags, "Radius");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        touched |= ImGui::DragFloat("##radius", &m_radius, 0.001f, 0.f, SLIDER_MAX_FLOAT);
        ImGui::NextColumn();
        ImGui::PopID();

        ImGui::PopID();

        return touched;
    }

#ifdef NORI_USE_OPTIX
    void getOptixEmitterData(EmitterData &sbtData) override
    {
        sbtData.type = EmitterData::DIRECTIONAL;
        sbtData.directional.position = make_float3(m_position);
        sbtData.directional.angle = m_angle;
        sbtData.directional.direction = make_float3(m_direction);

        Emitter::getOptixEmitterData(sbtData);
    }
#endif

protected:
    Color3f m_power;      // power of the light source in Watts
    Vector3f m_direction; // direction of the light
    float m_angle;        // world radius (bounding box radius)
    Point3f m_position;
    float m_radius; // offset of the hit point
    Frame m_coord;
};

NORI_REGISTER_CLASS(DirectionalLight, "directional")
NORI_NAMESPACE_END