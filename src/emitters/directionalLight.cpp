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

#ifdef NORI_USE_OPTIX
#include <nori/optix/sutil/host_vec_math.h>
#endif

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
        m_coord = Frame(m_direction);
    }

    NoriObject *cloneAndInit() override
    {
        auto clone = new DirectionalLight(*this);

        Emitter::cloneAndInit(clone);

        clone->m_angle = m_angle;
        clone->m_direction = m_direction;
        return clone;
    }

    void update(const NoriObject *guiObject) override
    {
        const auto *gui = static_cast<const DirectionalLight *>(guiObject);
        if (!gui->touched)
            return;
        gui->touched = false;

        m_angle = gui->m_angle;
        m_direction = gui->m_direction;
        m_coord = gui->m_coord;

        Emitter::update(guiObject);
    }

    virtual std::string toString() const override
    {
        return tfm::format(
            "DirectionalLight[\n"
            "  direction = %s,\n"
            "  radiance = %s,\n"
            "  angle = %f\n"
            "]",
            m_direction.toString(),
            m_radiance.toString(),
            m_angle);
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
        Vector3f offsetZ = Warp::squareToUniformSphereCap(sample, cos(degToRad(m_angle)));
        lRec.wi = -m_coord.toWorld(offsetZ).normalized();
        lRec.shadowRay = Ray3f(lRec.ref, lRec.wi); // Epsilon, INFINITY are added by default

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
        return m_radiance;
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
        Vector3f localVector = m_coord.toLocal(-lRec.wi);
        float prob = Warp::squareToUniformSphereCapPdf(localVector, cos(degToRad(m_angle)));
        return prob;
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
        touched |= ImGui::DragFloat("##angle", &m_angle, 0.001f, 0.f, 360.f);
        ImGui::NextColumn();
        ImGui::PopID();

        ImGui::PopID();

        return touched;
    }

#ifdef NORI_USE_OPTIX
    void getOptixEmitterData(EmitterData &sbtData) override
    {
        sbtData.type = EmitterData::DIRECTIONAL;
        sbtData.directional.angle = m_angle;
        sbtData.directional.direction = make_float3(m_direction);

        Emitter::getOptixEmitterData(sbtData);
    }
#endif

protected:
    Vector3f m_direction; // direction of the light
    float m_angle;        // world radius (bounding box radius)
    Frame m_coord;
};

NORI_REGISTER_CLASS(DirectionalLight, "directional")
NORI_NAMESPACE_END