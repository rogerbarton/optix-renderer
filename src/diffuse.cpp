/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob, Romain Prévost

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

#include <nori/bsdf.h>
#include <nori/frame.h>
#include <nori/warp.h>
#include <nori/texture.h>

NORI_NAMESPACE_BEGIN

/**
 * \brief Diffuse / Lambertian BRDF model
 */
class Diffuse : public BSDF
{
public:
    explicit Diffuse(const PropertyList &propList) : m_albedo(nullptr)
    {
        PropertyList l;
        l.setColor("value", propList.has("albedo") ? propList.getColor("albedo") : Color3f(0.5f));
        m_albedo = static_cast<Texture<Color3f> *>(NoriObjectFactory::createInstance("constant_color", l));
    }

	NoriObject *cloneAndInit() override {
    	auto clone = new Diffuse(*this);
    	clone->m_albedo = static_cast<Texture<Color3f>*>(m_albedo->cloneAndInit());
    	return clone;
    }

	void update(const NoriObject *guiObject) override
	{
		const auto* gui = static_cast<const Diffuse *>(guiObject);
		if (!gui->touched)return;
		gui->touched = false;

		m_albedo->update(gui->m_albedo);
	}

    ~Diffuse() override
    {
        delete m_albedo;
    }

    /// Add texture for the albedo
    virtual void addChild(NoriObject *obj) override
    {
        switch (obj->getClassType())
        {
        case ETexture:
            if (obj->getIdName() == "albedo")
            {
                if (m_albedo)
                    throw NoriException("There is already an albedo defined!");
                m_albedo = static_cast<Texture<Color3f> *>(obj);
            }
            else
            {
                throw NoriException("The name of this texture does not match any field!");
            }
            break;

        default:
            throw NoriException("Diffuse::addChild(<%s>) is not supported!",
                                classTypeName(obj->getClassType()));
        }
    }

    /// Evaluate the BRDF model
    virtual Color3f eval(const BSDFQueryRecord &bRec) const override
    {
        /* This is a smooth BRDF -- return zero if the measure
           is wrong, or when queried for illumination on the backside */
        if (bRec.measure != ESolidAngle || Frame::cosTheta(bRec.wi) <= 0 || Frame::cosTheta(bRec.wo) <= 0)
            return Color3f(0.0f);

        /* The BRDF is simply the albedo / pi */
        return m_albedo->eval(bRec.uv) * INV_PI;
    }

    /// Compute the density of \ref sample() wrt. solid angles
    virtual float pdf(const BSDFQueryRecord &bRec) const override
    {
        /* This is a smooth BRDF -- return zero if the measure
           is wrong, or when queried for illumination on the backside */
        if (bRec.measure != ESolidAngle || Frame::cosTheta(bRec.wi) <= 0 || Frame::cosTheta(bRec.wo) <= 0)
            return 0.0f;

        /* Importance sampling density wrt. solid angles:
           cos(theta) / pi.

           Note that the directions in 'bRec' are in local coordinates,
           so Frame::cosTheta() actually just returns the 'z' component.
        */
        return INV_PI * Frame::cosTheta(bRec.wo);
    }

    /// Draw a a sample from the BRDF model
    virtual Color3f sample(BSDFQueryRecord &bRec, const Point2f &sample) const override
    {
        if (Frame::cosTheta(bRec.wi) <= 0)
            return Color3f(0.0f);

        bRec.measure = ESolidAngle;

        /* Warp a uniformly distributed sample on [0,1]^2
           to a direction on a cosine-weighted hemisphere */
        bRec.wo = Warp::squareToCosineHemisphere(sample);

        /* Relative index of refraction: no change */
        bRec.eta = 1.0f;

        /* eval() / pdf() * cos(theta) = albedo. There
           is no need to call these functions. */
        return m_albedo->eval(bRec.uv);
    }

    bool isDiffuse() const
    {
        return true;
    }

    /// Return a human-readable summary
    virtual std::string toString() const override
    {
        return tfm::format(
            "Diffuse[\n"
            "  albedo = %s\n"
            "]",
            m_albedo ? indent(m_albedo->toString()) : std::string("null"));
    }

    virtual EClassType getClassType() const override { return EBSDF; }
#ifndef NORI_USE_NANOGUI
    virtual const char *getImGuiName() const override { return "Diffuse"; }
    virtual bool getImGuiNodes() override
    {
        bool node_open = ImGui::TreeNode("Texture");
        ImGui::NextColumn();
        ImGui::AlignTextToFramePadding();

        ImGui::Text(m_albedo->getImGuiName());
        ImGui::NextColumn();
        bool ret = false;
        if (node_open)
        {
            ret |= m_albedo->getImGuiNodes();
            ImGui::TreePop();
        }
        return ret;
    }
#endif

private:
    Texture<Color3f> *m_albedo;
};

NORI_REGISTER_CLASS(Diffuse, "diffuse");
NORI_NAMESPACE_END
