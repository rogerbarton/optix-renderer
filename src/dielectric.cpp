/*
    This file is part of Nori, a simple educational ray tracer

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

#include <nori/bsdf.h>
#include <nori/frame.h>

NORI_NAMESPACE_BEGIN

/// Ideal dielectric BSDF
class Dielectric : public BSDF
{
public:
    Dielectric(const PropertyList &propList)
    {
        /* Interior IOR (default: BK7 borosilicate optical glass) */
        m_intIOR = propList.getFloat("intIOR", 1.5046f);

        /* Exterior IOR (default: air) */
        m_extIOR = propList.getFloat("extIOR", 1.000277f);
    }

    virtual Color3f eval(const BSDFQueryRecord &) const override
    {
        /* Discrete BRDFs always evaluate to zero in Nori */
        return Color3f(0.0f);
    }

    virtual float pdf(const BSDFQueryRecord &) const override
    {
        /* Discrete BRDFs always evaluate to zero in Nori */
        return 0.0f;
    }

    virtual Color3f sample(BSDFQueryRecord &bRec, const Point2f &sample) const override
    {
        // http://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Reflection_Functions.html#sec:mc-specular-deltas
        // https://github.com/mitsuba-renderer/mitsuba/blob/master/src/bsdfs/dielectric.cpp

        // 4b appearance modeling slide 46

        float cosThetaWi = Frame::cosTheta(bRec.wi); // from fresnel
        float F = fresnel(cosThetaWi, m_extIOR, m_intIOR);

        bRec.measure = EDiscrete;
        if (sample.x() < F)
        {
            // perfect specular reflection == mirror
            /*if (cosThetaWi <= 0.f)
            {
                return Color3f(0.f);
            }*/

            bRec.wo = -bRec.wi;
            bRec.wo.z() = bRec.wi.z();

            bRec.eta = 1.0f;

            return Color3f(1.f);
        }
        else
        {
            // specualar refract

            // interiour == eta2
            // exteriour == eta1
            Vector3f normal = Vector3f(0.f, 0.f, 1.f);

            if (cosThetaWi < 0.f)
            {
                normal = -normal;
                bRec.eta = m_intIOR / m_extIOR;
            }
            else
            {
                bRec.eta = m_extIOR / m_intIOR;
            }

            Vector3f wt_1 = -bRec.eta * (bRec.wi - bRec.wi.dot(normal) * normal);
            Vector3f wt_2 = -std::sqrt(1.f - bRec.eta * bRec.eta * (1.f - std::pow(bRec.wi.dot(normal), 2))) * normal;

            bRec.wo = wt_1 + wt_2;

            return Color3f(1.f / bRec.eta / bRec.eta);
        }
    }

    virtual std::string toString() const override
    {
        return tfm::format(
            "Dielectric[\n"
            "  intIOR = %f,\n"
            "  extIOR = %f\n"
            "]",
            m_intIOR, m_extIOR);
    }

private:
    float m_intIOR, m_extIOR;
};

NORI_REGISTER_CLASS(Dielectric, "dielectric");
NORI_NAMESPACE_END
