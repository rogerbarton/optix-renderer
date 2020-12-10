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

NORI_NAMESPACE_BEGIN

class DirectionalLight : public Emitter {
public:
    DirectionalLight(const PropertyList &props) {
        m_direction = props.getVector3("direction").normalized();
        m_power = props.getColor("power");
        m_radius = props.getFloat("radius");
    }

    virtual std::string toString() const override {
        return tfm::format(
                "DirectionalLight[\n"
                "  direction = %s,\n"
                "  power = %s,\n"
                "  world radius = %f\n"
                "]",
                m_direction.toString(),
                m_power.toString(),
                m_radius);
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
    virtual Color3f sample(EmitterQueryRecord &lRec, const Point2f &sample) const override {  
        
        // create shadow ray
        // distant point
        // instead of hardcoded it would be better more robust to have it as the
        // bounding sphere radius time 2 ?
        float far = 2*m_radius;
        lRec.shadowRay = Ray3f(lRec.ref - far*m_direction, m_direction, Epsilon, far-Epsilon);

        // wi points from the surface away
        lRec.wi = -m_direction;

        // calculate the pdf
        lRec.pdf = pdf(lRec);
    
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
    virtual Color3f eval(const EmitterQueryRecord &lRec) const override {
            // squared distance falloff
            return m_power * M_PI * m_radius * m_radius;
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
    virtual float pdf(const EmitterQueryRecord &lRec) const override {

    /**
     * Compute the probability of sampling the given point lRec.p on the emitter
     * 
     * All points are equally likely
     */
        return 1.0f;
    }

    ~DirectionalLight() {};

    EClassType getClassType() const {return EEmitter; };

protected:
    Color3f m_power; // power of the light source in Watts
    Vector3f m_direction; // direction of the light
    float m_radius; // world radius (bounding box radius)
};

NORI_REGISTER_CLASS(DirectionalLight, "directional")
NORI_NAMESPACE_END