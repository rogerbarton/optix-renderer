/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Romain Prévost

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
#include <nori/shape.h>
#include <nori/warp.h>

NORI_NAMESPACE_BEGIN

class AreaEmitter : public Emitter {
public:
  AreaEmitter(const PropertyList &props) {
    // store the radiance
    m_radiance = props.getColor("radiance");
  }

  virtual std::string toString() const override {
    return tfm::format("AreaLight[\n"
                       "  radiance = %s,\n"
                       "]",
                       m_radiance.toString());
  }

  virtual Color3f eval(const EmitterQueryRecord &lRec) const override {
    // we need a shape for the arealight to work
    if (!m_shape)
      throw NoriException("There is no shape attached to this Area light!");
    // check the normal if we are on the back
    if (lRec.n.dot(lRec.wi) >= 0) {
      return Color3f(0.f); // we are on the back, return black
    } else {
      return m_radiance; // we are on the front, return the radiance
    }
  }

  virtual Color3f sample(EmitterQueryRecord &lRec,
                         const Point2f &sample) const override {
    if (!m_shape)
      throw NoriException("There is no shape attached to this Area light!");

    // sample the surface using a shapeQueryRecord
    ShapeQueryRecord sqr(lRec.ref);
    m_shape->sampleSurface(sqr, sample);

    // create an emitter query
    // we create a new one because we do not want to change the existing one
    // until we actually should return a color
    EmitterQueryRecord eqr(sqr.ref, sqr.p, sqr.n);
    eqr.shadowRay = Ray3f(eqr.ref, eqr.wi);

    // compute the pdf of this query
    float probs = pdf(eqr);
    // check for it being near zero
    if (std::abs(probs - 0.f) < Epsilon) {
      return Color3f(0.f);
    }

    // update lRec
    lRec = eqr;
    //lRec.pdf = probs;

    // return radiance
    return m_radiance / probs;
  }

  virtual float pdf(const EmitterQueryRecord &lRec) const override {
    if (!m_shape)
      throw NoriException("There is no shape attached to this Area light!");

    // if we are on the back, return 0
    if (lRec.n.dot(lRec.wi) >= 0) {
      return 0.f;
    }
    // create a shape query record and get the pdf of the surface
    // create by reference and sampled point
    ShapeQueryRecord sqr(lRec.ref, lRec.p);
    float prob = m_shape->pdfSurface(sqr);

    // transform the probability to solid angle
    // where the first part is the distance and
    // the second part is the cosine (computed using the normal)
    return prob * (lRec.p - lRec.ref).squaredNorm() /
           std::abs(lRec.n.dot(lRec.wi));
  }

  virtual Color3f samplePhoton(Ray3f &ray, const Point2f &sample1,
                               const Point2f &sample2) const override {
    throw NoriException("To implement...");
  }

protected:
  Color3f m_radiance;
};

NORI_REGISTER_CLASS(AreaEmitter, "area")
NORI_NAMESPACE_END