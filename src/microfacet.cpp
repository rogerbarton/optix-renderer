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
#include <nori/warp.h>

NORI_NAMESPACE_BEGIN

class Microfacet : public BSDF
{
public:
  explicit Microfacet(const PropertyList &propList)
  {
    /* RMS surface roughness */
    m_alpha = propList.getFloat("alpha", 0.1f);

    /* Interior IOR (default: BK7 borosilicate optical glass) */
    m_intIOR = propList.getFloat("intIOR", 1.5046f);

    /* Exterior IOR (default: air) */
    m_extIOR = propList.getFloat("extIOR", 1.000277f);

    /* Albedo of the diffuse base material (a.k.a "kd") */
    m_kd = propList.getColor("kd", Color3f(0.5f));

    /* To ensure energy conservation, we must scale the
       specular component by 1-kd.

       While that is not a particularly realistic model of what
       happens in reality, this will greatly simplify the
       implementation. Please see the course staff if you're
       interested in implementing a more realistic version
       of this BRDF. */
    m_ks = 1 - m_kd.maxCoeff();
  }
  NORI_OBJECT_DEFAULT_CLONE(Microfacet)
  NORI_OBJECT_DEFAULT_UPDATE(Microfacet)

  /// Evaluate the microfacet normal distribution D
  float evalBeckmann(const Normal3f &m) const
  {
    float temp = Frame::tanTheta(m) / m_alpha, ct = Frame::cosTheta(m),
          ct2 = ct * ct;

    return std::exp(-temp * temp) / (M_PI * m_alpha * m_alpha * ct2 * ct2);
  }

  /// Evaluate Smith's shadowing-masking function G1
  float smithBeckmannG1(const Vector3f &v, const Normal3f &m) const
  {
    float tanTheta = Frame::tanTheta(v);

    /* Perpendicular incidence -- no shadowing/masking */
    if (tanTheta == 0.0f)
      return 1.0f;

    /* Can't see the back side from the front and vice versa */
    if (m.dot(v) * Frame::cosTheta(v) <= 0)
      return 0.0f;

    float a = 1.0f / (m_alpha * tanTheta);
    if (a >= 1.6f)
      return 1.0f;
    float a2 = a * a;

    /* Use a fast and accurate (<0.35% rel. error) rational
       approximation to the shadowing-masking function */
    return (3.535f * a + 2.181f * a2) / (1.0f + 2.276f * a + 2.577f * a2);
  }

  /// Evaluate the BRDF for the given pair of directions
  virtual Color3f eval(const BSDFQueryRecord &bRec) const override
  {
    if (bRec.wo.z() < 0.f)
      return Color3f(0.f);
    // calculate w_h
    Vector3f wh = (bRec.wi + bRec.wo).normalized();

    float denominator =
        m_ks * evalBeckmann(wh) * fresnel(wh.dot(bRec.wi), m_extIOR, m_intIOR) *
        smithBeckmannG1(bRec.wi, wh) * smithBeckmannG1(bRec.wo, wh);
    float numerator = 4.f * bRec.wi.z() * bRec.wo.z();

    return m_kd * INV_PI + denominator / numerator;
  }

  /// Evaluate the sampling density of \ref sample() wrt. solid angles
  virtual float pdf(const BSDFQueryRecord &bRec) const override
  {
    // check if below surface
    if (bRec.wo.z() <= 0)
      return 0.f;
    // compute wh
    Vector3f wh = (bRec.wo + bRec.wi).normalized();

    float part1 = m_ks * evalBeckmann(wh) * wh.z() / (4.f * bRec.wo.dot(wh));
    float part2 = (1.f - m_ks) * bRec.wo.z() * INV_PI;
    return part1 + part2;
  }

  /// Sample the BRDF
  virtual Color3f sample(BSDFQueryRecord &bRec,
                         const Point2f &_sample) const override
  {
    if (bRec.wi.z() < 0)
      return Color3f(0.f);

    Point2f sample_ = _sample;

    if (sample_.y() < m_ks)
    {
      sample_.y() /= m_ks;
      // randomly generate wh with PDF prop to D
      Vector3f wh = Warp::squareToBeckmann(sample_, m_alpha);
      // reflect wi about wh to obtain wo
      Vector3f wo = 2.f * (bRec.wi.dot(wh) * wh) - bRec.wi;
      bRec.wo = wo;
      return bRec.wo.z() <= 0.f ? Color3f(0.f)
                                : eval(bRec) / pdf(bRec) * bRec.wo.z();
    }
    else
    {
      sample_.y() = (sample_.y() - m_ks) / (1.f - m_ks);
      bRec.wo = Warp::squareToCosineHemisphere(sample_);
      return bRec.wo.z() <= 0.f ? Color3f(0.f)
                                : eval(bRec) / pdf(bRec) * bRec.wo.z();
    }
  }

  virtual std::string toString() const override
  {
    return tfm::format("Microfacet[\n"
                       "  alpha = %f,\n"
                       "  intIOR = %f,\n"
                       "  extIOR = %f,\n"
                       "  kd = %s,\n"
                       "  ks = %f\n"
                       "]",
                       m_alpha, m_intIOR, m_extIOR, m_kd.toString(), m_ks);
  }
#ifndef NORI_USE_NANOGUI
  virtual const char *getImGuiName() const override
  {
    return "Microfacet";
  }
  virtual bool getImGuiNodes() override
  {
	  touched |= BSDF::getImGuiNodes();

	  ImGui::AlignTextToFramePadding();
	  ImGui::PushID(1);
	  ImGui::TreeNodeEx("alpha", ImGuiLeafNodeFlags, "Alpha");
	  ImGui::NextColumn();
	  ImGui::SetNextItemWidth(-1);
	  touched |= ImGui::DragFloat("##value", &m_alpha, 0.01f, 0, 10.f, "%f%", ImGuiSliderFlags_AlwaysClamp);
	  ImGui::NextColumn();
	  ImGui::PopID();

	  ImGui::AlignTextToFramePadding();
	  ImGui::PushID(2);
	  ImGui::TreeNodeEx("intIOR", ImGuiLeafNodeFlags, "Interior IOR");
	  ImGui::NextColumn();
	  ImGui::SetNextItemWidth(-1);
	  touched |= ImGui::DragFloat("##value", &m_intIOR, 0.01f, 0, 10.f, "%f%", ImGuiSliderFlags_AlwaysClamp);
	  ImGui::NextColumn();
	  ImGui::PopID();

	  ImGui::AlignTextToFramePadding();
	  ImGui::PushID(3);
	  ImGui::TreeNodeEx("Exterior IOR", ImGuiLeafNodeFlags, "Exterior IOR");
	  ImGui::NextColumn();
	  ImGui::SetNextItemWidth(-1);
	  touched |= ImGui::DragFloat("##value", &m_extIOR, 0.01f, 0, 10.f, "%f%", ImGuiSliderFlags_AlwaysClamp);
	  ImGui::NextColumn();
	  ImGui::PopID();

	  ImGui::AlignTextToFramePadding();
	  ImGui::PushID(3);
	  ImGui::TreeNodeEx("Albedo of Diffuse Base", ImGuiLeafNodeFlags, "Albedo of Diffuse Base");
	  ImGui::NextColumn();
	  ImGui::SetNextItemWidth(-1);
	  touched |= ImGui::ColorPicker("##value", &m_kd);
	  ImGui::NextColumn();
	  ImGui::PopID();

	  return touched;
  }
#endif

private:
  float m_alpha;
  float m_intIOR, m_extIOR;
  float m_ks;
  Color3f m_kd;
};

NORI_REGISTER_CLASS(Microfacet, "microfacet");
NORI_NAMESPACE_END
