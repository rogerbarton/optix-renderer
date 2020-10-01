#include <nori/emitter.h>

NORI_NAMESPACE_BEGIN

/*
 * http://www.pbr-book.org/3ed-2018/Light_Sources/Point_Lights.html
 */

class PointLight : public Emitter {
public:
  PointLight(const PropertyList &propList) {
    m_power = propList.getColor("power"); // we store power, not intensity
    m_position = propList.getPoint3("position");
  }

  Color3f sample(EmitterQueryRecord &lRec,
                 const Point2f &sample) const override {
    // store sample data inside the query record
    // shadow ray is ray from the light to the original first intersection
    // add a little bit to the initial position (move a bit in
    // the direction) because of collision with the object itself
    lRec.shadowRay =
        Ray3f(this->m_position + Epsilon * (lRec.ref - m_position).normalized(),
              (lRec.ref - m_position).normalized());
    lRec.wi = (m_position - lRec.ref).normalized(); // pointing to light
    lRec.pdf = pdf(lRec);                           // 1.0 in this case
    // don't store the normal, because this does not make sense for a point
    // light?

    // based on emitter.h comment on sample function
    return eval(lRec) / pdf(lRec);
  }

  Color3f eval(const EmitterQueryRecord &lRec) const override {
    // intensity divided by squared distance
    // since we store power, we need to convert to intensity (solid angles)
    Color3f Color =
        m_power / (4 * M_PI) / (lRec.ref - m_position).squaredNorm();
    return Color3f(Color.x(), Color.y(), Color.z());
  }

  float pdf(const EmitterQueryRecord &lRec) const override {
    // http://www.pbr-book.org/3ed-2018/Light_Sources/Point_Lights.html
    return 1.f; // all directions uniform, taken from webpage of book
  }

  virtual std::string toString() const override {
    return tfm::format("PointLight[\n"
                       "  power = %s,\n"
                       "  position = %s,\n"
                       "]",
                       m_power.toString(), m_position.toString());
  }

protected:
  Color3f m_power;
};

NORI_REGISTER_CLASS(PointLight, "point");
NORI_NAMESPACE_END