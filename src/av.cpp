#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/warp.h>

NORI_NAMESPACE_BEGIN

class AverageVisibilityIntegrator : public Integrator {
public:
    AverageVisibilityIntegrator(const PropertyList& props) {
        m_length = props.getFloat("length");
    }

    Color3f Li(const Scene *scene, Sampler * sampler, const Ray3f & ray) const {
        Intersection its;
        if(!scene->rayIntersect(ray, its))
            return Color3f(1.0f);

        // compute average visibility

        // sample a point using the normal from the world space shading
        Vector3f dir = Warp::sampleUniformHemisphere(sampler, its.shFrame.n);

        // create the final ray
        Ray3f final_ray = Ray3f(its.p, dir, Epsilon, m_length);
        // check for intersection
        Intersection i;
        if(scene->rayIntersect(final_ray, i)) {
            // not occluded
            return Color3f(0.0f);
        } else {
            // occluded
            return Color3f(1.0f);
        }
    }

    std::string toString() const {
        return tfm::format(
            "AverageVisibilityIntegrator[\n"
            "  length = %f\n"
            "]",
            m_length
        );
    }

protected:
    float m_length = 0.0;
};

NORI_REGISTER_CLASS(AverageVisibilityIntegrator, "av");
NORI_NAMESPACE_END