
#include <nori/bsdf.h>
#include <nori/integrator.h>
#include <nori/mesh.h>
#include <nori/sampler.h>
#include <nori/scene.h>

NORI_NAMESPACE_BEGIN

class EnvMapTester : public Integrator
{
public:
    explicit EnvMapTester(const PropertyList &props) {}
    NORI_OBJECT_DEFAULT_CLONE(EnvMapTester)
    NORI_OBJECT_DEFAULT_UPDATE(EnvMapTester)

    Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const
    {
        auto envmap = scene->getEnvMap();
        if (!envmap)
        {
            throw NoriException("This integrator requires an env map.");
        }
        EmitterQueryRecord eqr(ray.o); // give origin as reference
        eqr.wi = ray.d;
        return Color3f(envmap->pdf(eqr)/100);
    }

    std::string toString() const override
    {
        return "EnvMapTester[]";
    }
#ifdef NORI_USE_IMGUI
    NORI_OBJECT_IMGUI_NAME("EnvMapTester");
    virtual bool getImGuiNodes() override { return Integrator::getImGuiNodes(); }
#endif
};

NORI_REGISTER_CLASS(EnvMapTester, "envmaptester");
NORI_NAMESPACE_END