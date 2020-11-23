#include <nori/bsdf.h>
#include <nori/frame.h>
#include <nori/common.h>
#include <nori/warp.h>

NORI_NAMESPACE_BEGIN

// this helper macro adds the ImGui code for one of the 10 variables
#define ImGuiValue(variable, varName)                                                               \
    ImGui::AlignTextToFramePadding();                                                               \
    ImGui::TreeNodeEx(varName, flags, varName);                                                     \
    ImGui::NextColumn();                                                                            \
    ImGui::SetNextItemWidth(-1);                                                                    \
    ImGui::PushID(id++);                                                                            \
    ret |= ImGui::SliderFloat("##value", &variable, 0, 1, "%f%", ImGuiSliderFlags_AlwaysClamp); \
    ImGui::PopID();                                                                                 \
    ImGui::NextColumn();

class Disney : public BSDF
{
public:
    Disney(const PropertyList &props)
    {
        baseColor = props.getColor("baseColor", Color3f(0.f));

        // clamp all attributes to [0,1]
        metallic = clamp(props.getFloat("metallic", 0.f), 0.f, 1.f);
        subsurface = clamp(props.getFloat("subsurface", 0.f), 0.f, 1.f);
        specular = clamp(props.getFloat("specular", 0.5f), 0.f, 1.f);
        roughness = clamp(props.getFloat("roughness", 0.5f), 0.f, 1.f);
        specularTint = clamp(props.getFloat("specularTint", 0.f), 0.f, 1.f);
        anisotropic = clamp(props.getFloat("anisotropic", 0.f), 0.f, 1.f);
        sheen = clamp(props.getFloat("sheen", 0.f), 0.f, 1.f);
        sheenTint = clamp(props.getFloat("sheenTint", 0.5f), 0.f, 1.f);
        clearcoat = clamp(props.getFloat("clearcoat", 0.f), 0.f, 1.f);
        clearcoatGloss = clamp(props.getFloat("clearcoatGloss", 1.f), 0.f, 1.f);
    }

    Color3f eval(const BSDFQueryRecord &bRec) const override
    {
        // https://github.com/wdas/brdf/blob/main/src/brdfs/disney.brdf

        Vector3f L = bRec.wi;                 // light incident Vector
        Vector3f V = bRec.wo;                 // view vector
        Vector3f N = Vector3f(0.f, 0.f, 1.f); // normal
        Vector3f X = Vector3f(1.f, 0.f, 0.f); // tangent
        Vector3f Y = Vector3f(0.f, 1.f, 0.f); // bitangent

        float NdotL = Frame::cosTheta(L);
        float NdotV = Frame::cosTheta(V);
        if (NdotL < Epsilon || NdotV < Epsilon)
            return Color3f(0.f);

        Vector3f H = (L + V).normalized();
        float NdotH = Frame::cosTheta(H);
        float LdotH = L.dot(H);

        Vector3f Cdlin = mon2lin(baseColor);
        float Cdlum = 0.3f * Cdlin.x() + 0.6f * Cdlin.y() + 0.1f * Cdlin.z();

        Vector3f Ctint = (Cdlum > 0.f) ? Vector3f(Cdlin / Cdlum) : Vector3f(1.f);
        Vector3f Cspec0 = mix<Vector3f>(specular * 0.08f * mix(Vector3f(1.f), Ctint, specularTint), Cdlin, metallic);
        Vector3f Csheen = mix<Vector3f>(Vector3f(1.f), Ctint, sheenTint);

        float FL = SchlickFresnel(NdotL);
        float FV = SchlickFresnel(NdotV);
        float Fd90 = 0.5f + 2.f * LdotH * LdotH * roughness;
        float Fd = mix<float>(1.0, Fd90, FL) * mix<float>(1.0, Fd90, FV);

        float Fss90 = LdotH * LdotH * roughness;
        float Fss = mix<float>(1.0, Fss90, FL) * mix<float>(1.0, Fss90, FV);
        float ss = 1.25f * (Fss * (1.f / (NdotL + NdotV) - 0.5f) + 0.5f);

        // specular
        float aspect = sqrt(1.f - anisotropic * 0.9f);
        float ax = std::max(0.001f, roughness * roughness / aspect);
        float ay = std::max(0.001f, roughness * roughness * aspect);
        float Ds = GTR2_aniso(NdotH, H.dot(X), H.dot(Y), ax, ay);
        float FH = SchlickFresnel(LdotH);
        Vector3f Fs = mix<Vector3f>(Cspec0, Vector3f(1.f), FH);
        float Gs = smithG_GGX_aniso(NdotL, L.dot(X), L.dot(Y), ax, ay);
        Gs *= smithG_GGX_aniso(NdotV, V.dot(X), V.dot(Y), ax, ay);

        // sheen
        Vector3f Fsheen = FH * sheen * Csheen;

        // clearcoat
        float Dr = GTR1(NdotH, mix<float>(0.1f, 0.001f, clearcoatGloss));
        float Fr = mix<float>(0.04f, 1.f, FH);
        float Gr = smithG_GGX(NdotL, 0.25f) * smithG_GGX(NdotV, 0.25f);

        Vector3f finalCol = (INV_PI * mix<float>(Fd, ss, subsurface) * Cdlin + Fsheen) * (1.f - metallic) + Vector3f(Gs * Fs * Ds) + Vector3f(0.25f * clearcoat * Gr * Fr * Dr);

        return Color3f(finalCol.x(), finalCol.y(), finalCol.z());
    }

    Color3f sample(BSDFQueryRecord &bRec, const Point2f &sample) const override
    {
        if (Frame::cosTheta(bRec.wi) <= 0)
        {
            //throw NoriException("cos(bRec.wi) > 0");
            return Color3f(0.f);
        }
        bRec.measure = ESolidAngle;
        bRec.wo = Warp::squareToCosineHemisphere(sample);
        bRec.eta = 1.f;

        // we don't need the pdf here, because col / pdf * solid_angle = col (for cosine weighted)
        Color3f finalCol = eval(bRec);

        return finalCol;
    }

    float pdf(const BSDFQueryRecord &bRec) const override
    {
        if (bRec.measure != ESolidAngle || Frame::cosTheta(bRec.wi) <= 0 || Frame::cosTheta(bRec.wo) <= 0)
        {
            return 0.0f;
        }
        return INV_PI * Frame::cosTheta(bRec.wo);
    }

    virtual std::string toString() const override
    {
        return tfm::format(
            "Disney[\n"
            "  baseColor = %s,\n"
            "  metallic = %f,\n"
            "  subsurface = %f,\n"
            "  specular = %f,\n"
            "  roughness = %f,\n"
            "  specularTint = %f,\n"
            "  anisotropic = %f,\n"
            "  sheen = %f,\n"
            "  sheenTint = %f,\n"
            "  clearcoat = %f,\n"
            "  clearcoatGloss = %f,\n"
            "]",
            baseColor.toString(), metallic, subsurface,
            specular, roughness, specularTint, anisotropic,
            sheen, sheenTint, clearcoat, clearcoatGloss);
    }

    // set diffuse to true for photon mapper
    bool isDiffuse() const
    {
        return true;
    }
#ifndef NORI_USE_NANOGUI
    virtual const char *getImGuiName() const override
    {
        return "Disney BSDF";
    }
    virtual bool getImGuiNodes() override
    {
        bool ret = BSDF::getImGuiNodes();

        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen |
                                   ImGuiTreeNodeFlags_Bullet;

        int id = 1;

        ImGui::AlignTextToFramePadding();

        ImGui::TreeNodeEx("baseColor", flags, "Base Color");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::PushID(id++);
        ret |= ImGui::ColorPicker("##value", &baseColor);
        ImGui::PopID();
        ImGui::NextColumn();

        ImGuiValue(metallic, "Metallic");
        ImGuiValue(subsurface, "Subsurface");
        ImGuiValue(specular, "Specular");
        ImGuiValue(roughness, "Roughness");
        ImGuiValue(specularTint, "Specular Tint");
        ImGuiValue(anisotropic, "Anisotropic");
        ImGuiValue(sheen, "Sheen");
        ImGuiValue(sheenTint, "Sheen Tint");
        ImGuiValue(clearcoat, "Clearcoat");
        ImGuiValue(clearcoatGloss, "Clearcoat Gloss");

        return ret;
    }
#endif
private:
    static Vector3f mon2lin(Color3f vec)
    {
        return Vector3f(pow(vec.r(), 2.2f),
                        pow(vec.g(), 2.2f),
                        pow(vec.b(), 2.2f));
    }

    template <typename T>
    static T mix(T a, T b, float t)
    {
        return a + (b - a) * t;
    }

    static float SchlickFresnel(float a)
    {
        float m = clamp(1.f - a, 0.f, 1.f);
        float m2 = m * m;
        return m2 * m2 * m; // pow(m, 5)
    }
    static float smithG_GGX_aniso(float NdotV, float VdotX, float VdotY, float ax, float ay)
    {
        return 1.f / (NdotV + sqrt(VdotX * ax * VdotX * ax + VdotY * ay * VdotY * ay + NdotV * NdotV));
    }

    static float smithG_GGX(float NdotV, float alphaG)
    {
        float a = alphaG * alphaG;
        float b = NdotV * NdotV;
        return 1.f / (NdotV + sqrt(a + b - a * b));
    }

    static float GTR1(float NdotH, float a)
    {
        if (a >= 1.f)
            return INV_PI;
        float a2 = a * a;
        float t = 1.f + (a2 - 1.f) * NdotH * NdotH;
        return (a2 - 1.f) / (M_PI * log(a2) * t);
    }
    static float GTR2_aniso(float NdotH, float HdotX, float HdotY, float ax, float ay)
    {
        return 1.f / (M_PI * ax * ay * pow(pow(HdotX / ax, 2.f) + pow(HdotY / ay, 2.f) + NdotH * NdotH, 2.f));
    }

    Color3f baseColor;
    float metallic;
    float subsurface;
    float specular;
    float roughness;
    float specularTint;
    float anisotropic;
    float sheen;
    float sheenTint;
    float clearcoat;
    float clearcoatGloss;
};
NORI_REGISTER_CLASS(Disney, "disney");
NORI_NAMESPACE_END