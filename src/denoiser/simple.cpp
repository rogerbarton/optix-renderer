#include <nori/denoiser.h>
#include <nori/block.h>
#include <tbb/parallel_for.h>

#ifdef NORI_USE_IMGUI
#include <imgui/imgui_internal.h>
#endif

NORI_NAMESPACE_BEGIN

class SimpleDenoiser : public Denoiser
{
public:
    SimpleDenoiser(const PropertyList &props)
    {
        // gauss param for pixel distance
        sigma_d = clamp(props.getFloat("sigma_d", 0.f), Epsilon, 10.f);
        sigma_vr = clamp(props.getFloat("sigma_vr", 0.6f), Epsilon, 10.f);

        // patch size for inner loop
        inner_range = clamp(props.getInteger("range", 1), 0, 50);

        max_amount = clamp(props.getInteger("amount", 1), 1, 10);
    }

    NORI_OBJECT_DEFAULT_CLONE(SimpleDenoiser)
    NORI_OBJECT_DEFAULT_UPDATE(SimpleDenoiser)

    void denoise(ImageBlock *block) const override
    {
        std::cout << "DENOISING" << std::endl;

        // for each pixel, compute variance to the overallMean
        tbb::blocked_range<int> range(0, block->getSize()[1]);

        int bs = block->getBorderSize();

        for (int amount = 0; amount < max_amount; amount++)
        {
            const Eigen::MatrixXf variance = computeVarianceFromImage(*block);

            auto map = [&](const tbb::blocked_range<int> &range) {
                for (int i = range.begin(); i < range.end(); i++)
                {
                    for (int j = 0; j < block->getSize()[0]; j++)
                    {
                        float sum_weights = 0.f;
                        float result[] = {0.f, 0.f, 0.f, 0.f};

                        int i_s = clamp(i - inner_range, 0, block->getSize()[1]);
                        int i_e = clamp(i + inner_range + 1, 0, block->getSize()[1]);

                        int j_s = clamp(j - inner_range, 0, block->getSize()[0]);
                        int j_e = clamp(j + inner_range + 1, 0, block->getSize()[0]);

                        for (int i_ = i_s; i_ < i_e; i_++)
                        {
                            for (int j_ = j_s; j_ < j_e; j_++)
                            {
                                float g_ = g_sigma(Point2i(i, j), Point2i(i_, j_));
                                float f_ = f_prime(block, variance, Point2i(i, j), Point2i(i_, j_));
                                float weight = g_ * f_; // w
                                for (int k = 0; k < 4; k++)
                                {
                                    result[k] += block->coeffRef(i_ + bs, j_ + bs)[k] * weight; // Iq * w
                                }
                                sum_weights += weight;
                            }
                        }

                        for (int k = 0; k < 4; k++)
                            block->coeffRef(i + bs, j + bs)[k] = result[k] / sum_weights;
                    
                    }
                }
            };

            tbb::parallel_for(range, map);
        }
    }

    std::string toString() const override
    {
        return tfm::format("SimpleDenoiser[\n"
                           "  sigma_d = %f,\n"
                           "  range = %i,\n"
                           "  sigma_vr = %f,\n"
                           "  amount = %d\n"
                           "]",
                           sigma_d, inner_range, sigma_vr, max_amount);
    }
#ifdef NORI_USE_IMGUI
    NORI_OBJECT_IMGUI_NAME("Simple Denoier");
    virtual bool getImGuiNodes() override
    {
        touched |= Denoiser::getImGuiNodes();

        ImGui::AlignTextToFramePadding();
        ImGui::PushID(2);
        ImGui::TreeNodeEx("Sigma D", ImGuiLeafNodeFlags, "Sigma D");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        touched |= ImGui::DragFloat("##value", &sigma_d, 0.001f, Epsilon, SLIDER_MAX_FLOAT, "%f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::NextColumn();
        ImGui::PopID();

        ImGui::AlignTextToFramePadding();
        ImGui::PushID(1);
        ImGui::TreeNodeEx("sigma_vr", ImGuiLeafNodeFlags, "Sigma VR");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        touched |= ImGui::DragFloat("##value", &sigma_vr, 0.001f, Epsilon, SLIDER_MAX_FLOAT, "%f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::NextColumn();
        ImGui::PopID();

        ImGui::AlignTextToFramePadding();
        ImGui::PushID(3);
        ImGui::TreeNodeEx("Inner Range", ImGuiLeafNodeFlags, "Inner Range");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        touched |= ImGui::DragInt("##value", &inner_range, 1, 0, 20, "%f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::NextColumn();
        ImGui::PopID();

        ImGui::AlignTextToFramePadding();
        ImGui::PushID(4);
        ImGui::TreeNodeEx("amount", ImGuiLeafNodeFlags, "Amount");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        touched |= ImGui::DragInt("##value", &max_amount, 1, 1, 10, "%f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::NextColumn();
        ImGui::PopID();

        return touched;
    }
#endif

private:
    float g_sigma(const Point2i &p, const Point2i &q) const
    {
        return std::exp(-(p - q).squaredNorm() / 2.f / sigma_d / sigma_d);
    }
    float f_prime(const ImageBlock *block, const Eigen::MatrixXf &variance, const Point2i &p, const Point2i &q) const
    {
        // exp(- 0.5f *(|Ip - Iq| * sigmaP,N / simgavr)^2)
        int bs = block->getBorderSize();
        Color4f Ip = block->coeff(p.x() + bs, p.y() + bs);
        Color4f Iq = block->coeff(q.x() + bs, q.y() + bs);

        Vector4f c_diff = Vector4f(Ip[0] - Iq[0], Ip[1] - Iq[1], Ip[2] - Iq[2], Ip[3] - Iq[3]);

        return std::exp(-0.5f * std::pow((c_diff.lpNorm<1>() * variance(p.x(), p.y())) / sigma_vr, 2));
    }

    float colorWeight(const ImageBlock *block, const Eigen::MatrixXf &variance, const Point2i &p, const Point2i &q) const
    {
        // based on https://studios.disneyresearch.com/wp-content/uploads/2019/03/Denoising-Deep-Monte-Carlo-Renderings-1.pdf, equations 1-3
        // iterate over a patch around p and q
        return 0.f;
    }

    float sigma_d;
    float sigma_vr;
    int inner_range;
    int max_amount;
};

NORI_REGISTER_CLASS(SimpleDenoiser, "simple");

NORI_NAMESPACE_END