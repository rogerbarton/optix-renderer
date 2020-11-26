#include <nori/sampler.h>
#include <nori/block.h>
#include <pcg32.h>

NORI_NAMESPACE_BEGIN

/*
 * THis class implements adaptive sampling based on the variance of the current image
 * All functions except getSampleIndices are the same as for the independent sampler
 * 
 * This class is based on "Robust Adaptive Sampling for Monte-Carlo-Based Rendering",
 * Anthony Pajot, Loic Barthe, Mathias Paulin from IRIT, Toulouse, France
 */
class AdaptiveSampler : public Sampler
{
public:
    explicit AdaptiveSampler(const PropertyList &propList)
    {
        m_sampleCount = propList.getInteger("sampleCount", 1);
        keepImproving = propList.getInteger("keepImproving", 3);
        initialUniform = propList.getInteger("initialUniform", 2);
        maxRetries = propList.getInteger("maxRetries", 1000);
    }
    NORI_OBJECT_DEFAULT_CLONE(AdaptiveSampler)
    NORI_OBJECT_DEFAULT_UPDATE(AdaptiveSampler)

    std::unique_ptr<Sampler> clone() const override
    {
        std::unique_ptr<AdaptiveSampler> cloned(new AdaptiveSampler());
        cloned->m_sampleCount = m_sampleCount;
        cloned->m_sampleRound = m_sampleRound;
        cloned->m_random = m_random;
        cloned->keepImproving = keepImproving;
        cloned->initialUniform = initialUniform;
        cloned->maxRetries = maxRetries;
        return std::move(cloned);
    }

    void prepare(const ImageBlock &block) override
    {
        m_random.seed(
            block.getOffset().x(),
            block.getOffset().y());
        m_oldVariance = Eigen::Matrix<Color3f, -1, -1>::Zero(block.rows() - 2 * block.getBorderSize(), block.cols() - 2 * block.getBorderSize());
    }
    void generate() override {}
    void advance() override {}

    float next1D() override
    {
        return m_random.nextFloat();
    }

    Point2f next2D() override
    {
        return Point2f(
            m_random.nextFloat(),
            m_random.nextFloat());
    }

    std::string toString() const override
    {
        return tfm::format("AdaptiveSampler[\n"
                           "  sampleCount = %i,\n"
                           "  keepImproving = %i,\n"
                           "  initialUniform = %i\n"
                           "  ]",
                           m_sampleCount, keepImproving, initialUniform);
    }

    bool computeVariance(const ImageBlock &block) override
    {
        if (m_sampleRound == 0)
        {
            // a new round has been started, clear the old variance matrix
            m_finished = false;
            m_oldVariance.setZero();
            m_oldNorm = 10000.f; // set arbitrary high
        }
        else if (m_sampleRound == maxRetries)
        {
            // after many retries, go to next
            return false;
        }
        if (m_finished)
            return false;

        histogram.clear();

        // if we are in the initial round, use uniform sampling, don't fill histogram
        if (m_sampleRound < initialUniform)
            return true; // true to render and don't compute variance

        Eigen::Matrix<Color3f, -1, -1> variance = Eigen::Matrix<Color3f, -1, -1>::Zero(m_oldVariance.rows(), m_oldVariance.cols());
        for (int i = block.getBorderSize(); i < block.rows() - block.getBorderSize(); i++)
        {
            for (int j = block.getBorderSize(); j < block.cols() - block.getBorderSize(); j++)
            {
                Eigen::Array<float, -1, -1> weights(3, 3);
                weights << 0.f, 1.f, 1.f, 1.f, -3.f, 0.f, 0.f, 0.f, 0.f;

                // we use array multiplication == cwise product
                Color4f curr = Color4f(0.f);
                for (int k = -1; k < 2; k++)
                {
                    for (int l = -1; l < 2; l++)
                    {
                        if (l == 0 && k == 0)
                        {
                            curr -= 8.f * block(i + k, j + l);
                        }
                        else
                        {
                            curr += block(i + k, j + l);
                        }
                    }
                }
                Color3f sum = curr.divideByFilterWeight();
                variance(i - block.getBorderSize(), j - block.getBorderSize()) = sum;
            }
        }

        if (variance.sum().getLuminance() < Epsilon)
        {
            m_oldNorm = (m_oldVariance - variance).sum().getLuminance();
            m_oldVariance = variance;
            return false; // stop immediately, because we have 0 variance in this block.
        }

        //variance.normalize();

        for (int i = 0; i < variance.rows(); i++)
        {
            for (int j = 0; j < variance.cols(); j++)
            {
                histogram.add_element(i, j, variance(i, j).getLuminance());
            }
        }

        // if decreasing variance, render again
        float newNorm = (m_oldVariance - variance).sum().getLuminance();
        if (newNorm < m_oldNorm)
        {
            m_notImproving = 0; // reset m_notImproving, because we had an improvement
        }
        /* we want X samples not improving to stop */
        else if (m_notImproving < keepImproving)
        {
            m_notImproving++;
        }
        else
        {
            /* not decreasing anymore, stop now */
            return false;
        }

        m_oldNorm = newNorm;
        m_oldVariance = variance;
        return true; // rerender
    }

    /*
     * This function takes the full image of the last render step, and calculates the variance of it.
     */
    std::vector<std::pair<int, int>> getSampleIndices(const ImageBlock &block) override
    {
        Vector2i size = block.getSize();
        std::vector<std::pair<int, int>> result;
        result.reserve(size.x() * size.y());

        if (histogram.size() == 0)
        {
            // uniform sampling, every pixel
            for (int i = 0; i < size.x(); i++)
            {
                for (int j = 0; j < size.y(); j++)
                {
                    result.push_back({i, j});
                }
            }
            return result;
        }

        // adaptive sampling
        // place size.x()*size.y() samples based on the variance matrix
        for (int i = 0; i < size.x() * size.y(); i++)
        {
            // https://stackoverflow.com/questions/33426921/pick-a-matrix-cell-according-to-its-probability

            // TODO only select 95% (not those with hightes probability)
            Histogram::elem_type it = histogram.getElement(next1D());
            if (it != histogram.map.end())
            {
                result.push_back(it->second);
            }
            else
            {
                throw NoriException("Adaptive: histogram could not find data point...");
            }
        }

        return result;
    }

    void writeVarianceMatrix(ImageBlock &block, bool fullColor) override
    {
        // write my variance matrix in here
        for (int i = 0; i < m_oldVariance.rows(); i++)
        {
            for (int j = 0; j < m_oldVariance.cols(); j++)
            {
                if (fullColor)
                    block(i + block.getBorderSize(), j + block.getBorderSize()) = Color4f(Color3f(m_oldVariance(i, j).cwiseAbs()));
                else
                    block(i + block.getBorderSize(), j + block.getBorderSize()) = Color4f(Color3f(std::abs(m_oldVariance(i, j).getLuminance())));
            }
        }
    }

#ifdef NORI_USE_IMGUI
    NORI_OBJECT_IMGUI_NAME("Adaptive");
    bool getImGuiNodes() override
    {
        touched |= Sampler::getImGuiNodes();

        ImGui::PushID(1);
        ImGui::AlignTextToFramePadding();
        ImGui::TreeNodeEx("keepImproving", ImGuiLeafNodeFlags, "Keep Improving");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        touched |= ImGui::DragInt("##value", &keepImproving, 1, 1, SLIDER_MAX_INT, "%d%", ImGuiSliderFlags_AlwaysClamp);
        ImGui::NextColumn();
        ImGui::PopID();

        ImGui::PushID(2);
        ImGui::AlignTextToFramePadding();
        ImGui::TreeNodeEx("initialUniform", ImGuiLeafNodeFlags, "Initial Uniform");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        touched |= ImGui::DragInt("##value", &initialUniform, 1, 1, m_sampleCount, "%d%", ImGuiSliderFlags_AlwaysClamp);
        ImGui::NextColumn();
        ImGui::PopID();

        ImGui::PushID(3);
        ImGui::AlignTextToFramePadding();
        ImGui::TreeNodeEx("maxRetries", ImGuiLeafNodeFlags, "Max. Retries");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        touched |= ImGui::DragInt("##value", &maxRetries, 1, 1, SLIDER_MAX_INT, "%d%", ImGuiSliderFlags_AlwaysClamp);
        ImGui::NextColumn();
        ImGui::PopID();

        return touched;
    }
#endif

    bool isAdaptive() const override
    {
        return true;
    }

protected:
    AdaptiveSampler() {}

private:
    pcg32 m_random;
    int keepImproving;
    bool m_finished = false;
    Histogram histogram;
    Eigen::Matrix<Color3f, -1, -1> m_oldVariance;
    float m_oldNorm = 10000.f; // arbitrary big
    int m_notImproving = 0;
    int initialUniform;
    int maxRetries;
};

NORI_REGISTER_CLASS(AdaptiveSampler, "adaptive");

NORI_NAMESPACE_END