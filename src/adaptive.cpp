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
        uniform_every = propList.getInteger("uniformEvery", 100);
    }
    NORI_OBJECT_DEFAULT_CLONE(AdaptiveSampler)
    NORI_OBJECT_DEFAULT_UPDATE(AdaptiveSampler)

    std::unique_ptr<Sampler> clone() const override
    {
        std::unique_ptr<AdaptiveSampler> cloned(new AdaptiveSampler());
        cloned->m_sampleCount = m_sampleCount;
        cloned->m_sampleRound = m_sampleRound;
        cloned->m_random = m_random;
        cloned->uniform_every = uniform_every;
        return std::move(cloned);
    }

    void prepare(const ImageBlock &block) override
    {
        m_random.seed(
            block.getOffset().x(),
            block.getOffset().y());
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
                           "  uniformEvery = %i\n"
                           "  ]",
                           m_sampleCount, uniform_every);
    }

    bool computeVariance() const override
    {
        // filter out all cases were we don't compute the variance
        if (m_sampleRound < 2)
        {
            return false;
        }
        else if (m_sampleRound % uniform_every == 0)
        {
            return false;
        }

        // compute the adaptive samples
        return true;
    }

    /*
     * This function takes the full image of the last render step, and calculates the variance of it.
     */
    std::vector<std::pair<int, int>> getSampleIndices(const ImageBlock &block, const Histogram &histogram) override
    {
        Vector2i size = block.getSize();
        std::vector<std::pair<int, int>> result;
        result.reserve(size.x() * size.y());

        if (!computeVariance())
        {
            // initially run twice over all pixels
            for (int i = 0; i < size.x(); i++)
            {
                for (int j = 0; j < size.y(); j++)
                {
                    result.push_back(std::make_pair(i, j));
                }
            }
        }
        else
        {
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
                    break;
                } else {
                    throw NoriException("Adaptive: histogram could not find data point...");
                }
            }
        }

        return result;
    }
#ifdef NORI_USE_IMGUI
    NORI_OBJECT_IMGUI_NAME("Adaptive");
    bool getImGuiNodes() override
    {
        touched |= Sampler::getImGuiNodes();

        ImGui::AlignTextToFramePadding();
        ImGui::TreeNodeEx("uniformEvery", ImGuiLeafNodeFlags, "Uniform Every");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        touched |= ImGui::DragInt("##value", &uniform_every, 1, 1, SLIDER_MAX_INT, "%d%", ImGuiSliderFlags_AlwaysClamp);
        ImGui::NextColumn();
        return touched;
    }
#endif

protected:
    AdaptiveSampler() {}

private:
    pcg32 m_random;
    int uniform_every;
};

NORI_REGISTER_CLASS(AdaptiveSampler, "adaptive");

NORI_NAMESPACE_END