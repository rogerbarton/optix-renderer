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

#if !defined(__NORI_SAMPLER_H)
#define __NORI_SAMPLER_H

#include <nori/object.h>
#include <memory>

NORI_NAMESPACE_BEGIN

class ImageBlock;

/**
 * \brief Abstract sample generator
 *
 * A sample generator is responsible for generating the random number stream
 * that will be passed an \ref Integrator implementation as it computes the
 * radiance incident along a specified ray.
 *
 * The most simple conceivable sample generator is just a wrapper around the
 * Mersenne-Twister random number generator and is implemented in
 * <tt>independent.cpp</tt> (it is named this way because it generates 
 * statistically independent random numbers).
 *
 * Fancier samplers might use stratification or low-discrepancy sequences
 * (e.g. Halton, Hammersley, or Sobol point sets) for improved convergence.
 * Another use of this class is in producing intentionally correlated
 * random numbers, e.g. as part of a Metropolis-Hastings integration scheme.
 *
 * The general interface between a sampler and a rendering algorithm is as 
 * follows: Before beginning to render a pixel, the rendering algorithm calls 
 * \ref generate(). The first pixel sample can now be computed, after which
 * \ref advance() needs to be invoked. This repeats until all pixel samples have
 * been exhausted.  While computing a pixel sample, the rendering 
 * algorithm requests (pseudo-) random numbers using the \ref next1D() and
 * \ref next2D() functions.
 *
 * Conceptually, the right way of thinking of this goes as follows:
 * For each sample in a pixel, a sample generator produces a (hypothetical)
 * point in an infinite dimensional random number hypercube. A rendering 
 * algorithm can then request subsequent 1D or 2D components of this point 
 * using the \ref next1D() and \ref next2D() functions. Fancy implementations
 * of this class make certain guarantees about the stratification of the 
 * first n components with respect to the other points that are sampled 
 * within a pixel.
 */
class Sampler : public NoriObject
{
public:
    /// Release all memory
    virtual ~Sampler() {}

    /// Create an exact clone of the current instance
    virtual std::unique_ptr<Sampler> clone() const = 0;

    /**
     * \brief Prepare to render a new image block
     * 
     * This function is called when the sampler begins rendering
     * a new image block. This can be used to deterministically
     * initialize the sampler so that repeated program runs
     * always create the same image.
     */
    virtual void prepare(const ImageBlock &block) = 0;

    /**
     * \brief Prepare to generate new samples
     * 
     * This function is called initially and every time the 
     * integrator starts rendering a new pixel.
     */
    virtual void generate() = 0;

    /// Advance to the next sample
    virtual void advance() = 0;

    /// Retrieve the next component value from the current sample
    virtual float next1D() = 0;

    /// Retrieve the next two component values from the current sample
    virtual Point2f next2D() = 0;

    /// Return the number of configured pixel samples
    virtual int getSampleCount() const { return m_sampleCount; }

    /// sets the current sample round
    void setSampleRound(int sampleRound) { m_sampleRound = sampleRound; }
    void setSampleCount(int sampleCount) { m_sampleCount = sampleCount; }

    /// create pixels to sample based on cumulative variance probabilities
    virtual std::vector<std::pair<int, int>> getSampleIndices(const ImageBlock &block, const Histogram &histogram) = 0;

    virtual bool computeVariance() const { return false; }

    /**
     * \brief Return the type of object (i.e. Mesh/Sampler/etc.) 
     * provided by this instance
     * */
    virtual EClassType getClassType() const override { return ESampler; }
#ifndef NORI_USE_NANOGUI
    virtual bool getImGuiNodes() override
    {
        ImGui::PushID(ESampler);
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen |
                                   ImGuiTreeNodeFlags_Bullet;

        ImGui::AlignTextToFramePadding();
        ImGui::TreeNodeEx("sampleCount", flags, "Sample Count");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        bool ret = ImGui::DragInt("##value", &m_sampleCount, 1, 0, SLIDER_MAX_INT, "%d%", ImGuiSliderFlags_AlwaysClamp);
        ImGui::NextColumn();
        ImGui::PopID();
        return ret;
    }
    virtual const char *getImGuiName() const override { return "Sampler base"; }
#endif
protected:
    int m_sampleCount;
    int m_sampleRound;
};

NORI_NAMESPACE_END

#endif /* __NORI_SAMPLER_H */
