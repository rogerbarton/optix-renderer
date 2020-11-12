#include <nori/denoiser.h>
#include <tbb/parallel_for.h>

NORI_NAMESPACE_BEGIN

class SimpleDenoiser : public Denoiser
{
public:
    SimpleDenoiser(const PropertyList &props)
    {
        // gauss param for color distance
        sigma_r = clamp(props.getFloat("sigma_r", 0.f), Epsilon, 10.f);

        // gauss param for pixel distance
        sigma_d = clamp(props.getFloat("sigma_d", 0.f), Epsilon, 10.f);

        // patch size for inner loop
        inner_range = clamp(props.getInteger("range", 1), 1, 50);
    }

    Bitmap *denoise(const Bitmap *bitmap) const override
    {
        std::cout << "Denoising the image..." << std::endl;

        Bitmap *output = new Bitmap(Vector2i(bitmap->rows(), bitmap->cols()));

        // Bilateral Filter (https://en.wikipedia.org/wiki/Bilateral_filter)

        tbb::blocked_range<int> range(0, bitmap->rows());

        auto map = [&](const tbb::blocked_range<int> &range) {
            int lower = -inner_range / 2;
            int upper = lower + inner_range;
            for (int i = range.begin(); i < range.end(); i++)
            {
                for (int j = 0; j < bitmap->cols(); j++)
                {
                    Color3f I_weight = 0;
                    float weight_ijkl = 0;
                    float sum_weights = 0;
                    for (int k = lower; k < upper; k++)
                    {
                        for (int l = lower; l < upper; l++)
                        {
                            int k_ = clamp(i + k, 0, bitmap->rows());
                            int l_ = clamp(j + l, 0, bitmap->cols());
                            if (k_ != i + k || l_ != j + l)
                                continue; // we must skip this one
                            weight_ijkl = weight(i, j, k_, l_, (*bitmap)(i, j), (*bitmap)(k_, l_));
                            sum_weights += weight_ijkl;
                            I_weight += (*bitmap)(k_, l_) * weight_ijkl;
                        }
                    }
                    if (sum_weights < Epsilon)
                    {
                        throw NoriException("Denoiser: Sum of Weights was below 0");
                    }
                    (*output)(i, j) = I_weight / sum_weights;
                }
            }
        };

        tbb::parallel_for(range, map);

        return output;
    }

    std::string toString() const override
    {
        return tfm::format("SimpleDenoiser[\n"
                           "  sigma_r = %f,\n"
                           "  sigma_d = %f,\n"
                           "  range = %i\n"
                           "]",
                           sigma_r, sigma_d, inner_range);
    }

private:
    float weight(int i_, int j_, int k_, int l_, const Color3f& Iij, const Color3f& Ikl) const
    {
        // source https://en.wikipedia.org/wiki/Bilateral_filter
        float i = static_cast<float>(i_);
        float j = static_cast<float>(j_);
        float k = static_cast<float>(k_);
        float l = static_cast<float>(l_);

        Color3f col_diff = Iij - Ikl;
        Vector3f vec_diff = Vector3f(col_diff.r(), col_diff.g(), col_diff.b());

        float value = std::exp(-((i - k) * (i - k) + (j - l) * (j - l)) / (2.f * sigma_d * sigma_d) - (vec_diff.squaredNorm()) / (2.f * sigma_r * sigma_r));
        return value;
    }
    float sigma_r;
    float sigma_d;
    int inner_range;
};

NORI_REGISTER_CLASS(SimpleDenoiser, "basic");

NORI_NAMESPACE_END