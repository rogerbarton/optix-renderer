#include <nori/denoiser.h>
#include <tbb/parallel_for.h>

NORI_NAMESPACE_BEGIN

class SimpleDenoiser : public Denoiser
{
public:
    SimpleDenoiser(const PropertyList &props)
    {
        iters = props.getInteger("iterations", 1);
        sigma_r = props.getFloat("sigma_r", 0.f);
        sigma_d = props.getFloat("sigma_d", 0.f);
    }

    void denoise(std::unique_ptr<Bitmap> &bitmap) override
    {
        std::cout << "Denoising the image..." << std::endl;

        Bitmap *output = new Bitmap(Vector2i(bitmap->rows(), bitmap->cols()));

        // Bilateral Filter (https://en.wikipedia.org/wiki/Bilateral_filter)

        tbb::blocked_range<int> range(0, bitmap->rows());

        auto map = [&](const tbb::blocked_range<int> &range) {
            for (int i = range.begin(); i < range.end(); i++)
            {
                for (int j = 0; j < bitmap->cols(); j++)
                {
                    Color3f I_weight = 0;
                    float weight_ijkl = 0;
                    float sum_weights = 0;
                    for (int k = 0; k < bitmap->rows(); k++)
                    {
                        for (int l = 0; l < bitmap->cols(); l++)
                        {
                            weight_ijkl = weight(i, j, k, l, bitmap->operator()(i, j), bitmap->operator()(k, l));
                            sum_weights += weight_ijkl;
                            I_weight += bitmap->operator()(k, l) * weight_ijkl;
                        }
                    }
                    output->operator()(i, j) = I_weight / sum_weights;
                }
            }
        };

        tbb::parallel_for(range, map);

        // copy back to original
        for (int i = 0; i < bitmap->rows(); i++)
        {
            for (int j = 0; j < bitmap->cols(); j++)
            {
                bitmap->operator()(i, j) = output->operator()(i, j);
            }
        }

        delete output; // free memory again
    }

    std::string toString() const override
    {
        return tfm::format("SimpleDenoiser[\n"
                           "  sigma_r = %f,\n"
                           "  sigma_d = %f,\n"
                           "  iterations = %i\n"
                           "]",
                           sigma_r, sigma_d, iters);
    }

private:
    float weight(int i_, int j_, int k_, int l_, Color3f Iij, Color3f Ikl)
    {
        // source https://en.wikipedia.org/wiki/Bilateral_filter
        float i = static_cast<float>(i_);
        float j = static_cast<float>(j_);
        float k = static_cast<float>(k_);
        float l = static_cast<float>(l_);

        Color3f col_diff = Iij - Ikl;
        Vector3f vec_diff = Vector3f(col_diff.r(), col_diff.g(), col_diff.b());

        return std::exp(-((i - k) * (i - k) + (j - l) * (j - l)) / (2 * sigma_d * sigma_d) - (vec_diff.squaredNorm()) / (2 * sigma_r * sigma_r));
    }
    float sigma_r;
    float sigma_d;
    int iters;
};

NORI_REGISTER_CLASS(SimpleDenoiser, "basic");

NORI_NAMESPACE_END