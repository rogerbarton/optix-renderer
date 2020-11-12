#include <nori/denoiser.h>
#include <bits/unique_ptr.h>

NORI_NAMESPACE_BEGIN

class SimpleDenoiser : public Denoiser
{
public:
    SimpleDenoiser(const PropertyList &props)
    {
        iters = props.getInteger("iterations", 1);
        sigma_r = props.getFloat("sigma_r", 0.f);
        sigma_s = props.getFloat("sigma_s", 0.f);
    }

    void denoise(Bitmap *bitmap) override
    {
        std::cout << "Denoising the image..." << std::endl;

        // Bilateral Filter with Uniform Variance
        // since the bitmap is just a fancy class on top of an eigen matrix with 3 channels, this results in this:
    }

    std::string toString() const override
    {
        return tfm::format("SimpleDenoiser[\n"
                           "  sigma_r = %f,\n"
                           "  sigma_s = %f,\n"
                           "  iterations = %i\n"
                           "]",
                           sigma_r, sigma_s, iters);
    }

private:
    float sigma_r;
    float sigma_s;
    int iters;
};

NORI_REGISTER_CLASS(SimpleDenoiser, "basic");

NORI_NAMESPACE_END