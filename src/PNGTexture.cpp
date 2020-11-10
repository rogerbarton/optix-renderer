#pragma once

#include <nori/texture.h>
#include <filesystem/resolver.h>
#include <lodepng/lodepng.h>
#include <nori/bsdf.h>

NORI_NAMESPACE_BEGIN

class PNGTexture : public Texture<Vector3f>
{
public:
	PNGTexture(const PropertyList &props)
	{
		filename = getFileResolver()->resolve(props.getString("filename"));
		std::vector<unsigned char> tmp;
		lodepng::decode(tmp, width, height, filename.str());
		data.resize(tmp.size());

		for (unsigned int i = 0; i < data.size(); ++i)
		{
			data[i] = InverseGammaCorrect(static_cast<float>(tmp[i]) / 255.f);
		}

		scaleU = props.getFloat("scaleU", 1.f);
		scaleV = props.getFloat("scaleV", 1.f);
	}

	//4 bytes per pixel, ordered RGBA
	Vector3f eval(const Point2f &_uv) override
	{
		Vector3f out(0.f);
		unsigned int w = static_cast<unsigned int>(_uv[0] * scaleU * (float)width);
		unsigned int h = static_cast<unsigned int>(_uv[1] * scaleV * (float)height);
		unsigned int index = (h * width + w) % (width * height);
		out[0] = data[4 * index];
		out[1] = data[(4 * index) + 1];
		out[2] = data[(4 * index) + 2];
		return out;
	};

	std::string toString() const override
	{
		return tfm::format("PNGTexture[\n"
						   "filename: %s,\n"
						   "scaleU: %f,\n"
						   "scaleV: %f,\n"
						   "]",
						   filename, scaleU, scaleV);
	};

	unsigned int getWidth() override
	{
		return width;
	}

	unsigned int getHeight() override
	{
		return height;
	}

private:
	filesystem::path filename;
	unsigned int width, height;
	std::vector<float> data;
	float scaleU, scaleV;

	float InverseGammaCorrect(float value)
	{
		if (value <= 0.04045f)
			return value * 1.f / 12.92f;
		return std::pow((value + 0.055f) * 1.f / 1.055f, 2.4f);
	}
};

NORI_REGISTER_CLASS(PNGTexture, "png_texture");

class NormalMap : public Texture<Vector3f>
{

	unsigned int width, height;
	std::vector<float> data;

	float scaleU, scaleV;

public:
	NormalMap(const PropertyList &props)
	{
		filesystem::path filename = getFileResolver()->resolve(props.getString("filename"));
		std::vector<unsigned char> d;
		lodepng::decode(d, width, height, filename.str());

		data.reserve(3 * width * height);

		for (unsigned int x = 0; x < width; ++x)
		{
			for (unsigned int y = 0; y < height; ++y)
			{
				unsigned int index = y * width + x;
				float r = static_cast<float>(d[4 * index]) / 255.f;
				float g = static_cast<float>(d[(4 * index) + 1]) / 255.f;
				float b = static_cast<float>(d[(4 * index) + 2]) / 255.f;

				data.emplace_back(r);
				data.emplace_back(g);
				data.emplace_back(b);
			}
		}

		scaleU = props.getFloat("scaleU", 1.f);
		scaleV = props.getFloat("scaleV", 1.f);
	}

	//4 bytes per pixel, ordered RGBA
	Vector3f eval(const Point2f &_uv) override
	{
		unsigned int w = static_cast<unsigned int>(_uv[0] * scaleU * (float)width);
		unsigned int h = static_cast<unsigned int>(_uv[1] * scaleV * (float)height);
		unsigned int index = (h * width + w) % (width * height);
		float r = data[3 * index];
		float g = data[(3 * index) + 1];
		float b = data[(3 * index) + 2];
		return Vector3f(r, g, b);
	};

	std::string toString() const override
	{
		return tfm::format(
			"NormalMap[]");
	};

	unsigned int getWidth() override
	{
		return width;
	}

	unsigned int getHeight() override
	{
		return height;
	}
};
NORI_REGISTER_CLASS(NormalMap, "normal_map");

NORI_NAMESPACE_END