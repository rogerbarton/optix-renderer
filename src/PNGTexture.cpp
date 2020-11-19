#include <nori/texture.h>
#include <filesystem/resolver.h>
#include <lodepng/lodepng.h>
#include <nori/bsdf.h>
#include <nori/HDRLoader.h>
#include <Eigen/Geometry>

NORI_NAMESPACE_BEGIN

class PNGTexture : public Texture<Vector3f>
{
public:
	PNGTexture(const PropertyList &props)
	{
		filename = getFileResolver()->resolve(props.getString("filename"));
		if (!filename.exists())
			throw NoriException("PNGTexture: image file not found %s", filename);

		std::string extension = filename.extension();

		if (extension == "png")
		{
			std::vector<unsigned char> tmp;
			lodepng::decode(tmp, width, height, filename.str());
			data.resize(tmp.size());

			for (unsigned int i = 0; i < data.size(); ++i)
			{
				data[i] = InverseGammaCorrect(static_cast<float>(tmp[i]) / 255.f);
			}
		}
		else if (extension == "hdr")
		{
			nori::HDRLoader::HDRLoaderResult result{};
			bool ret = nori::HDRLoader::load(filename.str().c_str(), result);
			if (!ret)
			{
				throw NoriException("Could not load HDR file...");
			}

			width = result.width;
			height = result.height;

			data.resize(width * height * 4);

			// copy over the data
			for (unsigned int i = 0; i < data.size(); ++i)
			{
				data[i] = result.cols[i];
			}

			delete result.cols; // delete image buffer
		}
		else
		{
			throw NoriException("PNGTexture: file extension .%s unknown.", extension);
		}

		scaleU = props.getFloat("scaleU", 1.f);
		scaleV = props.getFloat("scaleV", 1.f);

		eulerAngles = props.getVector3("eulerAngles", Vector3f(0.f)) * M_PI / 180.f;
	}

	//4 bytes per pixel, ordered RGBA
	Vector3f eval(const Point2f &_uv) override
	{
		Vector3f wi = sphericalDirection(_uv[1] * M_PI, _uv[0] * 2.f * M_PI);

		Eigen::Matrix3f rot = Eigen::Quaternionf(
								  Eigen::Quaternionf::Identity() *
								  Eigen::AngleAxisf(eulerAngles.x(), Eigen::Vector3f::UnitZ()) *
								  Eigen::AngleAxisf(eulerAngles.y(), Eigen::Vector3f::UnitX())) *
							  Eigen::AngleAxisf(eulerAngles.z(), Eigen::Vector3f::UnitZ())
								  .toRotationMatrix();
		Point2f uv_coords = sphericalCoordinates(rot * wi);
		Point2f uv;
		uv.x() = uv_coords.y() / (2.f * M_PI);
		uv.y() = uv_coords.x() / M_PI;

		Vector3f out;
		unsigned int w = static_cast<unsigned int>((uv[0]) * scaleU * (float)width);
		unsigned int h = static_cast<unsigned int>((uv[1]) * scaleV * (float)height);

		unsigned int index = (h * width + w) % (width * height);
		out[0] = data[4 * index];
		out[1] = data[(4 * index) + 1];
		out[2] = data[(4 * index) + 2];
		return out;
	};

	std::string toString() const override
	{
		return tfm::format("PNGTexture[\n"
						   "  filename = %s,\n"
						   "  scaleU = %f,\n"
						   "  scaleV = %f,\n"
						   "  eulerAngles = %s\n"
						   "]",
						   filename, scaleU, scaleV, eulerAngles.toString());
	};

	unsigned int getWidth() override
	{
		return width;
	}

	unsigned int getHeight() override
	{
		return height;
	}

#ifndef NORI_USE_NANOGUI
	virtual const char* getImGuiName() const override { return "PNG Texture"; }
    virtual void getImGuiNodes() override {
		ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen |
								   ImGuiTreeNodeFlags_Bullet;

		Texture::getImGuiNodes();

		ImGui::AlignTextToFramePadding();
		ImGui::TreeNodeEx("fileName", flags, "Filename");
		ImGui::NextColumn();
		ImGui::SetNextItemWidth(-1);
		ImGui::Text(filename.str().c_str());
		ImGui::NextColumn();

		ImGui::AlignTextToFramePadding();
        ImGui::TreeNodeEx("width", flags, "Width");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::Text("%d Pixels", width);
        ImGui::NextColumn();

		ImGui::AlignTextToFramePadding();
        ImGui::TreeNodeEx("height", flags, "height");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::Text("%d Pixels", height);
        ImGui::NextColumn();

        ImGui::AlignTextToFramePadding();
        ImGui::PushID(2);
        ImGui::TreeNodeEx("scale V", flags, "Scale V");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragFloat("##value", &scaleV, 0.01, 0, 10.f, "%f%", ImGuiSliderFlags_AlwaysClamp);
        ImGui::NextColumn();
        ImGui::PopID();

		ImGui::AlignTextToFramePadding();
        ImGui::PushID(3);
        ImGui::TreeNodeEx("scale U", flags, "Scale U");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragFloat("##value", &scaleU, 0.01, 0, 10.f, "%f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::NextColumn();
        ImGui::PopID();

		ImGui::AlignTextToFramePadding();
        ImGui::PushID(4);
        ImGui::TreeNodeEx("Euler Angles", flags, "EulerAngles");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragVector3f("##value", &eulerAngles, 0.1, 0, 360, "%f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::NextColumn();
        ImGui::PopID();
	}
#endif

private:
	filesystem::path filename;
	unsigned int width, height;
	std::vector<float> data;
	float scaleU, scaleV;
	Vector3f eulerAngles;

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