#include <nori/texture.h>
#include <filesystem/resolver.h>
#include <lodepng/lodepng.h>
#include <nori/bsdf.h>
#include <nori/HDRLoader.h>
#include <Eigen/Geometry>

NORI_NAMESPACE_BEGIN

class PNGTexture : public Texture<Color3f>
{
public:
	explicit PNGTexture(const PropertyList &props)
	{
		filename = getFileResolver()->resolve(props.getString("filename"));

		scaleU = props.getFloat("scaleU", 1.f);
		scaleV = props.getFloat("scaleV", 1.f);

		eulerAngles = props.getVector3("eulerAngles", Vector3f(0.f)) * M_PI / 180.f;
	}

	NoriObject *cloneAndInit() override
	{
		auto clone = new PNGTexture(*this);
		clone->loadFromFile();
		return clone;
	}

	void update(const NoriObject *guiObject) override
	{
		const auto* gui = static_cast<const PNGTexture*>(guiObject);
		if (!gui->touched) return;
		gui->touched = false;

		// reload file if the filename has changed. TODO: reload if file has been touched
		if(filename.str() != gui->filename.str())
		{
			filename = gui->filename;
			loadFromFile();
		}

		scaleU = gui->scaleU;
		scaleV = gui->scaleV;
		eulerAngles = gui->eulerAngles;
	}

	void loadFromFile() {
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
	}

	//4 bytes per pixel, ordered RGBA
	Color3f eval(const Point2f &_uv) override
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

		Color3f out;
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
    virtual bool getImGuiNodes() override
    {
		touched |= Texture::getImGuiNodes();

		ImGui::AlignTextToFramePadding();
		ImGui::TreeNodeEx("fileName", ImGuiLeafNodeFlags, "Filename");
		ImGui::NextColumn();
		ImGui::SetNextItemWidth(-1);
		ImGui::Text(filename.filename().c_str());
		ImGui::NextColumn();

		ImGui::AlignTextToFramePadding();
        ImGui::TreeNodeEx("width", ImGuiLeafNodeFlags, "Width");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::Text("%d Pixels", width);
        ImGui::NextColumn();

		ImGui::AlignTextToFramePadding();
        ImGui::TreeNodeEx("height", ImGuiLeafNodeFlags, "Height");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::Text("%d Pixels", height);
        ImGui::NextColumn();

        ImGui::AlignTextToFramePadding();
        ImGui::PushID(2);
        ImGui::TreeNodeEx("scale V", ImGuiLeafNodeFlags, "Scale V");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
	    touched |= ImGui::DragFloat("##value", &scaleV, 0.01f, 0, 10.f, "%f%", ImGuiSliderFlags_AlwaysClamp);
        ImGui::NextColumn();
        ImGui::PopID();

		ImGui::AlignTextToFramePadding();
        ImGui::PushID(3);
        ImGui::TreeNodeEx("scale U", ImGuiLeafNodeFlags, "Scale U");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
	    touched |= ImGui::DragFloat("##value", &scaleU, 0.01f, 0, 10.f, "%f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::NextColumn();
        ImGui::PopID();

		eulerAngles *= 180.f * INV_PI;
		ImGui::AlignTextToFramePadding();
        ImGui::PushID(4);
        ImGui::TreeNodeEx("Euler Angles", ImGuiLeafNodeFlags, "EulerAngles");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
	    touched |= ImGui::DragVector3f("##value", &eulerAngles, 0.5, -360, 360, "%f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::NextColumn();
        ImGui::PopID();
		eulerAngles *= M_PI / 180.f;

		return touched;
	}
#endif

private:
	filesystem::path filename;
	float scaleU, scaleV;
	Vector3f eulerAngles;

	unsigned int width, height;
	std::vector<float> data;

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
	filesystem::path filename;
	float scaleU, scaleV;

	unsigned int width, height;
	std::vector<float> data;

public:
	explicit NormalMap(const PropertyList &props)
	{
		filename = getFileResolver()->resolve(props.getString("filename"));

		scaleU = props.getFloat("scaleU", 1.f);
		scaleV = props.getFloat("scaleV", 1.f);
	}

	NoriObject *cloneAndInit() override
	{
		auto clone = new NormalMap(*this);
		clone->loadFromFile();
		return clone;
	}

	void update(const NoriObject *guiObject) override
	{
		const auto* gui = static_cast<const NormalMap*>(guiObject);
		if (!gui->touched) return;
		gui->touched = false;

		// reload file if the filename has changed. TODO: reload if file has been touched
		if(filename.str() != gui->filename.str())
		{
			filename = gui->filename;
			loadFromFile();
		}
		scaleU = gui->scaleU;
		scaleV = gui->scaleV;
	}

	void loadFromFile() {
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