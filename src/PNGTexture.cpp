#include <nori/texture.h>
#include <lodepng/lodepng.h>
#include <nori/bsdf.h>
#include <nori/HDRLoader.h>
#include <Eigen/Geometry>
#include <filesystem/resolver.h>
#include <filesystem>
#include <imgui/filebrowser.h>

NORI_NAMESPACE_BEGIN

	class PNGTexture : public Texture<Color3f>
	{
	public:
		explicit PNGTexture(const PropertyList &props)
		{
			filename    = getFileResolver()->resolve(props.getString("filename")).str();
			sRgb        = props.getBoolean("sRGB", true);
			scaleU      = props.getFloat("scaleU", 1.f);
			scaleV      = props.getFloat("scaleV", 1.f);
			eulerAngles = props.getVector3("eulerAngles", Vector3f(0.f)) * M_PI / 180.f;
		}

		NORI_OBJECT_DEFAULT_CLONE(PNGTexture)

		void update(const NoriObject *guiObject) override
		{
			const auto *gui = static_cast<const PNGTexture *>(guiObject);
			if (!gui->touched) return;
			gui->touched = false;

			if (gui->fileTouched)
			{
				gui->fileTouched      = false;
				gui->fileLastReadTime = std::filesystem::last_write_time(gui->filename);
				filename = gui->filename;
				sRgb     = gui->sRgb;

				loadFromFile();

				gui->width  = width;
				gui->height = height;
			}

			scaleU      = gui->scaleU;
			scaleV      = gui->scaleV;
			eulerAngles = gui->eulerAngles;
		}

		void loadFromFile()
		{
			if (!std::filesystem::exists(filename))
				throw NoriException("PNGTexture: image file not found %s", filename);

			std::string extension = filename.extension().string();

			if (extension == ".png")
			{
				std::vector<unsigned char> tmp;
				lodepng::decode(tmp, width, height, filename.string());
				data.resize(tmp.size());

				if (sRgb)
				{
					for (unsigned int i = 0; i < data.size(); ++i)
						data[i] = InverseGammaCorrect(static_cast<float>(tmp[i]) / 256);
				}
				else
				{
					for (unsigned int i = 0; i < data.size(); ++i)
						data[i] = static_cast<float>(tmp[i]) / 256;
				}
			}
			else if (extension == ".hdr")
			{
				nori::HDRLoader::HDRLoaderResult result{};

				bool ret = nori::HDRLoader::load(filename.string().c_str(), result);
				if (!ret)
				{
					throw NoriException("Could not load HDR file...");
				}

				width  = result.width;
				height = result.height;

				data.resize(width * height * 4);

				// copy over the data
				for (unsigned int i = 0; i < data.size(); ++i)
					data[i] = result.cols[i];

				delete result.cols; // delete image buffer
			}
			else
			{
				throw NoriException("PNGTexture: file extension %s unknown.", extension);
			}
		}

		//4 bytes per pixel, ordered RGBA
		Color3f eval(const Point2f &_uv) override
		{
			Vector3f wi = sphericalDirection(_uv[1] * M_PI, _uv[0] * 2.f * M_PI);

			Eigen::Matrix3f rot       = Eigen::Quaternionf(
					Eigen::Quaternionf::Identity() *
					Eigen::AngleAxisf(eulerAngles.x(), Eigen::Vector3f::UnitZ()) *
					Eigen::AngleAxisf(eulerAngles.y(), Eigen::Vector3f::UnitX())) *
			                            Eigen::AngleAxisf(eulerAngles.z(), Eigen::Vector3f::UnitZ())
					                            .toRotationMatrix();
			Point2f         uv_coords = sphericalCoordinates(rot * wi);
			Point2f         uv;
			uv.x()         = uv_coords.y() / (2.f * M_PI);
			uv.y()         = uv_coords.x() / M_PI;

			Color3f      out;
			unsigned int w = static_cast<unsigned int>((uv[0]) * scaleU * (float) width);
			unsigned int h = static_cast<unsigned int>((uv[1]) * scaleV * (float) height);

			unsigned int index = (h * width + w) % (width * height);
			out[0] = data[4 * index];
			out[1] = data[4 * index + 1];
			out[2] = data[4 * index + 2];
			return out;
		};

		std::string toString() const override
		{
			return tfm::format("PNGTexture[\n"
			                   "  filename = %s,\n"
			                   "  sRGB = %s, \n"
			                   "  scaleU = %f,\n"
			                   "  scaleV = %f,\n"
			                   "  eulerAngles = %s\n"
			                   "]",
			                   filename, sRgb, scaleU, scaleV, eulerAngles.toString());
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
		NORI_OBJECT_IMGUI_NAME("Image (png)")
		virtual bool getImGuiNodes() override
		{
			touched |= Texture::getImGuiNodes();

			ImGui::AlignTextToFramePadding();
			ImGui::TreeNodeEx("fileName", ImGuiLeafNodeFlags, "Filename");
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(-1);
			ImGui::Text(tfm::format("%s%s", filename.filename().string().c_str(), (fileTouched ? "*" : "")).c_str());
			ImGui::NextColumn();

			// -- Change filename
			ImGui::NextColumn(); // skip column
			static ImGui::FileBrowser fileBrowser;
			if (ImGui::Button("Open"))
			{
				fileBrowser.Open();
				fileBrowser.SetTitle("Open Image File");
				fileBrowser.SetTypeFilters({".png", ".hdr"});
				if (filename.has_parent_path())
					fileBrowser.SetPwd(filename.parent_path());
			}

			ImGui::SameLine();
			if (ImGui::Button("Refresh"))
				fileTouched |= std::filesystem::last_write_time(filename) > fileLastReadTime;
			ImGui::NextColumn();

			fileBrowser.Display();
			if (fileBrowser.HasSelected())
			{
				filename    = fileBrowser.GetSelected();
				fileTouched = true;
				fileBrowser.ClearSelected();
			}

			// -- Image Info
			ImGui::AlignTextToFramePadding();
			ImGui::TreeNodeEx("width", ImGuiLeafNodeFlags, "Width");
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(-1);
			ImGui::Text("%d px", width);
			ImGui::NextColumn();

			ImGui::AlignTextToFramePadding();
			ImGui::TreeNodeEx("height", ImGuiLeafNodeFlags, "Height");
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(-1);
			ImGui::Text("%d px", height);
			ImGui::NextColumn();

			// -- Remaining Properties
			ImGui::AlignTextToFramePadding();
			ImGui::PushID(2);
			ImGui::TreeNodeEx("sRGB", ImGuiLeafNodeFlags, "sRGB");
			ImGui::SameLine();
			ImGui::HelpMarker("Enable this for most textures. Disable for normal maps.");
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(-1);
			fileTouched |= ImGui::Checkbox("##value", &sRgb);
			ImGui::NextColumn();
			ImGui::PopID();

			ImGui::AlignTextToFramePadding();
			ImGui::PushID(3);
			ImGui::TreeNodeEx("scale V", ImGuiLeafNodeFlags, "Scale V");
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(-1);
			touched |= ImGui::DragFloat("##value", &scaleV, 0.01f, 0, 10.f, "%f%", ImGuiSliderFlags_AlwaysClamp);
			ImGui::NextColumn();
			ImGui::PopID();

			ImGui::AlignTextToFramePadding();
			ImGui::PushID(4);
			ImGui::TreeNodeEx("scale U", ImGuiLeafNodeFlags, "Scale U");
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(-1);
			touched |= ImGui::DragFloat("##value", &scaleU, 0.01f, 0, 10.f, "%f", ImGuiSliderFlags_AlwaysClamp);
			ImGui::NextColumn();
			ImGui::PopID();

			eulerAngles *= 180.f * INV_PI;
			ImGui::AlignTextToFramePadding();
			ImGui::PushID(5);
			ImGui::TreeNodeEx("Euler Angles", ImGuiLeafNodeFlags, "EulerAngles");
			ImGui::NextColumn();
			ImGui::SetNextItemWidth(-1);
			touched |= ImGui::DragVector3f("##value", &eulerAngles, 0.5, -360, 360, "%f", ImGuiSliderFlags_AlwaysClamp);
			ImGui::NextColumn();
			ImGui::PopID();
			eulerAngles *= M_PI / 180.f;

			touched |= fileTouched;
			return touched;
		}
#endif

	private:
		std::filesystem::path filename;
		bool                  sRgb;
		float                 scaleU, scaleV;
		Vector3f              eulerAngles;

		std::vector<float>   data;
		mutable unsigned int width, height;

		mutable std::filesystem::file_time_type fileLastReadTime;
		mutable bool                            fileTouched = true;

		float InverseGammaCorrect(float value)
		{
			if (value <= 0.04045f)
				return value * 1.f / 12.92f;
			return std::pow((value + 0.055f) * 1.f / 1.055f, 2.4f);
		}
	};

	NORI_REGISTER_CLASS(PNGTexture, "png_texture");
NORI_NAMESPACE_END