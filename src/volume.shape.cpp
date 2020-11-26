#include <nori/volume.h>

#include <nori/ImguiHelpers.h>
#include <imgui/filebrowser.h>

/**
 * This file contains the shape overrides for a volume. It partially defines volume implementation.
 */

NORI_NAMESPACE_BEGIN

	Volume::Volume(const PropertyList &props)
	{
		filename = getFileResolver()->resolve(props.getString("filename")).str();
	}

	NoriObject *Volume::cloneAndInit()
	{
		auto clone = new Volume{};
		Shape::cloneAndInit(clone);
		return clone;
	}

	void Volume::update(const NoriObject *guiObject)
	{
		const auto *gui = static_cast<const Volume *>(guiObject);
		if (!gui->touched) return;
		gui->touched = false;

		if (gui->fileTouched)
		{
			gui->fileLastReadTime = std::filesystem::last_write_time(gui->filename);
			filename = gui->filename;
			loadFromFile();
		}

		Shape::update(guiObject);

		gui->fileTouched = false;
	}

	bool Volume::rayIntersect(uint32_t index, const Ray3f &ray, float &u, float &v, float &t) const
	{
		return false;
	}
	void Volume::setHitInformation(uint32_t index, const Ray3f &ray, Intersection &its) const
	{

	}
	void Volume::sampleSurface(ShapeQueryRecord &sRec, const Point2f &sample) const
	{

	}
	float Volume::pdfSurface(const ShapeQueryRecord &sRec) const
	{
		return 0;
	}

	std::string Volume::toString() const
	{
		return tfm::format("Volume[\n"
		                   "  filename = %s,\n"
		                   "]",
		                   filename);
	}

#ifdef NORI_USE_IMGUI
	bool Volume::getImGuiNodes()
	{
		ImGui::PushID(EVolume);
		ImGui::AlignTextToFramePadding();
		ImGui::TreeNodeEx("name", ImGuiLeafNodeFlags, "Filename");
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
			fileBrowser.SetTitle("Open Volume File");
			fileBrowser.SetTypeFilters({".vdb", ".nvdb"});
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

		ImGui::PopID();

		return touched;
	}
#endif

	NORI_REGISTER_CLASS(Volume, "volume");

NORI_NAMESPACE_END