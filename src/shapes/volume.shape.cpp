#include <nori/volume.h>

#include <nori/ImguiHelpers.h>
#include <imgui/filebrowser.h>

/**
 * This file contains the shape overrides for a volume. It partially defines volume implementation.
 */

NORI_NAMESPACE_BEGIN

	NvdbVolume::NvdbVolume(const PropertyList &props)
	{
		filename = getFileResolver()->resolve(props.getString("filename")).str();
	}

	NoriObject *NvdbVolume::cloneAndInit()
	{
		auto clone = new NvdbVolume{};
		return clone;
	}

	void NvdbVolume::update(const NoriObject *guiObject)
	{
		const auto *gui = static_cast<const NvdbVolume *>(guiObject);
		if (!gui->touched) return;
		gui->touched = false;

		if (gui->fileTouched)
		{
			gui->fileLastReadTime = std::filesystem::last_write_time(gui->filename);
			filename = gui->filename;
			loadFromFile();
		}

		gui->fileTouched = false;
	}

	std::string NvdbVolume::toString() const
	{
		return tfm::format("Volume[\n"
		                   "  filename = %s,\n"
		                   "]",
		                   filename);
	}

#ifdef NORI_USE_IMGUI
	bool NvdbVolume::getImGuiNodes()
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

	NORI_REGISTER_CLASS(NvdbVolume, "volume");

NORI_NAMESPACE_END