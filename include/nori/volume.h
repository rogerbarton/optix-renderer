#pragma once
#ifdef NORI_USE_VOLUMES

#include <nori/object.h>
#include <nanovdb/util/IO.h>
#include <filesystem/resolver.h>
#include <filesystem>

NORI_NAMESPACE_BEGIN

class Volume : public NoriObject
{
public:
	Volume(const PropertyList &props);

	EClassType getClassType() const override { return EVolume; }

	std::string toString() const override;

	filesystem::path filename;
	nanovdb::GridHandle<nanovdb::HostBuffer> densityHandle, heatHandle;
	nanovdb::NanoGrid<float> *densityGrid;
	nanovdb::NanoGrid<float> *heatGrid;

#ifndef NORI_USE_NANOGUI
	virtual const char *getImGuiName() const override
	{
		return "Volume";
	}
	virtual void getImGuiNodes() override
	{
		ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen |
								   ImGuiTreeNodeFlags_Bullet;
		ImGui::PushID(EVolume);
		ImGui::AlignTextToFramePadding();
		ImGui::TreeNodeEx("fileName", flags, "Filename");
		ImGui::NextColumn();
		ImGui::SetNextItemWidth(-1);
		ImGui::Text(filename.str().c_str());
		ImGui::NextColumn();
		ImGui::PopID();
	}
#endif

private:
	/**
		 * Loads a .nvdb (NanoVDB) file directly
		 */
	void loadNanoVdb();

	/**
		 * Loads a .vdb (OpenVDB) file and converts it to the NanoVDB format.
		 * Currently the only function that uses the OpenVDB lib
		 */
	void loadOpenVdbAndCacheNanoVdb(const std::filesystem::path &cacheFilename) const;

	/**
		 * Writes the .nvdb (NanoVDB) file to disk. Use this to prevent formats converting each time.
		 */
	void writeToNanoVdb(const std::string &outfile) const;

	// -- Helpers
	/**
		 * Reads and processes a single grid from an nvdb file
		 * @param file .nvdb file
		 * @param gridId id of the grid within the file
		 * @param gridHandle OUT
		 * @param grid OUT
		 */
	void readGrid(filesystem::path &file, uint64_t gridId,
				  nanovdb::GridHandle<nanovdb::HostBuffer> &gridHandle, nanovdb::NanoGrid<float> *&grid);

	/**
		 * Prints information about the grid handle metadata, e.g. type, class
		 * @param gridHandle
		 */
	static void printGridMetaData(const nanovdb::GridHandle<nanovdb::HostBuffer> &gridHandle);

	/**
		 * Prints information about the loaded grid, e.g. active voxels, voxel size
		 */
	static void printGridData(const nanovdb::NanoGrid<float> *grid);
};

NORI_NAMESPACE_END

#endif // NORI_USE_VOLUMES