#pragma once
#ifdef NORI_USE_VDB

#include <nori/object.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/SampleFromVoxels.h>

#include <filesystem/resolver.h>
#include <filesystem>

NORI_NAMESPACE_BEGIN

	struct NvdbVolume : public NoriObject
	{
		float getDensity(const nanovdb::Vec3f &point);
		float getDensity(const Vector3f &point);
		float getTemperature(const nanovdb::Vec3f &point);
		float getTemperature(const Vector3f &point);

		NvdbVolume() = default;
		explicit NvdbVolume(const PropertyList &props);
		NoriObject *cloneAndInit() override;
		void update(const NoriObject *guiObject) override;

		EClassType getClassType() const override { return EVolume; }
		std::string toString() const override;

#ifdef NORI_USE_IMGUI
		NORI_OBJECT_IMGUI_NAME("Volume");
		bool getImGuiNodes() override;
#endif

		std::filesystem::path                    filename;
		nanovdb::GridHandle<nanovdb::HostBuffer> densityHandle, temperatureHandle;
		nanovdb::NanoGrid<float>                 *densityGrid     = nullptr;
		nanovdb::NanoGrid<float>                 *temperatureGrid = nullptr;

		static constexpr int                                                               InterpolationOrder  = 2;
		nanovdb::SampleFromVoxels<nanovdb::DefaultReadAccessor<float>, InterpolationOrder> *densitySampler     = nullptr;
		nanovdb::SampleFromVoxels<nanovdb::DefaultReadAccessor<float>, InterpolationOrder> *temperatureSampler = nullptr;

		mutable bool                            fileTouched = true;
		mutable std::filesystem::file_time_type fileLastReadTime;

	private:
		void loadFromFile();

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
		void readGrid(std::filesystem::path &file, uint64_t gridId,
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

#endif // NORI_USE_VDB