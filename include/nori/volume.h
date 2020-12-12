#pragma once
#ifdef NORI_USE_VDB

#include <nori/shape.h>
#include <nanovdb/util/IO.h>
#include <filesystem/resolver.h>
#include <filesystem>

NORI_NAMESPACE_BEGIN

	class Volume : public Shape
	{
	public:
		Volume() = default;
		explicit Volume(const PropertyList &props);
		NoriObject *cloneAndInit() override;
		void update(const NoriObject *guiObject) override;

		// -- Shape overrides
		BoundingBox3f getBoundingBox(uint32_t index) const override { return m_bbox; }
		Point3f getCentroid(uint32_t index) const override { return m_bbox.getCenter(); }
		bool rayIntersect(uint32_t index, const Ray3f &ray, float &u, float &v, float &t) const override;
		void setHitInformation(uint32_t index, const Ray3f &ray, Intersection &its) const override;
		void sampleSurface(ShapeQueryRecord &sRec, const Point2f &sample) const override;
		float pdfSurface(const ShapeQueryRecord &sRec) const override;
		std::string toString() const override;

#ifdef NORI_USE_IMGUI
		NORI_OBJECT_IMGUI_NAME("Volume");
		bool getImGuiNodes() override;
#endif

#ifdef NORI_USE_OPTIX
		void getOptixHitgroupRecords(OptixState &state, std::vector<HitGroupRecord> &hitgroupRecords) override {
			throw NoriException("not implemented volume");
		}
#endif

		std::filesystem::path                    filename;
		nanovdb::GridHandle<nanovdb::HostBuffer> densityHandle, heatHandle;
		nanovdb::NanoGrid<float>                 *densityGrid = nullptr;
		nanovdb::NanoGrid<float>                 *heatGrid    = nullptr;

		mutable std::filesystem::file_time_type fileLastReadTime;
		mutable bool                            fileTouched   = true;

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