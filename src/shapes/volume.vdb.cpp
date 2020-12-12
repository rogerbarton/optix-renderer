#include <nori/volume.h>

#include <openvdb/tools/LevelSetSphere.h>
#include <nanovdb/util/OpenToNanoVDB.h>

NORI_NAMESPACE_BEGIN

	void NvdbVolume::loadFromFile()
	{
		const auto originalExtension = filename.extension();

		// NanoVDB has its own file format
		// We cache the converted file, and update it if the original .vdb file has been touched
		if (originalExtension == ".vdb")
		{
			std::filesystem::path filenameNvdb{filename};
			filenameNvdb.replace_extension(".nvdb");

			// Check if cache exists and is up to date
			if (!std::filesystem::exists(filenameNvdb) ||
			    std::filesystem::last_write_time(filename) > std::filesystem::last_write_time(filenameNvdb))
			{
				std::cout << "Updating .nvdb cache..." << std::endl;
				loadOpenVdbAndCacheNanoVdb(filenameNvdb);
				loadNanoVdb();
			}
			else
			{
				std::cout << ".nvdb cache hit" << std::endl;
				filename = filenameNvdb.string();
				loadNanoVdb();
			}
		}
		else if (originalExtension == ".nvdb")
			loadNanoVdb();
		else
			throw NoriException("Volume: file extension %s unknown.", originalExtension);
	}

	void
	NvdbVolume::readGrid(std::filesystem::path &file, uint64_t gridId, nanovdb::GridHandle<nanovdb::HostBuffer> &gridHandle,
	                     nanovdb::NanoGrid<float> *&grid)
	{
		gridHandle = nanovdb::io::readGrid(filename.string(), gridId);

		// -- Density
		{
			printGridMetaData(gridHandle);

			// Define a (raw) pointer to the NanoVDB grid on the host. Note we match the value type of the srcGrid!
			grid = gridHandle.grid<float>();

			if (!grid)
				throw NoriException("GridHandle %i does not contain a grid with value type float. (%s)",
				                    gridId, file.string());

			printGridData(grid);
		}
	}

	void NvdbVolume::loadNanoVdb()
	{
		readGrid(filename, 0, densityHandle, densityGrid);
		readGrid(filename, 1, heatHandle, heatGrid);

		// Get accessors for the two grids. Note that accessors only accelerate repeated access!
		auto densityAcc = densityGrid->getAccessor();

		// Access and print out a cross-section of the narrow-band level set from the two grids
		for (int i = 97; i < 104; ++i)
			printf("(%3i,0,0) NanoVDB cpu: % -4.2f\n", i, densityAcc.getValue(nanovdb::Coord(i, 0, 0)));
	}

	void NvdbVolume::loadOpenVdbAndCacheNanoVdb(const std::filesystem::path &cacheFilename) const
	{
		try
		{
			// -- Read all grids from the OpenVDB .vdb file
			openvdb::initialize();
			openvdb::io::File file(filename.string());
			file.open();

			openvdb::FloatGrid::Ptr              ovdbDensityGrid;
			openvdb::FloatGrid::Ptr              ovdbHeatGrid;
			for (openvdb::io::File::NameIterator nameIt = file.beginName(); nameIt != file.endName(); ++nameIt)
			{
				std::cout << "\t" << nameIt.gridName() << std::endl;
				openvdb::GridBase::Ptr grid = file.readGrid(nameIt.gridName());

				std::cout << "\t grid type    : " << grid->type() << std::endl;
				std::cout << "\t active voxels: " << grid->activeVoxelCount() << std::endl;
				std::cout << "\t active dim   : " << grid->evalActiveVoxelDim() << std::endl;
				std::cout << "\t voxel size   : " << grid->voxelSize() << std::endl;

				// Blender names: density, shadow, temperature, velocity
				if (nameIt.gridName() == "density")
					ovdbDensityGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(grid);
				else if (nameIt.gridName() == "temperature")
					ovdbHeatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(grid);
			}

			file.close();
			// -- end of OpenVDB part

			// -- Convert OpenVDB to NanoVDB
			// Convert the OpenVDB grid into a NanoVDB grid handle.
			std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> gridHandles;
			gridHandles.push_back(nanovdb::openToNanoVDB(*ovdbDensityGrid));
			gridHandles.push_back(nanovdb::openToNanoVDB(*ovdbHeatGrid));

			// Write the NanoVDB grid to file and throw if writing fails
			nanovdb::io::writeGrids<nanovdb::HostBuffer, std::vector>(cacheFilename.string(), gridHandles
#ifdef NANOVDB_USE_BLOSC
					, nanovdb::io::Codec::BLOSC
#endif
			);

			openvdb::uninitialize();
		}
		catch (const std::exception &e)
		{
			std::cerr << "loadOpenVdbAndCacheNanoVdb exception occurred: " << e.what() << std::endl;
			openvdb::uninitialize();
		}
	}

	void NvdbVolume::writeToNanoVdb(const std::string &outfile) const
	{
		try
		{
			nanovdb::io::writeGrid(outfile, densityHandle);
		}
		catch (const std::exception &e)
		{
			std::cerr << "Volume: writeToNanoVdb exception occurred: " << e.what() << std::endl;
		}
	}

	void NvdbVolume::printGridMetaData(const nanovdb::GridHandle<nanovdb::HostBuffer> &gridHandle)
	{
		const auto meta = gridHandle.gridMetaData();
		std::cout << "\t" << meta->gridName() << std::endl;
		std::cout << "\t grid type    : " << static_cast<int>(meta->gridType()) << std::endl;
		std::cout << "\t grid class   : " << static_cast<int>(meta->gridClass()) << std::endl;
	}

	void NvdbVolume::printGridData(const nanovdb::NanoGrid<float> *grid)
	{
		std::cout << "\t active voxels: " << static_cast<unsigned long>(grid->activeVoxelCount()) << std::endl;
		std::cout << "\t voxel size   : " << grid->voxelSize()[0] << ", " << grid->voxelSize()[1] << ", "
		          << grid->voxelSize()[2] << std::endl;
	}

NORI_NAMESPACE_END