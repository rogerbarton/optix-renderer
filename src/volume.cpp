#include <iostream>

#include <openvdb/tools/LevelSetSphere.h>
#include <nanovdb/util/OpenToNanoVDB.h>
#include <nori/volume.h>

NORI_NAMESPACE_BEGIN

Volume::Volume(const PropertyList &props)
{
	filename = getFileResolver()->resolve(props.getString("filename"));
	if (!filename.exists())
		throw NoriException(tfm::format("Volume: file not found %s", filename).c_str());

	const auto originalExtension = filename.extension();

	// NanoVDB has its own file format
	// We cache the converted file, and update it if the original .vdb file has been touched
	if (originalExtension == "vdb")
	{
		std::filesystem::path filenameStd{filename.str()};
		std::filesystem::path filenameNvdb{filenameStd};
		filenameNvdb.replace_extension(".nvdb");

		// Check if cache exists and is up to date
		if (!std::filesystem::exists(filenameNvdb) ||
			std::filesystem::last_write_time(filenameStd) > std::filesystem::last_write_time(filenameNvdb))
		{
			std::cout << "Updating .nvdb cache..." << std::endl;
			loadOpenVdbAndCacheNanoVdb(filenameNvdb);
		}
		else
		{
			std::cout << ".nvdb cache hit" << std::endl;
			filename = filenameNvdb.string();
			loadNanoVdb();
		}
	}
	else if (originalExtension == "nvdb")
		loadNanoVdb();
	else
		throw NoriException("Volume: file extension .%s unknown.", originalExtension);
}

std::string Volume::toString() const
{
	return tfm::format("Volume[\n"
					   "  filename = %s,\n"
					   "]",
					   filename);
}

/**
		 * Loads a .nvdb (NanoVDB) file directly
		 */
void Volume::loadNanoVdb()
{
	densityHandle = nanovdb::io::readGrid(filename.str(), 0);
	heatHandle = nanovdb::io::readGrid(filename.str(), 1);

	{
		const auto meta = densityHandle.gridMetaData();
		std::cout << "\t" << meta->gridName() << std::endl;
		std::cout << "\t grid type    : " << static_cast<int>(meta->gridType()) << std::endl;
		std::cout << "\t grid class   : " << static_cast<int>(meta->gridClass()) << std::endl;

		// Define a (raw) pointer to the NanoVDB grid on the host. Note we match the value type of the srcGrid!
		grid = densityHandle.grid<float>();

		if (!grid)
			throw std::runtime_error("GridHandle does not contain a grid with value type float");

		std::cout << "\t active voxels: " << static_cast<unsigned long>(grid->activeVoxelCount()) << std::endl;
		std::cout << "\t voxel size   : " << grid->voxelSize()[0] << ", " << grid->voxelSize()[1] << ", "
				  << grid->voxelSize()[2] << std::endl;
	}

	{
		const auto meta = heatHandle.gridMetaData();
		std::cout << "\t" << meta->gridName() << std::endl;
		std::cout << "\t grid type    : " << static_cast<int>(meta->gridType()) << std::endl;
		std::cout << "\t grid class   : " << static_cast<int>(meta->gridClass()) << std::endl;

		// Define a (raw) pointer to the NanoVDB grid on the host. Note we match the value type of the srcGrid!
		heat = heatHandle.grid<float>();

		if (!heat)
			throw std::runtime_error("GridHandle does not contain a grid with value type float");

		std::cout << "\t active voxels: " << static_cast<unsigned long>(heat->activeVoxelCount()) << std::endl;
		std::cout << "\t voxel size   : " << heat->voxelSize()[0] << ", " << heat->voxelSize()[1] << ", "
				  << heat->voxelSize()[2] << std::endl;
	}

	// Get accessors for the two grids. Note that accessors only accelerate repeated access!
	auto gridAcc = grid->getAccessor();

	// Access and print out a cross-section of the narrow-band level set from the two grids
	for (int i = 97; i < 104; ++i)
	{
		printf("(%3i,0,0) NanoVDB cpu: % -4.2f\n", i, gridAcc.getValue(nanovdb::Coord(i, 0, 0)));
	}
}

/**
		 * Loads a .vdb (OpenVDB) file and converts it to the NanoVDB format
		 */
void Volume::loadOpenVdbAndCacheNanoVdb(const std::filesystem::path &cacheFilename)
{
	try
	{
		openvdb::initialize();

		openvdb::io::File file(filename.str());
		file.open();
		std::cout << "\tcompressed: " << std::boolalpha << file.hasBloscCompression() << std::endl;

		openvdb::FloatGrid::Ptr densityGrid;
		openvdb::FloatGrid::Ptr heatGrid;
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
				densityGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(grid);
			else if (nameIt.gridName() == "temperature")
				heatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(grid);
		}

		file.close();

		// Create an OpenVDB grid (here a level set surface but replace this with your own code)
		// auto srcGrid = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(100.0f, openvdb::Vec3f(0.0f),
		//                                                                         1.0f);

		// Convert the OpenVDB grid, srcGrid, into a NanoVDB grid handle.
		std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> gridHandles;
		gridHandles.push_back(nanovdb::openToNanoVDB(*densityGrid));
		gridHandles.push_back(nanovdb::openToNanoVDB(*heatGrid));

		// Define a (raw) pointer to the NanoVDB grid on the host. Note we match the value type of the srcGrid!
		grid = gridHandles[0].grid<float>();

		if (!grid)
			throw std::runtime_error("GridHandle does not contain a grid with value type float");

		// Get accessors for the two grids. Note that accessors only accelerate repeated access!
		auto gridAcc = grid->getAccessor();
		auto srcAcc = densityGrid->getAccessor();

		// Access and print out a cross-section of the narrow-band level set from the two grids
		for (int i = 97; i < 104; ++i)
		{
			printf("(%3i,0,0) OpenVDB cpu: % -4.2f, NanoVDB cpu: % -4.2f\n", i,
				   srcAcc.getValue(openvdb::Coord(i, 0, 0)), gridAcc.getValue(nanovdb::Coord(i, 0, 0)));
		}

		// Write the NanoVDB grid to file and throw if writing fails
		std::cout << "Cache file: " << cacheFilename.string() << std::endl;
		nanovdb::io::writeGrids<nanovdb::HostBuffer, std::vector>(cacheFilename.string(), gridHandles
#ifdef NANOVDB_USE_BLOSC
																  ,
																  nanovdb::io::Codec::BLOSC
#endif
		);

		openvdb::uninitialize();
	}
	catch (const std::exception &e)
	{
		std::cerr << "An exception occurred: " << e.what() << std::endl;
	}
}

/**
		 * Writes the .nvdb (NanoVDB) file to disk. Use this to prevent formats converting each time.
		 */
void Volume::writeToNanoVdb(std::string filename)
{
	try
	{
		nanovdb::io::writeGrid(filename.c_str(), densityHandle);
	}
	catch (const std::exception &e)
	{
		std::cerr << "Volume: writeToNanoVdb exception occurred: " << e.what() << std::endl;
	}
}

NORI_REGISTER_CLASS(Volume, "volume");

NORI_NAMESPACE_END