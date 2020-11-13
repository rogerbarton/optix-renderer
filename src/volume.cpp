#include "nori/volume.h"
#include <nori/object.h>
#include <filesystem/resolver.h>

#include <openvdb/tools/LevelSetSphere.h>
#include <nanovdb/util/OpenToNanoVDB.h>
#include <nanovdb/util/IO.h>

NORI_NAMESPACE_BEGIN

	struct Volume : public NoriObject
	{
		explicit Volume(const PropertyList& props)
		{
			filename = getFileResolver()->resolve(props.getString("filename"));
			if (!filename.exists())
				throw NoriException(tfm::format("PNGTexture: volume file not found %s", filename).c_str());

			const std::string extension = filename.extension();

			if (extension == "nvdb")
				loadNanoVdb();
			else if (extension == "vdb")
				loadOpenVdbToNanoVdb();
			else
				throw NoriException("Volume: file extension .%s unknown.", extension);

			std::cout << "Extension: " << extension << std::endl;
		}

		EClassType getClassType() const override { return ETexture; }

		std::string toString() const override
		{
			return tfm::format("Volume[\n"
			                   "  filename = %s,\n"
			                   "]",
			                   filename);
		};

		filesystem::path filename;
	private:

		/**
		 * Loads a .nvdb (NanoVDB) file directly
		 */
		void loadNanoVdb()
		{

		}

		/**
		 * Loads a .vdb (OpenVDB) file and converts it to the NanoVDB format
		 */
		void loadOpenVdbToNanoVdb()
		{

		}

		/**
		 * Writes the .nvdb (NanoVDB) file to disk. Use this to prevent formats converting each time.
		 */
		void writeToNanoVdb()
		{

		}
	};

	NORI_REGISTER_CLASS(Volume, "volume");

NORI_NAMESPACE_END