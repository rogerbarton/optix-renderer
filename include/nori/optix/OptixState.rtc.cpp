//
// Created by roger on 06/12/2020.
//

/**
 * This file contains all runtime compilation (rtc) parts for OptixState
 * Compiles .cu files to .ptx files with nvrtc
 * Compiles .ptx files to optix modules with optixModuleCreateFromPTX
 */

#include <nori/optix/OptixRenderer.h>

#include <optix.h>
#include <optix_stubs.h>

#include <cuda.h>
#include <nvrtc.h>

#include "sutil/Exception.h"

#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <map>

static bool readSourceFile(std::string &str, const std::string &filename)
{
	// Try to open file
	std::ifstream file(filename.c_str());
	if (file.good())
	{
		// Found usable source file
		std::stringstream source_buffer;
		source_buffer << file.rdbuf();
		str = source_buffer.str();
		return true;
	}
	return false;
}

static void getCuStringFromFile(std::string &cu, std::string &location, const char *filename)
{
	std::vector<std::string> source_locations;

	std::string base_dir = std::string(__FILE__);
	base_dir = base_dir.substr(0, base_dir.find_last_of('/')).substr(0, base_dir.find_last_of('\\')) + "/";

	source_locations.push_back(base_dir + filename);

	for (const std::string &loc : source_locations)
	{
		// Try to get source code from file
		if (readSourceFile(cu, loc))
		{
			location = loc;
			return;
		}
	}

	// Wasn't able to find or open the requested file
	throw std::runtime_error("Couldn't open source file " + std::string(filename));
}

static std::string g_nvrtcLog;
#define NANOVDB_OPTIX_RELATIVE_INCLUDE_DIRS \
    "optix", \
    "optix/cuda", \
        "../../nanovdb", \
        "../..", \
        "..", \
        ".",

// These must be defined or NVRTC will fail to compile optix programs.
// CMake will define them automatically.
#define NANOVDB_OPTIX_ABSOLUTE_INCLUDE_DIRS \
    NORI_NVRTC_OPTIX_DIR, \
    NORI_NVRTC_CUDA_DIR

#define NANOVDB_CUDA_NVRTC_OPTIONS \
    "--std=c++11", \
        "-arch", \
        "compute_60", \
        "-use_fast_math", \
        "-lineinfo", \
        "-default-device", \
        "-rdc", \
        "true", \
        "-D__x86_64",


static void getPtxFromCuString(std::string &ptx, const char *cu_source, const char *name, const char **log_string)
{
	// Create program
	nvrtcProgram prog = 0;
	NVRTC_CHECK_ERROR(nvrtcCreateProgram(&prog, cu_source, name, 0, NULL, NULL));

	// Gather NVRTC options
	std::vector<const char *> options;

	std::string base_dir = std::string(__FILE__);
	base_dir                             =
			base_dir.substr(0, base_dir.find_last_of('/')).substr(0, base_dir.find_last_of('\\')) + "/";

	// Collect include dirs
	std::vector<std::string> include_dirs;
	const char               *abs_dirs[] = {NANOVDB_OPTIX_ABSOLUTE_INCLUDE_DIRS};
	const char               *rel_dirs[] = {NANOVDB_OPTIX_RELATIVE_INCLUDE_DIRS};

	for (const char *dir : abs_dirs)
	{
		include_dirs.push_back(std::string("--include-path=") + dir);
	}

	for (const char *dir : rel_dirs)
	{
		include_dirs.push_back("--include-path=" + base_dir + dir);
	}

	for (const std::string &dir : include_dirs)
	{
		options.push_back(dir.c_str());
	}

	// NVRTC options
#ifdef _WIN32
	const char *compiler_options[] = {NANOVDB_CUDA_NVRTC_OPTIONS "-DNANOVDB_OPTIX_RTC_WIN32"};
#else
	const char* compiler_options[] = {NANOVDB_CUDA_NVRTC_OPTIONS};
#endif
	std::copy(std::begin(compiler_options), std::end(compiler_options), std::back_inserter(options));

	// JIT compile CU to PTX
	const nvrtcResult compileRes = nvrtcCompileProgram(prog, (int) options.size(), options.data());

	// Retrieve log output
	size_t log_size = 0;
	NVRTC_CHECK_ERROR(nvrtcGetProgramLogSize(prog, &log_size));
	g_nvrtcLog.resize(log_size);
	if (log_size > 1)
	{
		NVRTC_CHECK_ERROR(nvrtcGetProgramLog(prog, &g_nvrtcLog[0]));
		if (log_string)
			*log_string = g_nvrtcLog.c_str();
	}
	if (compileRes != NVRTC_SUCCESS)
		throw std::runtime_error("NVRTC Compilation failed.\n" + g_nvrtcLog);

	// Retrieve PTX code
	size_t ptx_size = 0;
	NVRTC_CHECK_ERROR(nvrtcGetPTXSize(prog, &ptx_size));
	ptx.resize(ptx_size);
	NVRTC_CHECK_ERROR(nvrtcGetPTX(prog, &ptx[0]));

	// Cleanup
	NVRTC_CHECK_ERROR(nvrtcDestroyProgram(&prog));
}

struct PtxSourceCache
{
	std::map<std::string, std::string *> map;

	~PtxSourceCache()
	{
		for (std::map<std::string, std::string *>::const_iterator it = map.begin(); it != map.end(); ++it)
			delete it->second;
	}
};

static PtxSourceCache g_ptxSourceCache;

const char *getPtxString(const char *filename, const char **log = NULL)
{
	if (log)
		*log                                            = NULL;

	std::string                                    *ptx, cu;
	std::string                                    key  = std::string(filename);
	std::map<std::string, std::string *>::iterator elem = g_ptxSourceCache.map.find(key);

	if (elem == g_ptxSourceCache.map.end())
	{
		ptx = new std::string();
		std::string location;
		getCuStringFromFile(cu, location, filename);
		getPtxFromCuString(*ptx, cu.c_str(), location.c_str(), log);
		g_ptxSourceCache.map[key] = ptx;
	}
	else
	{
		ptx = elem->second;
	}

	return ptx->c_str();
}

/**
 * Compiles the .cu to .ptx using nvrtc then compiles this to a module using optixModuleCreateFromPTX
 * @param state sets *_module
 */
void OptixState::createPtxModules()
{
	// Note: module_compile_options set in create()
	// .cu paths relative to current file __FILE__
	{
		const std::string ptx = getPtxString("cuda/geometry.cpp");
		OPTIX_CHECK_LOG2(optixModuleCreateFromPTX(m_context,
		                                          &m_module_compile_options,
		                                          &m_pipeline_compile_options,
		                                          ptx.c_str(),
		                                          ptx.size(),
		                                          LOG,
		                                          &LOG_SIZE,
		                                          &m_geometry_module));
	}

	{
		const std::string ptx = getPtxString("cuda/camera.cpp");
		OPTIX_CHECK_LOG2(optixModuleCreateFromPTX(m_context,
		                                          &m_module_compile_options,
		                                          &m_pipeline_compile_options,
		                                          ptx.c_str(),
		                                          ptx.size(),
		                                          LOG,
		                                          &LOG_SIZE,
		                                          &m_camera_module));
	}

	{
		const std::string ptx = getPtxString("cuda/shading.cpp");
		OPTIX_CHECK_LOG2(optixModuleCreateFromPTX(m_context,
		                                          &m_module_compile_options,
		                                          &m_pipeline_compile_options,
		                                          ptx.c_str(),
		                                          ptx.size(),
		                                          LOG,
		                                          &LOG_SIZE,
		                                          &m_shading_module));
	}
}