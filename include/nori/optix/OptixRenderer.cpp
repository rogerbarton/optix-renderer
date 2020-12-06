//
// Created by roger on 06/12/2020.
//

#include <nori/optix/OptixRenderer.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <cuda.h>
#include <cuda.h>
#include <nvrtc.h>
#include <cuda_runtime_api.h>

#include "sutil/Exception.h"

#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <map>

static void createContext(OptixState& state);

static void buildGases(OptixState& state);

static void buildIas(OptixState& state);

static void createPtxModules(OptixState& state);

static void createPipeline(OptixState& state);

static void createCameraProgram(OptixState& state, std::vector<OptixProgramGroup> vector);

static void createHitProgram(OptixState& state, std::vector<OptixProgramGroup> vector);

static void createVolumeProgram(OptixState& state, std::vector<OptixProgramGroup> vector);

static void createMissProgram(OptixState& state, std::vector<OptixProgramGroup> vector);

static void createSbt(OptixState& state);

OptixState* nori::OptixRenderer::createOptixState() {
    OptixState* state = new OptixState();

    createContext(*state);
    buildGases(*state);
    buildIas(*state);
    createPtxModules(*state);
    createPipeline(*state);
    createSbt(*state);

    std::cout << "Optix state created.\n";
}

static void optixLogCallback(unsigned int level, const char* tag, const char* message, void* /*cbdata */) {
    std::cerr << "[" << std::setw(2) << level << "|" << std::setw(12) << tag << "]: " << message << "\n";
}

/**
 * Initializes cuda and optix
 * @param state Sets state.context
 */
static void createContext(OptixState& state) {
    CUDA_CHECK(cudaFree(nullptr));
    CUcontext cuCtx = 0;

    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &optixLogCallback;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &state.context));

    // Print device debug info
    int32_t deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    std::cout << "Total GPUs available: " << deviceCount << std::endl;

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop = {};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        std::cout << "\t[" << i << "]: " << prop.name << " (" << prop.totalGlobalMem / 1024 / 1024 << "MB)"
                  << std::endl;
    }
}

static void buildGases(OptixState& state) {

}

static void buildIas(OptixState& state) {

}

static bool readSourceFile(std::string& str, const std::string& filename) {
    // Try to open file
    std::ifstream file(filename.c_str());
    if (file.good()) {
        // Found usable source file
        std::stringstream source_buffer;
        source_buffer << file.rdbuf();
        str = source_buffer.str();
        return true;
    }
    return false;
}

static void getCuStringFromFile(std::string& cu, std::string& location, const char* filename) {
    std::vector<std::string> source_locations;

    std::string base_dir = std::string(__FILE__);
    base_dir = base_dir.substr(0, base_dir.find_last_of('/')).substr(0, base_dir.find_last_of('\\')) + "/";

    source_locations.push_back(base_dir + filename);

    for (const std::string& loc : source_locations) {
        // Try to get source code from file
        if (readSourceFile(cu, loc)) {
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


static void getPtxFromCuString(std::string& ptx, const char* cu_source, const char* name, const char** log_string) {
    // Create program
    nvrtcProgram prog = 0;
    NVRTC_CHECK_ERROR(nvrtcCreateProgram(&prog, cu_source, name, 0, NULL, NULL));

    // Gather NVRTC options
    std::vector<const char*> options;

    std::string base_dir = std::string(__FILE__);
    base_dir = base_dir.substr(0, base_dir.find_last_of('/')).substr(0, base_dir.find_last_of('\\')) + "/";

    // Collect include dirs
    std::vector<std::string> include_dirs;
    const char* abs_dirs[] = {NANOVDB_OPTIX_ABSOLUTE_INCLUDE_DIRS};
    const char* rel_dirs[] = {NANOVDB_OPTIX_RELATIVE_INCLUDE_DIRS};

    for (const char* dir : abs_dirs) {
        include_dirs.push_back(std::string("--include-path=") + dir);
    }

    for (const char* dir : rel_dirs) {
        include_dirs.push_back("--include-path=" + base_dir + dir);
    }

    for (const std::string& dir : include_dirs) {
        options.push_back(dir.c_str());
    }

    // NVRTC options
#ifdef _WIN32
    const char* compiler_options[] = {NANOVDB_CUDA_NVRTC_OPTIONS "-DNANOVDB_OPTIX_RTC_WIN32"};
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
    if (log_size > 1) {
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

struct PtxSourceCache {
    std::map<std::string, std::string*> map;

    ~PtxSourceCache() {
        for (std::map<std::string, std::string*>::const_iterator it = map.begin(); it != map.end(); ++it)
            delete it->second;
    }
};

static PtxSourceCache g_ptxSourceCache;

const char* getPtxString(const char* filename, const char** log = NULL) {
    if (log)
        *log = NULL;

    std::string* ptx, cu;
    std::string key = std::string(filename);
    std::map<std::string, std::string*>::iterator elem = g_ptxSourceCache.map.find(key);

    if (elem == g_ptxSourceCache.map.end()) {
        ptx = new std::string();
        std::string location;
        getCuStringFromFile(cu, location, filename);
        getPtxFromCuString(*ptx, cu.c_str(), location.c_str(), log);
        g_ptxSourceCache.map[key] = ptx;
    } else {
        ptx = elem->second;
    }

    return ptx->c_str();
}

/**
 * Compiles the .cu to .ptx using nvrtc then compiles this to a module using optixModuleCreateFromPTX
 * @param state sets *_module
 */
static void createPtxModules(OptixState& state) {
    OptixModuleCompileOptions module_compile_options;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    {
        // Path relative to current file __FILE__
        const std::string ptx = getPtxString("cuda/geometry.cu");
        OPTIX_CHECK_LOG2(optixModuleCreateFromPTX(state.context,
                                                  &module_compile_options,
                                                  &state.pipeline_compile_options,
                                                  ptx.c_str(),
                                                  ptx.size(),
                                                  LOG,
                                                  &LOG_SIZE,
                                                  &state.geometry_module));
    }

    {
        const std::string ptx = getPtxString("cuda/camera.cu");
        OPTIX_CHECK_LOG2(optixModuleCreateFromPTX(state.context,
                                                  &module_compile_options,
                                                  &state.pipeline_compile_options,
                                                  ptx.c_str(),
                                                  ptx.size(),
                                                  LOG,
                                                  &LOG_SIZE,
                                                  &state.camera_module));
    }

    {
        const std::string ptx = getPtxString("cuda/shading.cu");
        OPTIX_CHECK_LOG2(optixModuleCreateFromPTX(state.context,
                                                  &module_compile_options,
                                                  &state.pipeline_compile_options,
                                                  ptx.c_str(),
                                                  ptx.size(),
                                                  LOG,
                                                  &LOG_SIZE,
                                                  &state.shading_module));
    }
}

/**
 * Creates program using the modules and creates the whole pipeline
 * @param state sets prog_groups and pipeline
 */
static void createPipeline(OptixState& state) {
    std::vector<OptixProgramGroup> program_groups;

    state.pipeline_compile_options = {};
    state.pipeline_compile_options.usesMotionBlur = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    state.pipeline_compile_options.numPayloadValues = 3;
    state.pipeline_compile_options.numAttributeValues = 6;
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "launchParams";

    // Prepare program groups
    createCameraProgram(state, program_groups);
    createHitProgram(state, program_groups);
    createVolumeProgram(state, program_groups);
    createMissProgram(state, program_groups);

    // Link program groups to pipeline
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = state.maxTraceDepth;
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    OPTIX_CHECK_LOG2(optixPipelineCreate(state.context,
                                         &state.pipeline_compile_options,
                                         &pipeline_link_options,
                                         program_groups.data(),
                                         static_cast<unsigned int>(program_groups.size()),
                                         LOG,
                                         &LOG_SIZE,
                                         &state.pipeline));
}

static void createCameraProgram(OptixState& state, std::vector<OptixProgramGroup> program_groups) {
    OptixProgramGroup cam_prog_group;
    OptixProgramGroupOptions cam_prog_group_options = {};
    OptixProgramGroupDesc cam_prog_group_desc = {};
    cam_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    cam_prog_group_desc.raygen.module = state.camera_module;
    cam_prog_group_desc.raygen.entryFunctionName = "__raygen__perspective";

    OPTIX_CHECK_LOG2(optixProgramGroupCreate(
            state.context, &cam_prog_group_desc, 1, &cam_prog_group_options, LOG, &LOG_SIZE, &cam_prog_group));

    program_groups.push_back(cam_prog_group);
    state.raygen_prog_group = cam_prog_group;

}

static void createHitProgram(OptixState& state, std::vector<OptixProgramGroup> program_groups) {

}

static void createVolumeProgram(OptixState& state, std::vector<OptixProgramGroup> program_groups) {

    {
        OptixProgramGroup radiance_prog_group;
        OptixProgramGroupOptions radiance_prog_group_options = {};
        OptixProgramGroupDesc radiance_prog_group_desc = {};
        radiance_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
                radiance_prog_group_desc.hitgroup.moduleIS = state.geometry_module;
        radiance_prog_group_desc.hitgroup.moduleCH = state.shading_module;
        radiance_prog_group_desc.hitgroup.moduleAH = nullptr;

        radiance_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__nanovdb_fogvolume";
        radiance_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__nanovdb_fogvolume_radiance";
        radiance_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

        OPTIX_CHECK_LOG2(optixProgramGroupCreate(state.context,
                                                 &radiance_prog_group_desc,
                                                 1,
                                                 &radiance_prog_group_options,
                                                 LOG,
                                                 &LOG_SIZE,
                                                 &radiance_prog_group));

        program_groups.push_back(radiance_prog_group);
        state.volume_prog_group[RAY_TYPE_RADIANCE] = radiance_prog_group;
    }

    {
        OptixProgramGroup occlusion_prog_group;
        OptixProgramGroupOptions occlusion_prog_group_options = {};
        OptixProgramGroupDesc occlusion_prog_group_desc = {};
        occlusion_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
                occlusion_prog_group_desc.hitgroup.moduleIS = state.geometry_module;
        occlusion_prog_group_desc.hitgroup.moduleCH = nullptr;
        occlusion_prog_group_desc.hitgroup.moduleAH = nullptr;
        occlusion_prog_group_desc.hitgroup.entryFunctionNameCH = nullptr;
        occlusion_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

        occlusion_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__nanovdb_fogvolume";
        occlusion_prog_group_desc.hitgroup.moduleCH = state.shading_module;
        occlusion_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__nanovdb_fogvolume_occlusion";

        OPTIX_CHECK_LOG2(optixProgramGroupCreate(
                state.context, &occlusion_prog_group_desc, 1, &occlusion_prog_group_options, LOG, &LOG_SIZE,
                &occlusion_prog_group));

        program_groups.push_back(occlusion_prog_group);
        state.volume_prog_group[RAY_TYPE_OCCLUSION] = occlusion_prog_group;
    }
}

static void createMissProgram(OptixState& state, std::vector<OptixProgramGroup> program_groups) {

    {
        OptixProgramGroup miss_prog_group;
        OptixProgramGroupOptions miss_prog_group_options = {};
        OptixProgramGroupDesc miss_prog_group_desc = {};

        miss_prog_group_desc.miss = {
                nullptr, // module
                nullptr // entryFunctionName
        };

        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = state.shading_module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__fogvolume_radiance";

        OPTIX_CHECK_LOG2(optixProgramGroupCreate(state.context,
                                                 &miss_prog_group_desc,
                                                 1,
                                                 &miss_prog_group_options,
                                                 LOG,
                                                 &LOG_SIZE,
                                                 &state.miss_prog_group[RAY_TYPE_RADIANCE]));
    }

    {
        OptixProgramGroup miss_prog_group;
        OptixProgramGroupOptions miss_prog_group_options = {};
        OptixProgramGroupDesc miss_prog_group_desc = {};

        miss_prog_group_desc.miss = {
                nullptr, // module
                nullptr // entryFunctionName
        };

        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = state.shading_module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__occlusion";

        OPTIX_CHECK_LOG2(optixProgramGroupCreate(state.context,
                                                 &miss_prog_group_desc,
                                                 1,
                                                 &miss_prog_group_options,
                                                 LOG,
                                                 &LOG_SIZE,
                                                 &state.miss_prog_group[RAY_TYPE_OCCLUSION]));
    }
}

static void createSbt(OptixState& state) {

    // Raygen program record
    {
        CUdeviceptr d_raygen_record;
        size_t      sizeof_raygen_record = sizeof(RayGenRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), sizeof_raygen_record));

        state.sbt.raygenRecord = d_raygen_record;
    }

    // Miss program record
    {
        CUdeviceptr d_miss_record;
        size_t      sizeof_miss_record = sizeof(MissRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_record), sizeof_miss_record * RAY_TYPE_COUNT));

        MissRecord ms_sbt[RAY_TYPE_COUNT];
        for (int i = 0; i < RAY_TYPE_COUNT; ++i) {
            optixSbtRecordPackHeader(state.miss_prog_group[i], &ms_sbt[i]);
            // data for miss program goes in here...
            //ms_sbt[i].data = {0.f, 0.f, 0.f};
        }

        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_miss_record),
                                          ms_sbt,
                                          sizeof_miss_record * RAY_TYPE_COUNT,
                                          cudaMemcpyHostToDevice));

        state.sbt.missRecordBase = d_miss_record;
        state.sbt.missRecordCount = RAY_TYPE_COUNT;
        state.sbt.missRecordStrideInBytes = static_cast<uint32_t>(sizeof_miss_record);
    }

    // Hitgroup program record
    {
        const size_t                count_records = RAY_TYPE_COUNT;
        std::vector<HitGroupRecord> hitgroup_records(RAY_TYPE_COUNT);

        {
            int sbt_idx = 0;
            OPTIX_CHECK(optixSbtRecordPackHeader(state.volume_prog_group[RAY_TYPE_RADIANCE], &hitgroup_records[sbt_idx]));
            hitgroup_records[sbt_idx].data.geometry.volume = geometry;
            hitgroup_records[sbt_idx].data.shading.volume = material;
            sbt_idx++;

            OPTIX_CHECK(optixSbtRecordPackHeader(state.volume_prog_group[RAY_TYPE_OCCLUSION], &hitgroup_records[sbt_idx]));
            hitgroup_records[sbt_idx].data.geometry.volume = geometry;
            hitgroup_records[sbt_idx].data.shading.volume = material;
            sbt_idx++;
        }

        CUdeviceptr d_hitgroup_records;
        size_t      sizeof_hitgroup_record = sizeof(HitGroupRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_records), sizeof_hitgroup_record * count_records));

        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hitgroup_records),
                                          hitgroup_records.data(),
                                          sizeof_hitgroup_record * count_records,
                                          cudaMemcpyHostToDevice));

        state.sbt.hitgroupRecordBase = d_hitgroup_records;
        state.sbt.hitgroupRecordCount = count_records;
        state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(sizeof_hitgroup_record);
    }
}

void nori::OptixRenderer::renderOptixState(OptixState* state) {
    if(state->ias_handle == 0) {
        std::cerr << "renderOptixState: state is not initialized.\n";
        return;
    }
    // TODO: map output buffer
    // TODO: update device copies, launch params etc

    OPTIX_CHECK(optixLaunch(state->pipeline,
                            state->stream,
                            reinterpret_cast<CUdeviceptr>(state->d_params),
                            sizeof(LaunchParams),
                            &state->sbt,
                            width,
                            height,
                            1));

    CUDA_CHECK(cudaStreamSynchronize(state->stream));

    // TODO: unmap output buffer
}


