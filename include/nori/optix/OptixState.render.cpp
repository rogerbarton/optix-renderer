//
// Created by roger on 07/12/2020.
//

#include <nori/optix/OptixState.h>

#include <nori/scene.h>
#include <nori/integrator.h>
#include <nori/sampler.h>
#include <nori/camera.h>
#include <nori/perspective.h>
#include <nori/emitter.h>
#include <nori/NvdbVolume.h>
#include <nori/bsdf.h>


bool OptixState::preRender(nori::Scene &scene, bool usePreview)
{
	// -- Iterate over all scene objects and copy them to the optix scene representation
	// Camera
	{
		const auto camera = scene.getCamera();
		m_params.imageWidth  = camera->getOutputSize().x();
		m_params.imageHeight = camera->getOutputSize().y();
		camera->getOptixData(m_params.camera);
	}

	// OptixRenderer
	{
		m_params.samplesPerLaunch = scene.m_optixRenderer->m_samplesPerLaunch;
	}

	// Integrator
	{
		m_params.integrator = scene.getIntegrator(usePreview)->getOptixIntegratorType();
	}

	// -- Create optix scene
	createContext();
	createCompileOptions();

	bool rebuildGases = true;
	bool rebuildIas   = true;
	if (rebuildGases)
		buildGases(scene.getShapes());
	if (rebuildIas)
		buildIas();

	bool recompile = true;
	if (recompile)
	{
		if (initializedState)
			clearPipeline();
		createPtxModules(scene.m_optixRenderer->m_enableSpecialization);
		createPipeline();
	}
	updateSbt(scene.getShapes());

	m_params.sceneHandle = m_ias_handle;

	if (!initializedState)
	{
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_d_params), sizeof(LaunchParams)));
	}

	CUDA_SYNC_CHECK();
	std::cout << "Optix state created.\n";
	initializedState = true;
	return true;
}
