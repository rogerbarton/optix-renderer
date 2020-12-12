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
		const auto camera = static_cast<const nori::PerspectiveCamera *>(scene.getCamera());
		m_params.imageWidth  = camera->getOutputSize().x();
		m_params.imageHeight = camera->getOutputSize().y();

		Eigen::Matrix4f cameraToWorld = camera->getTransform();
		// W = z = fwd
		m_params.camera.U   = make_float3(cameraToWorld(0, 0), cameraToWorld(0, 1), cameraToWorld(0, 2));
		m_params.camera.V   = make_float3(cameraToWorld(1, 0), cameraToWorld(1, 1), cameraToWorld(1, 2));
		m_params.camera.W   = make_float3(cameraToWorld(2, 0), cameraToWorld(2, 1), cameraToWorld(2, 2));
		m_params.camera.eye = make_float3(cameraToWorld(0, 3), cameraToWorld(1, 3), cameraToWorld(2, 3));

		m_params.camera.fov           = camera->getFov();
		m_params.camera.focalDistance = camera->getFocalDistance();
		m_params.camera.lensRadius    = camera->getLensRadius();
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

// NORI_NAMESPACE_BEGIN
// 	void Scene::updateOptix()
// 	{
// 		if (!touchedOptix) return;
// 		touchedOptix = false;
//
// 		if (m_camera->touchedOptix)
// 		{
// 			m_camera->touchedOptix = false;
// 			m_optixState->m_params.camera.eye = m_camera.eye;
// 			m_optixState->m_params.camera.U = m_camera.U;
// 			m_optixState->m_params.camera.V = m_camera.V;
// 			m_optixState->m_params.camera.W = m_camera.W;
// 		}
//
// 		for (int i = 0; i < m_shapes.size(); ++i)
// 		{
// 			if (!m_shapes[i]->touchedOptix) continue;
// 			m_shapes[i]->touchedOptix = false;
//
//
// 		}
//
// 		for (int i = 0; i < m_emitters.size(); ++i)
// 		{
// 			if (!m_emitters[i]->touchedOptix) continue;
// 			m_emitters[i]->touchedOptix = false;
//
//
// 		}
//
// 		// rebuild IAS if required
// 		// recompile modules if required
// 			// bind compile vars if required
// 		// sbt?
// 	}
// NORI_NAMESPACE_END
