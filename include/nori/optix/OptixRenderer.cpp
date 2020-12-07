//
// Created by roger on 06/12/2020.
//

#include <nori/optix/OptixRenderer.h>


void nori::OptixRenderer::renderOptixState(CUDAOutputBuffer<float4>& outputBuffer)
{
	m_optixState->render(outputBuffer);
}

nori::OptixRenderer::OptixRenderer(const nori::PropertyList &propList)
{
	m_samplesPerLaunch = propList.getInteger("samplesPerLaunch", 16);
}

nori::NoriObject *nori::OptixRenderer::cloneAndInit()
{
	auto clone = new OptixRenderer(*this);
	clone->m_optixState->create(); // TODO: do this in render if state is not initalized
	return clone;
}

void nori::OptixRenderer::update(const nori::NoriObject *guiObject)
{
	const auto *gui = static_cast<const OptixRenderer *>(guiObject);
	if (!gui->touched)return;
	gui->touched = false;

	m_samplesPerLaunch = gui->m_samplesPerLaunch;
}

nori::OptixRenderer::~OptixRenderer()
{
	delete m_optixState;
}

std::string nori::OptixRenderer::toString() const
{
	return tfm::format(
			"OptixRenderer[\n"
			"  samplesPerLaunch = %i\n"
			"]",
			m_samplesPerLaunch);
}

bool nori::OptixRenderer::getImGuiNodes() { return false; }
