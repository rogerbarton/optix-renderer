//
// Created by roger on 06/12/2020.
//

#include <nori/optix/OptixRenderer.h>

NORI_NAMESPACE_BEGIN

	OptixRenderer::OptixRenderer(const PropertyList &propList)
	{
		m_enabled          = propList.getBoolean("enabled", true);
		m_samplesPerLaunch = propList.getInteger("samplesPerLaunch", 16);
	}

	NoriObject *OptixRenderer::cloneAndInit()
	{
		auto clone = new OptixRenderer(*this);
		return clone;
	}

	void OptixRenderer::update(const NoriObject *guiObject)
	{
		const auto *gui = static_cast<const OptixRenderer *>(guiObject);
		if (!gui->touched)return;
		gui->touched = false;

		m_samplesPerLaunch = gui->m_samplesPerLaunch;
	}

	std::string OptixRenderer::toString() const
	{
		return tfm::format(
				"OptixRenderer[\n"
				"  samplesPerLaunch = %i\n"
				"]",
				m_samplesPerLaunch);
	}

	bool OptixRenderer::getImGuiNodes() { return false; }

NORI_NAMESPACE_END
