/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob, Romain Prévost

    Nori is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Nori is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <nori/scene.h>
#include <nori/bitmap.h>
#include <nori/integrator.h>
#include <nori/sampler.h>
#include <nori/camera.h>
#include <nori/emitter.h>
#include <nori/NvdbVolume.h>
#include <nori/bsdf.h>

#ifdef NORI_USE_OPTIX
#include <nori/optix/OptixState.h>
#endif

NORI_NAMESPACE_BEGIN

	Scene::Scene(const PropertyList &)
	{
	}

	Scene::~Scene()
	{
		delete m_bvh;
		delete m_sampler;
		delete m_camera;
		delete m_integrator;
		delete m_previewIntegrator;
		for (auto s : m_shapes)
			delete s;
		for (auto e : m_emitters)
			if (!e->hasShape())
				delete e;

		// delete m_envmap; // Already deleted as an emitter
		delete m_denoiser;
		delete m_ambientMedium;
#ifdef NORI_USE_OPTIX
		delete m_optixRenderer;
		delete m_optixState;
#endif
	}

	NoriObject *Scene::cloneAndInit()
	{
		// -- Validate scene before cloning, add defaults
		if (!m_integrator)
			throw NoriException("No integrator was specified!");
		if (!m_camera)
			throw NoriException("No camera was specified!");

		// Create a default (independent) sampler
		if (!m_sampler)
			m_sampler = static_cast<Sampler *>(NoriObjectFactory::createInstance("independent"));

		if (!m_previewIntegrator)
			m_previewIntegrator = static_cast<Integrator *>(NoriObjectFactory::createInstance("preview"));

		if (!m_ambientMedium)
			m_ambientMedium = static_cast<Medium *>(NoriObjectFactory::createInstance("vacuum"));

#ifdef NORI_USE_OPTIX
        if (!m_optixRenderer)
            m_optixRenderer = static_cast<OptixRenderer *>(NoriObjectFactory::createInstance("optix"));
#endif

		// -- Shallow copy
		Scene *clone = new Scene(*this);
		clone->m_bvh = new BVH();

		// -- Copy and initialize children
		clone->m_integrator        = static_cast<Integrator *>(m_integrator->cloneAndInit());
		clone->m_previewIntegrator = static_cast<Integrator *>(m_previewIntegrator->cloneAndInit());

		clone->m_sampler = static_cast<Sampler *>(m_sampler->cloneAndInit());
		clone->m_camera  = static_cast<Camera *>(m_camera->cloneAndInit());

		clone->m_ambientMedium = static_cast<Medium *>(m_ambientMedium->cloneAndInit());

		// envmap handled in emitters
		// if (m_envmap)
		// 	clone->m_envmap   = static_cast<Emitter *>(m_envmap->cloneAndInit());
		if (m_denoiser)
			clone->m_denoiser = static_cast<Denoiser *>(m_denoiser->cloneAndInit());

		// Workaround for having a pointer cycle
		// Create map from emitter ids to shape ids
		struct ShapeToEmitterRecord {
			int emitterId;
			int shapeId;
			bool isMediaEmitter;
		};

		for (int i = 0; i < m_shapes.size(); ++i)
			clone->m_shapes[i] = static_cast<Shape *>(m_shapes[i]->cloneAndInit());

		for (int i = 0; i < m_emitters.size(); ++i)
		{
			if (!m_emitters[i]->hasShape())
			{
				clone->m_emitters[i] = static_cast<Emitter *>(m_emitters[i]->cloneAndInit());

				// Update envmap pointer, don't clone it twice
				if (m_emitters[i]->isEnvMap())
					clone->m_envmap = clone->m_emitters[i];
			}
			else
			{
				// Link already cloned shape emitters
				int s = 0;
				for (; s < m_shapes.size(); ++s)
				{
					if (m_emitters[i] == m_shapes[s]->getEmitter())
					{
						clone->m_emitters[i] = clone->m_shapes[s]->getEmitter();
						break;
					}
					if (m_emitters[i] == m_shapes[s]->getMediumEmitter())
					{
						clone->m_emitters[i] = clone->m_shapes[s]->getMediumEmitter();
						break;
					}
				}
				if (s == m_shapes.size())
					throw NoriException("Failed to match emitter to shape");
			}
		}

#ifdef NORI_USE_VDB
		for (int i = 0; i < m_volumes.size(); ++i)
			clone->m_volumes[i] = static_cast<NvdbVolume *>(m_volumes[i]->cloneAndInit());
#endif

#ifdef NORI_USE_OPTIX
		clone->m_optixRenderer = static_cast<OptixRenderer*>(m_optixRenderer->cloneAndInit());
#endif

        // cout << endl << "Configuration: " << clone->toString() << endl << endl;

		return clone;
	}

	void Scene::update(const NoriObject *guiObject)
	{
		const auto *gui = static_cast<const Scene *>(guiObject);
		if (!gui->touched)return;
		gui->touched = false;

		// -- Update children
		m_integrator->update(gui->m_integrator);
		m_previewIntegrator->update(gui->m_previewIntegrator);

		m_sampler->update(gui->m_sampler);
		m_camera->update(gui->m_camera);
		m_ambientMedium->update(gui->m_ambientMedium);

		if (m_envmap)
			m_envmap->update(gui->m_envmap);
		if (m_denoiser)
			m_denoiser->update(gui->m_denoiser);

		for (int i = 0; i < gui->m_shapes.size(); ++i)
			m_shapes[i]->update(gui->m_shapes[i]);

		for (int i = 0; i < gui->m_emitters.size(); ++i)
			if (!m_emitters[i]->hasShape())
				m_emitters[i]->update(gui->m_emitters[i]);

		// recompute emitterDpdf
		emitterDpdf.clear();
		for (int i = 0; i < m_emitters.size(); ++i) {
			emitterDpdf.append(m_emitters[i]->lightProb);
		}
		emitterDpdf.normalize();

#ifdef NORI_USE_VDB
		for (int i = 0; i < gui->m_volumes.size(); ++i)
			m_volumes[i]->update(gui->m_volumes[i]);
#endif

		// -- Update this
		if (gui->geometryTouched || gui->transformTouched)
		{
			std::cout << "Rebuilding BVH" << std::endl;
			m_bvh->clear();
			for (const auto shape : m_shapes)
				m_bvh->addShape(shape);
			m_bvh->build();
		}

#ifdef NORI_USE_OPTIX
        m_optixRenderer->update(gui->m_optixRenderer);
#endif

        gui->geometryTouched  = false;
		gui->transformTouched = false;
	}

	void Scene::addChild(NoriObject *obj)
	{
		switch (obj->getClassType())
		{
			case EShape:
			{
				Shape *shape = static_cast<Shape *>(obj);
				m_shapes.push_back(shape);
				if (shape->getEmitter())
					m_emitters.push_back(shape->getEmitter());
				if (shape->getMediumEmitter())
					m_emitters.push_back(shape->getMedium()->getEmitter());
				break;
			}
			case EEmitter:
				m_emitters.push_back(static_cast<Emitter *>(obj));
				if (m_emitters.back()->isEnvMap())
					m_envmap = static_cast<Emitter *>(obj);
				break;

			case ESampler:
				if (m_sampler)
					throw NoriException("There can only be one sampler per scene!");
				m_sampler = static_cast<Sampler *>(obj);
				break;

			case ECamera:
				if (m_camera)
					throw NoriException("There can only be one camera per scene!");
				m_camera = static_cast<Camera *>(obj);
				break;

			case EIntegrator:
				if (m_integrator)
					throw NoriException("There can only be one integrator per scene!");
				m_integrator = static_cast<Integrator *>(obj);
				break;

			case EDenoiser:
				if (m_denoiser)
					throw NoriException("There can only be one denoiser per scene!");
				m_denoiser = static_cast<Denoiser *>(obj);
				break;

			case EMedium:
				if (m_ambientMedium)
					throw NoriException("There can only be one ambient medium per scene.");
				m_ambientMedium = static_cast<Medium *>(obj);
				break;

			case EVolume:
				// Skip if volumes are disabled
#ifdef NORI_USE_VDB
				m_volumes.push_back(static_cast<NvdbVolume *>(obj));
#endif
				break;

#ifdef NORI_USE_OPTIX
            case ERenderer:
                if (m_optixRenderer)
                    throw NoriException("There can only be one renderer per scene.");
                m_optixRenderer = static_cast<OptixRenderer *>(obj);
                break;
#endif

			default:
				throw NoriException("Scene::addChild(<%s>) is not supported!",
				                    classTypeName(obj->getClassType()));
		}
	}

	std::string Scene::toString() const
	{
		std::string shapes;
		for (size_t i = 0; i < m_shapes.size(); ++i)
		{
			shapes += std::string("  ") + indent(m_shapes[i]->toString(), 2);
			if (i + 1 < m_shapes.size())
				shapes += ",";
			shapes += "\n";
		}

		std::string lights;
		for (size_t i = 0; i < m_emitters.size(); ++i)
		{
			lights += std::string("  ") + indent(m_emitters[i]->toString(), 2);
			if (i + 1 < m_emitters.size())
				lights += ",";
			lights += "\n";
		}

#ifdef NORI_USE_VDB
		std::string volumes;
		for (size_t i = 0; i < m_volumes.size(); ++i)
		{
			volumes += std::string("  ") + indent(m_volumes[i]->toString(), 2);
			if (i + 1 < m_volumes.size())
				volumes += ",";
			volumes += "\n";
		}
#endif

		return tfm::format(
				"Scene[\n"
				"  integrator = %s,\n"
				"  sampler = %s\n"
				"  camera = %s,\n"
				"  shapes = {\n"
				"  %s  }\n"
				"  emitters = {\n"
				"  %s  }\n"
				"  envmap = %s,\n"
				"  denoiser = %s,\n"
				"  ambient medium = %s,\n"
				"  volumes {\n"
				"  %s  }\n"
				"]",
				indent(m_integrator->toString()),
				indent(m_sampler->toString()),
				indent(m_camera->toString()),
				indent(shapes, 2),
				indent(lights, 2),
				m_envmap ? indent(m_envmap->toString()) : "nullptr",
				m_denoiser ? indent(m_denoiser->toString()) : "nullptr",
				indent(m_ambientMedium->toString()),
#ifdef NORI_USE_VDB
				indent(volumes, 2)
#else
				"Volumes disabled during compilation"
#endif
		);
	}

#ifdef NORI_USE_IMGUI
	bool Scene::getImGuiNodes()
	{
		ImGui::PushID(EScene);
#ifdef NORI_USE_OPTIX
		NORI_IMGUI_CHILD_OBJECT(m_optixRenderer, "Renderer")
#endif

		NORI_IMGUI_CHILD_OBJECT(m_camera, "Camera")
		NORI_IMGUI_CHILD_OBJECT(m_integrator, "Integrator")
		NORI_IMGUI_CHILD_OBJECT(m_sampler, "Sampler")
		NORI_IMGUI_CHILD_OBJECT(m_envmap, "Environment Map")
		NORI_IMGUI_CHILD_OBJECT(m_denoiser, "Denoiser")
		NORI_IMGUI_CHILD_OBJECT(m_ambientMedium, "Ambient Medium")

		ImGui::Separator();
		ImGui::NextColumn();
		ImGui::NextColumn();

		bool node_open_shapes = ImGui::TreeNode("Shapes");
		ImGui::NextColumn();
		ImGui::AlignTextToFramePadding();
		ImGui::Text(m_shapes.size() == 1 ? "%d Shape" : "%d Shapes", (int) m_shapes.size());
		ImGui::NextColumn();
		if (node_open_shapes)
		{
			for (int i = 0; i < m_shapes.size(); i++)
			{
				ImGui::PushID(i);

				// for each shape, add a tree node
				bool node_open_shape = ImGui::TreeNode("Shape", "Shape %d", i + 1);
				ImGui::NextColumn();
				ImGui::AlignTextToFramePadding();

				ImGui::Text(m_shapes[i]->getImGuiName().c_str());
				ImGui::NextColumn();

				if (node_open_shape)
				{
					touched |= m_shapes[i]->getImGuiNodes();
					transformTouched |= m_shapes[i]->transformTouched;
					geometryTouched |= m_shapes[i]->geometryTouched;

					ImGui::TreePop();
				}

				ImGui::PopID();
			}

			ImGui::TreePop();
		}

		bool node_open_emitters = ImGui::TreeNode("Emitters");
		ImGui::NextColumn();
		ImGui::AlignTextToFramePadding();
		ImGui::Text(m_emitters.size() == 1 ? "%d Emitter" : "%d Emitters", (int) m_emitters.size());
		ImGui::NextColumn();
		if (node_open_emitters)
		{
			for (int i = 0; i < m_emitters.size(); i++)
			{
				ImGui::PushID(i);
				NORI_IMGUI_CHILD_OBJECT(m_emitters[i], "Emitter", "Emitter %d", i + 1)
				ImGui::PopID();
			}

			ImGui::TreePop();
		}

		ImGui::PopID();

		return touched;
	}
#endif

#ifdef NORI_USE_OPTIX
	OptixState *Scene::getOptixState()
	{
		if (m_optixState == nullptr)
			m_optixState = new OptixState{};
		return  m_optixState;
	}
#endif

	NORI_REGISTER_CLASS(Scene, "scene");
NORI_NAMESPACE_END
