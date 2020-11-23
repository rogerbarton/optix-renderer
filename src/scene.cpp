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
#include <nori/volume.h>
#include <nori/bsdf.h>

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
		for (auto s : m_shapes)
			delete s;
		for (auto e : m_emitters)
			if (!e->hasShape())
				delete e;

		// delete m_envmap; // Already deleted as an emitter
		delete m_denoiser;

		delete m_previewIntegrator;
	}

	NoriObject *Scene::cloneAndInit()
	{
		// -- Validate scene before cloning
		if (!m_integrator)
			throw NoriException("No integrator was specified!");
		if (!m_camera)
			throw NoriException("No camera was specified!");

		if (!m_sampler)
		{
			// Create a default (independent) sampler
			m_sampler = static_cast<Sampler *>(
					NoriObjectFactory::createInstance("independent", PropertyList()));
		}

		if (!m_previewIntegrator)
		{
			m_previewIntegrator = static_cast<Integrator *>(
					NoriObjectFactory::createInstance("preview", PropertyList()));
		}

		Scene *clone = new Scene(*this);
		clone->m_bvh = new BVH();

		// -- Copy and initialize children
		clone->m_integrator        = static_cast<Integrator *>(m_integrator->cloneAndInit());
		clone->m_previewIntegrator = static_cast<Integrator *>(m_previewIntegrator->cloneAndInit());

		clone->m_sampler = static_cast<Sampler *>(m_sampler->cloneAndInit());
		clone->m_camera  = static_cast<Camera *>(m_camera->cloneAndInit());

		// envmap handled in emitters
		// if (m_envmap)
		// 	clone->m_envmap   = static_cast<Emitter *>(m_envmap->cloneAndInit());
		if (m_denoiser)
			clone->m_denoiser = static_cast<Denoiser *>(m_denoiser->cloneAndInit());

		// Workaround for having a pointer cycle
		// Create map from emitter ids to shape ids
		std::map<int, int> shapeToEmitter{};
		for (int           i = 0; i < m_emitters.size(); ++i)
			if (!m_emitters[i]->hasShape())
			{
				clone->m_emitters[i] = static_cast<Emitter *>(m_emitters[i]->cloneAndInit());

				// Update envmap pointer, don't clone it twice
				if(m_emitters[i]->isEnvMap())
					clone->m_envmap = clone->m_emitters[i];
			}
			else
			{
				int m = 0;
				while(m_emitters[i] != m_shapes[m]->getEmitter())
					m++;
				shapeToEmitter.insert({m, i});
			}

		for (int i = 0; i < m_shapes.size(); ++i)
		{
			clone->m_shapes[i] = static_cast<Shape *>(m_shapes[i]->cloneAndInit());

			// Update shape pointer in emitter (!)
			if (clone->m_shapes[i]->isEmitter())
				m_emitters[shapeToEmitter[i]]->setShape(clone->m_shapes[i]);
		}

#ifdef NORI_USE_VOLUMES
		for (int i = 0; i < m_volumes.size(); ++i)
			clone->m_volumes[i] = static_cast<Volume *>(m_volumes[i]->cloneAndInit());
#endif

		cout << endl << "Configuration: " << clone->toString() << endl << endl;

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

		if (m_envmap)
			m_envmap->update(gui->m_envmap);
		if (m_denoiser)
			m_denoiser->update(gui->m_denoiser);

		for (int i = 0; i < gui->m_shapes.size(); ++i)
			m_shapes[i]->update(gui->m_shapes[i]);

		for (int i = 0; i < gui->m_emitters.size(); ++i)
			m_emitters[i]->update(gui->m_emitters[i]);

#ifdef NORI_USE_VOLUMES
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
				if (shape->isEmitter())
					m_emitters.push_back(shape->getEmitter());
			}
				break;

			case EEmitter:
				m_emitters.push_back(static_cast<Emitter *>(obj));
	      	if(m_emitters.back()->isEnvMap())
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
         /*
			case EEnvironmentMap:
				if (m_envmap)
					throw NoriException("There can only be one environment map per scene!");
				m_envmap = static_cast<EnvironmentMap *>(obj);
				break;*/
			case EDenoiser:
				if (m_denoiser)
					throw NoriException("There can only be one denoiser per scene!");
				m_denoiser = static_cast<Denoiser *>(obj);
				break;

			case EVolume:
				// Skip if volumes are disabled
#ifdef NORI_USE_VOLUMES
				m_volumes.push_back(static_cast<Volume *>(obj));
#endif
				break;

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

#ifdef NORI_USE_VOLUMES
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
#ifdef NORI_USE_VOLUMES
				indent(volumes, 2)
#else
				"Volumes disabled during compilation"
#endif
		);
	}

#ifndef NORI_USE_NANOGUI

	bool Scene::getImGuiNodes()
	{
		ImGui::PushID(EScene);
		if (m_camera)
		{
			bool node_open_camera = ImGui::TreeNode("Camera");
			ImGui::NextColumn();
			ImGui::AlignTextToFramePadding();
			ImGui::Text(m_camera->getImGuiName().c_str());
			ImGui::NextColumn();
			if (node_open_camera)
			{
				touched |= m_camera->getImGuiNodes();
				ImGui::TreePop();
			}
		}

		if (m_integrator)
		{
			bool node_open_integrator = ImGui::TreeNode("Integrator");
			ImGui::NextColumn();
			ImGui::AlignTextToFramePadding();
			ImGui::Text(m_integrator->getImGuiName().c_str());
			ImGui::NextColumn();
			if (node_open_integrator)
			{
				touched |= m_integrator->getImGuiNodes();
				ImGui::TreePop();
			}
		}

		if (m_sampler)
		{
			bool node_open_sampler = ImGui::TreeNode("Sampler");
			ImGui::NextColumn();
			ImGui::AlignTextToFramePadding();
			ImGui::Text(m_sampler->getImGuiName().c_str());
			ImGui::NextColumn();
			if (node_open_sampler)
			{
				touched |= m_sampler->getImGuiNodes();
				ImGui::TreePop();
			}
		}

		if (m_envmap)
		{
			bool node_open_envmap = ImGui::TreeNode("Environment Map");
			ImGui::NextColumn();
			ImGui::AlignTextToFramePadding();
			ImGui::Text(m_envmap->getImGuiName().c_str());
			ImGui::NextColumn();
			if (node_open_envmap)
			{
				touched |= m_envmap->getImGuiNodes();
				ImGui::TreePop();
			}
		}

		if (m_denoiser)
		{
			bool node_open_denoiser = ImGui::TreeNode("Denoiser");
			ImGui::NextColumn();
			ImGui::AlignTextToFramePadding();
			ImGui::Text(m_denoiser->getImGuiName().c_str());
			ImGui::NextColumn();
			if (node_open_denoiser)
			{
				touched |= m_denoiser->getImGuiNodes();
				ImGui::TreePop();
			}
		}

		bool node_open_shapes = ImGui::TreeNode("Shapes");
		ImGui::NextColumn();
		ImGui::AlignTextToFramePadding();
		ImGui::Text("%d Shapes", (int) m_shapes.size());
		ImGui::NextColumn();
		if (node_open_shapes)
		{
			for (int i = 0; i < m_shapes.size(); i++)
			{
				ImGui::PushID(i);

				// for each shape, add a tree node
				bool node_open_shape = ImGui::TreeNode("Shape", "%s %d", "Shape", i + 1);
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
		ImGui::Text("%d Emitters", (int) m_emitters.size());
		ImGui::NextColumn();
		if (node_open_emitters)
		{
			for (int i = 0; i < m_emitters.size(); i++)
			{
				ImGui::PushID(i);

				// for each shape, add a tree node
				bool node_open_emitter = ImGui::TreeNode("Emitter", "%s %d", "Emitter", i + 1);
				ImGui::NextColumn();
				ImGui::AlignTextToFramePadding();

				ImGui::Text(m_emitters[i]->getImGuiName().c_str());
				ImGui::NextColumn();

				if (node_open_emitter)
				{
					touched |= m_emitters[i]->getImGuiNodes();

					ImGui::TreePop();
				}

				ImGui::PopID();
			}

			ImGui::TreePop();
		}

		ImGui::PopID();

		return touched;
	}
#endif

	NORI_REGISTER_CLASS(Scene, "scene");
NORI_NAMESPACE_END
