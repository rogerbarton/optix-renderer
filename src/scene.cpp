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
    m_bvh = new BVH();
}

Scene::~Scene()
{
    delete m_bvh;
    delete m_sampler;
    delete m_camera;
    delete m_integrator;
    for (auto e : m_emitters)
        delete e;
    m_emitters.clear();

    if (m_envmap)
        delete m_envmap;
    if (m_denoiser)
        delete m_denoiser;
}

void Scene::activate()
{
    m_bvh->build();

    if (!m_integrator)
        throw NoriException("No integrator was specified!");
    if (!m_camera)
        throw NoriException("No camera was specified!");

    if (!m_sampler)
    {
        /* Create a default (independent) sampler */
        m_sampler = static_cast<Sampler *>(
            NoriObjectFactory::createInstance("independent", PropertyList()));
        m_sampler->activate();
    }

    cout << endl;
    cout << "Configuration: " << toString() << endl;
    cout << endl;
}

void Scene::addChild(NoriObject *obj)
{
    switch (obj->getClassType())
    {
    case EMesh:
    {
        Shape *mesh = static_cast<Shape *>(obj);
        m_bvh->addShape(mesh);
        m_shapes.push_back(mesh);
        if (mesh->isEmitter())
            m_emitters.push_back(mesh->getEmitter());
    }
    break;

    case EEmitter:
        m_emitters.push_back(static_cast<Emitter *>(obj));
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

    case EEnvironmentMap:
        if (m_envmap)
            throw NoriException("There can only be one environment map per scene!");
        m_envmap = static_cast<EnvironmentMap *>(obj);
        break;
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
void Scene::getImGuiNodes()
{
    ImGui::PushID(EScene);
    if (m_camera)
    {
        bool node_open_camera = ImGui::TreeNode("Camera");
        ImGui::NextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::Text(m_camera->getImGuiName());
        ImGui::NextColumn();
        if (node_open_camera)
        {
            m_camera->getImGuiNodes();
            ImGui::TreePop();
        }
    }

    if (m_integrator)
    {
        bool node_open_integrator = ImGui::TreeNode("Integrator");
        ImGui::NextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::Text(m_integrator->getImGuiName());
        ImGui::NextColumn();
        if (node_open_integrator)
        {
            m_integrator->getImGuiNodes();
            ImGui::TreePop();
        }
    }

    if (m_sampler)
    {
        bool node_open_sampler = ImGui::TreeNode("Sampler");
        ImGui::NextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::Text(m_sampler->getImGuiName());
        ImGui::NextColumn();
        if (node_open_sampler)
        {
            m_sampler->getImGuiNodes();
            ImGui::TreePop();
        }
    }

    if (m_envmap)
    {
        bool node_open_envmap = ImGui::TreeNode("Environment Map");
        ImGui::NextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::Text(m_envmap->getImGuiName());
        ImGui::NextColumn();
        if (node_open_envmap)
        {
            m_envmap->getImGuiNodes();
            ImGui::TreePop();
        }
    }

    if (m_denoiser)
    {
        bool node_open_denoiser = ImGui::TreeNode("Denoiser");
        ImGui::NextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::Text(m_denoiser->getImGuiName());
        ImGui::NextColumn();
        if (node_open_denoiser)
        {
            m_denoiser->getImGuiNodes();
            ImGui::TreePop();
        }
    }

    bool node_open_shapes = ImGui::TreeNode("Shapes");
    ImGui::NextColumn();
    ImGui::AlignTextToFramePadding();
    ImGui::Text("%d Shapes", (int)m_shapes.size());
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

            ImGui::Text(m_shapes[i]->getImGuiName());
            ImGui::NextColumn();

            if (node_open_shape)
            {
                m_shapes[i]->getImGuiNodes();

                ImGui::TreePop();
            }

            ImGui::PopID();
        }

        ImGui::TreePop();
    }

    bool node_open_emitter = ImGui::TreeNode("Emitter");
    ImGui::NextColumn();
    ImGui::AlignTextToFramePadding();
    ImGui::Text("%d Emitters", (int)m_emitters.size());
    ImGui::NextColumn();
    if (node_open_emitter)
    {
        for (int i = 0; i < m_emitters.size(); i++)
        {
            ImGui::PushID(i);

            // for each shape, add a tree node
            bool node_open_emitter = ImGui::TreeNode("Emitter", "%s %d", "Emitter", i + 1);
            ImGui::NextColumn();
            ImGui::AlignTextToFramePadding();

            ImGui::Text(m_emitters[i]->getImGuiName());
            ImGui::NextColumn();

            if (node_open_emitter)
            {
                m_emitters[i]->getImGuiNodes();

                ImGui::TreePop();
            }

            ImGui::PopID();
        }

        ImGui::TreePop();
    }

    ImGui::PopID();
}
#endif

NORI_REGISTER_CLASS(Scene, "scene");
NORI_NAMESPACE_END
