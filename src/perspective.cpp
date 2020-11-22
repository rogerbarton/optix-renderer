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

#include <nori/camera.h>
#include <nori/rfilter.h>
#include <nori/warp.h>
#include <Eigen/Geometry>

NORI_NAMESPACE_BEGIN

/**
 * \brief Perspective camera with depth of field
 *
 * This class implements a simple perspective camera model. It uses an
 * infinitesimally small aperture, creating an infinite depth of field.
 */
class PerspectiveCamera : public Camera {
public:
    PerspectiveCamera(const PropertyList &propList) {
        /* Width and height in pixels. Default: 720p */
        m_outputSize.x() = propList.getInteger("width", 1280);
        m_outputSize.y() = propList.getInteger("height", 720);
        m_invOutputSize = m_outputSize.cast<float>().cwiseInverse();

        /* Specifies an optional camera-to-world transformation. Default: none */
        m_cameraToWorld = propList.getTransform("toWorld", Transform());

        /* Horizontal field of view in degrees */
        m_fov = propList.getFloat("fov", 30.0f);

        /* Near and far clipping planes in world-space units */
        m_nearClip = propList.getFloat("nearClip", 1e-4f);
        m_farClip = propList.getFloat("farClip", 1e4f);

        // Depth of Field
        m_lensRadius    = propList.getFloat("lensRadius", 0.f);
	    m_focalDistance = propList.getFloat("focalDistance", 1.f);

        m_rfilter = NULL;
    }

    virtual void update(const NoriObject *guiObject) override {
        float aspect = m_outputSize.x() / (float) m_outputSize.y();

        /* Project vectors in camera space onto a plane at z=1:
         *
         *  xProj = cot * x / z
         *  yProj = cot * y / z
         *  zProj = (far * (z - near)) / (z * (far-near))
         *  The cotangent factor ensures that the field of view is 
         *  mapped to the interval [-1, 1].
         */
        float recip = 1.0f / (m_farClip - m_nearClip),
              cot = 1.0f / std::tan(degToRad(m_fov / 2.0f));

        Eigen::Matrix4f perspective;
        perspective <<
            cot, 0,   0,   0,
            0, cot,   0,   0,
            0,   0,   m_farClip * recip, -m_nearClip * m_farClip * recip,
            0,   0,   1,   0;

        /**
         * Translation and scaling to shift the clip coordinates into the
         * range from zero to one. Also takes the aspect ratio into account.
         */
        m_sampleToCamera = Transform( 
            Eigen::DiagonalMatrix<float, 3>(Vector3f(0.5f, -0.5f * aspect, 1.0f)) *
            Eigen::Translation<float, 3>(1.0f, -1.0f/aspect, 0.0f) * perspective).inverse();

        /* If no reconstruction filter was assigned, instantiate a Gaussian filter */
        if (!m_rfilter) {
            m_rfilter = static_cast<ReconstructionFilter *>(
                    NoriObjectFactory::createInstance("gaussian", PropertyList()));
	        m_rfilter->update(guiObject);
        }
    }

    Color3f sampleRay(Ray3f &ray,
            const Point2f &samplePosition,
            const Point2f &apertureSample) const {
        /* Compute the corresponding position on the 
           near plane (in local camera space) */
        const Point3f nearP = m_sampleToCamera * Point3f(
            samplePosition.x() * m_invOutputSize.x(),
            samplePosition.y() * m_invOutputSize.y(), 0.0f);

        /* Turn into a normalized ray direction, and
           adjust the ray interval accordingly */
        const Vector3f d = nearP.normalized();

        // Create local space ray
        ray.d = d;
	    ray.o = Point3f(0, 0, 0);

	    // Depth of field, adjusts ray in local space
        if(m_lensRadius > Epsilon)
        {
	        ray.update();

	        static Sampler *const sampler = dynamic_cast<Sampler *>(
			        NoriObjectFactory::createInstance("independent", PropertyList()));

	        const Point2f pLens = m_lensRadius * Warp::squareToUniformDisk(sampler->next2D());
	        const float ft = m_focalDistance / ray.d.z();
	        // position of ray at time of intersection with the focal plane
	        const Point3f pFocus = ray(ft);

	        ray.o = Point3f(pLens.x(), pLens.y(), 0.f);
	        // direction connecting aperture and focal plane points
	        ray.d = (pFocus - ray.o).normalized();
        }

        ray.o = m_cameraToWorld * ray.o;
        ray.d = m_cameraToWorld * ray.d;

	    const float invZ = 1.0f / d.z();
        ray.mint = m_nearClip * invZ;
        ray.maxt = m_farClip * invZ;
        ray.update();

        return Color3f(1.0f);
    }

    virtual void addChild(NoriObject *obj) override {
        switch (obj->getClassType()) {
            case EReconstructionFilter:
                if (m_rfilter)
                    throw NoriException("Camera: tried to register multiple reconstruction filters!");
                m_rfilter = static_cast<ReconstructionFilter *>(obj);
                break;

            default:
                throw NoriException("Camera::addChild(<%s>) is not supported!",
                    classTypeName(obj->getClassType()));
        }
    }

    /// Return a human-readable summary
    virtual std::string toString() const override {
        return tfm::format(
            "PerspectiveCamera[\n"
            "  cameraToWorld = %s,\n"
            "  outputSize = %s,\n"
            "  fov = %f,\n"
            "  clip = [%f, %f],\n"
            "  rfilter = %s\n"
            "]",
            indent(m_cameraToWorld.toString(), 18),
            m_outputSize.toString(),
            m_fov,
            m_nearClip,
            m_farClip,
            indent(m_rfilter->toString())
        );
    }
#ifndef NORI_USE_NANOGUI
    virtual const char* getImGuiName() const override { return "Perspective"; }
    virtual bool getImGuiNodes() override {
        bool ret = Camera::getImGuiNodes();
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen |
                                   ImGuiTreeNodeFlags_Bullet;

        ImGui::AlignTextToFramePadding();
        ImGui::PushID(0);
        bool node_open = ImGui::TreeNode("Transform");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::Text("To World");
        ImGui::NextColumn();
        if(node_open) {
            ret |= m_cameraToWorld.getImGuiNodes();
            ImGui::TreePop();
        }
        ImGui::PopID();

        ImGui::AlignTextToFramePadding();
        ImGui::PushID(1);
        ImGui::TreeNodeEx("fov", flags, "Field Of View");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        ret |= ImGui::DragFloat("##value", &m_fov, 1, 0, 360, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::NextColumn();
        ImGui::PopID();

        ImGui::AlignTextToFramePadding();
        ImGui::PushID(2);
        ImGui::TreeNodeEx("nearCLip", flags, "Near Clip");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        ret |= ImGui::DragFloat("##value", &m_nearClip, 1, 0, SLIDER_MAX_FLOAT, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::NextColumn();
        ImGui::PopID();

        ImGui::AlignTextToFramePadding();
        ImGui::PushID(3);
        ImGui::TreeNodeEx("fov", flags, "Far Clip");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        ret |= ImGui::DragFloat("##value", &m_farClip, 1, 0, SLIDER_MAX_FLOAT, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::NextColumn();
        ImGui::PopID();

        ImGui::AlignTextToFramePadding();
        ImGui::PushID(4);
        ImGui::TreeNodeEx("lensRadius", flags, "Lens Radius");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        ret |= ImGui::DragFloat("##value", &m_lensRadius, 0.01f, 0, SLIDER_MAX_FLOAT, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::NextColumn();
        ImGui::PopID();

        ImGui::AlignTextToFramePadding();
        ImGui::PushID(5);
        ImGui::TreeNodeEx("focalDistance", flags, "Focal Distance");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        ret |= ImGui::DragFloat("##value", &m_focalDistance, 0.1f, 0, SLIDER_MAX_FLOAT, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::NextColumn();
        ImGui::PopID();

        return ret;
    }
#endif
private:
    Vector2f m_invOutputSize;
    Transform m_sampleToCamera;
    Transform m_cameraToWorld;
    float m_fov;
    float m_nearClip;
    float m_farClip;

    // Depth of Field
    float m_lensRadius; // aka aperture
    float m_focalDistance; // aka focal length
};

NORI_REGISTER_CLASS(PerspectiveCamera, "perspective");
NORI_NAMESPACE_END
