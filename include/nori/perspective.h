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
#include <Eigen/src/Core/Matrix.h>

NORI_NAMESPACE_BEGIN

/**
 * \brief Perspective camera with depth of field
 *
 * This class implements a simple perspective camera model. It uses an
 * infinitesimally small aperture, creating an infinite depth of field.
 */
	class PerspectiveCamera : public Camera
	{
	public:
		explicit PerspectiveCamera(const PropertyList &propList);

		NoriObject *cloneAndInit() override;

		void update(const NoriObject *guiObject) override;

		Color3f sampleRay(Ray3f &ray,
		                  const Point2f &samplePosition,
		                  const Point2f &apertureSample) const;

		virtual void addChild(NoriObject *obj) override;

		/// Return a human-readable summary
		virtual std::string toString() const override;
#ifdef NORI_USE_IMGUI
		NORI_OBJECT_IMGUI_NAME("Perspective");
		virtual bool getImGuiNodes() override;
#endif

		Eigen::Matrix4f getTransform() const override { return m_cameraToWorld.getMatrix(); }
		float getFov() const { return m_fov; }
		float getFocalDistance() const { return m_focalDistance; }
		float getLensRadius() const { return m_lensRadius; }

#ifdef NORI_USE_OPTIX
		void getOptixData(RaygenData &data) const override;
#endif
	private:
		// -- Properties
		float m_fov;
		float m_nearClip;
		float m_farClip;

		// Depth of Field
		float m_focalDistance;  // focal length of the lens, derived from focus distance and fov
		float m_fstop;          // fstop = focalDistance / 2 lensRadius

		// -- Sub-objects
		Transform m_cameraToWorld;

		// -- Derived properties
		Transform m_sampleToCamera;
		Vector2f  m_invOutputSize;

		float m_lensRadius;     // aka aperture, derived from fstop
	};
NORI_NAMESPACE_END

