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

#if !defined(__NORI_SCENE_H)
#define __NORI_SCENE_H

#include <nori/bvh.h>
#include <nori/emitter.h>
#include <nori/denoiser.h>
#include <nori/volume.h>
#ifdef NORI_USE_OPTIX
#include <nori/optix/OptixRenderer.h>
#endif

NORI_NAMESPACE_BEGIN

	/**
	 * \brief Main scene data structure
	 *
	 * This class holds information on scene objects and is responsible for
	 * coordinating rendering jobs. It also provides useful query routines that
	 * are mostly used by the \ref Integrator implementations.
	 */
	class Scene : public NoriObject
	{

	public:
		/// Construct a new scene object
		Scene(const PropertyList &);

		/// Release all memory
		virtual ~Scene();

		/// Return a pointer to the scene's kd-tree
		const BVH *getBVH() const { return m_bvh; }

		/// Return a pointer to the scene's integrator
		const Integrator *getIntegrator(bool usePreview) const
		{
			return usePreview ? m_previewIntegrator : m_integrator;
		}

		Integrator *getIntegrator(bool usePreview)
		{
			return usePreview ? m_previewIntegrator : m_integrator;
		}

		/// Return a pointer to the scene's camera
		const Camera *getCamera() const { return m_camera; }

		/// Return a pointer to the scene's sample generator (const version)
		const Sampler *getSampler() const { return m_sampler; }

		/// Return a pointer to the scene's sample generator
		Sampler *getSampler() { return m_sampler; }

		/// Return a reference to an array containing all shapes
		const std::vector<Shape *> &getShapes() const { return m_shapes; }

		/// Return a reference to an array containing all lights
		const std::vector<Emitter *> &getLights() const { return m_emitters; }

		/// Return a random emitter
		const Emitter *getRandomEmitter(float rnd) const
		{
			auto const &n    = m_emitters.size();
			size_t     index = std::min(static_cast<size_t>(std::floor(n * rnd)), n - 1);
			if (index >= m_emitters.size())
				return nullptr;
			return m_emitters[index];
		}
#ifdef NORI_USE_VOLUMES
		const std::vector<Volume *> &getVolumes() const
		{
			return m_volumes;
		}
		const Volume *getRandomVolume(float rnd) const
		{
			auto const &n    = m_volumes.size();
			size_t     index = std::min(
					static_cast<size_t>(std::floor(n * rnd)),
					n - 1);
			if (index >= m_volumes.size())
				return nullptr;
			return m_volumes[index];
		}
#endif

		Emitter *getEnvMap() const { return m_envmap; }

		Denoiser *getDenoiser() const { return m_denoiser; }

		Medium *getAmbientMedium() const { return m_ambientMedium; }

		/**
		 * \brief Intersect a ray against all triangles stored in the scene
		 * and return detailed intersection information
		 *
		 * \param ray
		 *    A 3-dimensional ray data structure with minimum/maximum
		 *    extent information
		 *
		 * \param its
		 *    A detailed intersection record, which will be filled by the
		 *    intersection query
		 *
		 * \return \c true if an intersection was found
		 */
		bool rayIntersect(const Ray3f &ray, Intersection &its) const
		{
			return m_bvh->rayIntersect(ray, its, false);
		}

		/**
		 * \brief Intersect a ray against all triangles stored in the scene
		 * and \a only determine whether or not there is an intersection.
		 *
		 * This method much faster than the other ray tracing function,
		 * but the performance comes at the cost of not providing any
		 * additional information about the detected intersection
		 * (not even its position).
		 *
		 * \param ray
		 *    A 3-dimensional ray data structure with minimum/maximum
		 *    extent information
		 *
		 * \return \c true if an intersection was found
		 */
		bool rayIntersect(const Ray3f &ray) const
		{
			Intersection its; /* Unused */
			return m_bvh->rayIntersect(ray, its, true);
		}

		/**
		 * \brief Return an axis-aligned box that bounds the scene
		 */
		const BoundingBox3f &getBoundingBox() const
		{
			return m_bvh->getBoundingBox();
		}

		virtual NoriObject *cloneAndInit() override;

		/**
		 * \brief Inherited from \ref NoriObject::update()
		 *
		 * Initializes the internal data structures (kd-tree,
		 * emitter sampling data structures, etc.)
		 */
		virtual void update(const NoriObject *other) override;

		/// Add a child object to the scene (meshes, integrators etc.)
		virtual void addChild(NoriObject *obj) override;

		/// Return a string summary of the scene (for debugging purposes)
		virtual std::string toString() const override;

		virtual EClassType getClassType() const override { return EScene; }
#ifdef NORI_USE_IMGUI
		NORI_OBJECT_IMGUI_NAME("Scene");
		virtual bool getImGuiNodes() override;
#endif

#ifdef NORI_USE_OPTIX
		OptixState *getOptixState() const { return m_optixState; }
		OptixRenderer *m_optixRenderer;
		OptixState *m_optixState = nullptr;
#endif
	private:
		std::vector<Shape *> m_shapes;
		Integrator           *m_integrator        = nullptr;
		Integrator           *m_previewIntegrator = nullptr;

		Sampler *m_sampler = nullptr;
		Camera  *m_camera  = nullptr;
		BVH     *m_bvh     = nullptr;

		Emitter  *m_envmap   = nullptr;
		Denoiser *m_denoiser = nullptr;
		Medium * m_ambientMedium = nullptr;

		std::vector<Emitter *> m_emitters;

#ifdef NORI_USE_VOLUMES
		std::vector<Volume *> m_volumes;
#endif

		/**
		 * Has the shape only been moved/transformed. Only IAS needs to be reconstructed
		 */
		mutable bool transformTouched = true;

		/**
		 * Has the shape geometry been modified that the BVH, specifically GAS, needs to be reconstructed?
		 */
		mutable bool geometryTouched = true;
	};

NORI_NAMESPACE_END

#endif /* __NORI_SCENE_H */
