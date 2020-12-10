/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob

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

#if !defined(__NORI_INTEGRATOR_H)
#define __NORI_INTEGRATOR_H

#include <nori/object.h>
#ifdef NORI_USE_OPTIX
#include <nori/optix/cuda_shared/LaunchParams.h>
#endif

NORI_NAMESPACE_BEGIN

struct ERenderLayer
{
	using Type = int;
	static constexpr int Composite                    = 0;
	static constexpr int Albedo                       = 1;
	static constexpr int Normal                       = 2;
	static constexpr int Size                         = 3;
	static constexpr char *Strings[ERenderLayer::Size] = {"Composite", "Albedo", "Normal"};
};
using ERenderLayer_t = ERenderLayer::Type;

/**
 * \brief Abstract integrator (i.e. a rendering technique)
 *
 * In Nori, the different rendering techniques are collectively referred to as 
 * integrators, since they perform integration over a high-dimensional
 * space. Each integrator represents a specific approach for solving
 * the light transport equation---usually favored in certain scenarios, but
 * at the same time affected by its own set of intrinsic limitations.
 */
class Integrator : public NoriObject {
public:
    /// Release all memory
    virtual ~Integrator() { }

    /// Perform an (optional) preprocess step
    virtual void preprocess(const Scene *scene) { }

    /**
     * \brief Sample the incident radiance along a ray
     *
     * \param scene
     *    A pointer to the underlying scene
     * \param sampler
     *    A pointer to a sample generator
     * \param ray
     *    The ray in question
     * \return
     *    A (usually) unbiased estimate of the radiance in this direction
     */
    virtual Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray, Color3f &albedo, Color3f &normal) const = 0;

    /**
     * @return A bitmask of the render layers supported/set by the integrator
     */
    virtual int getSupportedLayers() const { return ERenderLayer::Composite; }

    /**
     * \brief Return the type of object (i.e. Mesh/BSDF/etc.) 
     * provided by this instance
     * */
    virtual EClassType getClassType() const override { return EIntegrator; }
#ifdef NORI_USE_IMGUI
	NORI_OBJECT_IMGUI_NAME("Integrator Base");
	virtual bool getImGuiNodes() override { return false; }
#endif

#ifdef NORI_USE_OPTIX
	/**
	 * @return The equivalent optix integrator to use, default is the direct.
	 */
	virtual IntegratorType getOptixIntegratorType() const { return INTEGRATOR_TYPE_DIRECT; }
#endif
};

NORI_NAMESPACE_END

#endif /* __NORI_INTEGRATOR_H */
