/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Romain Prévost

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

#if !defined(__NORI_SHAPE_H)
#define __NORI_SHAPE_H

#include <nori/object.h>
#include <nori/frame.h>
#include <nori/bbox.h>
#include <nori/texture.h>
#include <nori/medium.h>

#ifdef NORI_USE_OPTIX
struct OptixBuildInput;
#include <nori/optix/OptixSbtTypes.h>
#include <nori/optix/OptixState.h>
#endif

NORI_NAMESPACE_BEGIN

/**
 * \brief Intersection data structure
 *
 * This data structure records local information about a ray-triangle intersection.
 * This includes the position, traveled ray distance, uv coordinates, as well
 * as well as two local coordinate frames (one that corresponds to the true
 * geometry, and one that is used for shading computations).
 */
struct Intersection
{
    /// Position of the surface intersection
    Point3f p;
    /// Unoccluded distance along the ray
    float t;
    /// UV coordinates, if any
    Point2f uv;
    /// Shading frame (based on the shading normal)
    Frame shFrame;
    /// Geometric frame (based on the true geometry)
    Frame geoFrame;
    /// Pointer to the associated shape
    const Shape *shape;

    /// Create an uninitialized intersection record
    Intersection() : shape(nullptr) {}

    /// Transform a direction vector into the local shading frame
    Vector3f toLocal(const Vector3f &d) const
    {
        return shFrame.toLocal(d);
    }

    /// Transform a direction vector from local to world coordinates
    Vector3f toWorld(const Vector3f &d) const
    {
        return shFrame.toWorld(d);
    }

    /// Return a human-readable summary of the intersection record
    std::string toString() const;
};

/**
 * \brief Data record for conveniently querying and sampling the
 * a point on a shape
 */
struct ShapeQueryRecord
{
    /// Reference point
    Point3f ref;
    /// Sampled point
    Point3f p;
    /// Sampled normal
    Normal3f n;
    /// Probability of the sample
    float pdf;

    /// Empty constructor
    ShapeQueryRecord() {}
    /// Data structure with ref to call sampleSurface()
    ShapeQueryRecord(const Point3f &ref_) : ref(ref_) {}
    /// Data structure with ref and p to call pdfSurface()
    ShapeQueryRecord(const Point3f &ref_, const Point3f &p_) : ref(ref_), p(p_) {}
};

/**
 * \brief Superclass of all shapes
 */
class Shape : public NoriObject
{
public:
    /// Release all memory
    virtual ~Shape();

	NoriObject *cloneAndInit() override = 0;

	/**
	 * Finish initialization for other components
	 * @param clone The already created clone
	 */
	void cloneAndInit(Shape *clone);

    /// Initialize internal data structures (called once by the XML parser)
    virtual void update(const NoriObject *guiObject) override;

	virtual void addChild(NoriObject *child) override;

	//// Return an axis-aligned bounding box of the entire mesh
    const BoundingBox3f &getBoundingBox() const { return m_bbox; }

    /// Does the shape have a surface or volume emitter
    bool hasAnyEmitter() const { return m_emitter || getMediumEmitter(); }

	/// Return a pointer to an attached area emitter instance
    Emitter *getEmitter() { return m_emitter; }

    /// Return a pointer to an attached area emitter instance (const version)
    const Emitter *getEmitter() const { return m_emitter; }

    /// Get the surface or media emitter
    Emitter *getEmitter(bool surface) { return surface ? m_emitter : getMediumEmitter(); }
    const Emitter *getEmitter(bool surface) const { return surface ? m_emitter : getMediumEmitter(); }

	Emitter* getMediumEmitter() { return m_medium ? m_medium->getEmitter() : nullptr; }
	const Emitter* getMediumEmitter() const { return m_medium ? m_medium->getEmitter() : nullptr; }

	/// Return a pointer to the BSDF associated with this mesh
    BSDF *getBSDF() { return m_bsdf; }
    const BSDF *getBSDF() const { return m_bsdf; }

	Medium* getMedium() { return m_medium; }
	const Medium* getMedium() const { return m_medium; }

    /// Return the total number of primitives in this shape
    virtual uint32_t getPrimitiveCount() const { return 1; }

    //// Return an axis-aligned bounding box containing the given triangle
    virtual BoundingBox3f getBoundingBox(uint32_t index) const = 0;

    //// Return the centroid of the given triangle
    virtual Point3f getCentroid(uint32_t index) const = 0;

    //// Ray-Shape intersection test
    virtual bool rayIntersect(uint32_t index, const Ray3f &ray, float &u, float &v, float &t) const = 0;

    /// Set the intersection information: hit point, shading frame, UVs, etc.
    virtual void setHitInformation(uint32_t index, const Ray3f &ray, Intersection &its) const = 0;

	/**
	 * \brief Sample a point on the surface (potentially using the point sRec.ref to importance sample)
	 * This method should set sRec.p, sRec.n and sRec.pdf
	 * Probability should be with respect to area
	 * */
	virtual void sampleSurface(ShapeQueryRecord &sRec, const Point2f &sample) const = 0;
	/**
	 * \brief Return the probability of sampling a point sRec.p by the sampleSurface() method (sRec.ref should be set before)
	 * sRec.n and sRec.pdf are ignored
	 * */
	virtual float pdfSurface(const ShapeQueryRecord &sRec) const = 0;

	/**
	 * \brief Sample a point inside the volume (potentially using the point sRec.ref to importance sample)
	 * This method should set sRec.p and sRec.pdf. sRec.n is unset
	 * Probability should be with respect to volume. Uses the bbox if not overridden.
	 * */
	virtual void sampleVolume(ShapeQueryRecord &sRec, const Point3f &sample) const;
	/**
	 * \brief Return the probability of sampling a point inside the shape sRec.p by the sampleSurface() method (sRec.ref should be set before)
	 * sRec.n and sRec.pdf are ignored.
	 * Note: Does not check that the point is inside the shape, for efficiency reasons. Uses the bbox if not overridden.
	 * */
	virtual float pdfVolume(const ShapeQueryRecord &sRec) const;

#ifdef NORI_USE_OPTIX
	/**
	 * Defined in OptixState.as.cpp
	 * @return OptixBuildInput required to build the geometry accel structure
	 */
	virtual OptixBuildInput getOptixBuildInput();

	/**
	 * Fill the hitgroup record for this shape
	 * @param hitgroupRecords Append records to this vector, one record per ray type
	 */
	virtual void getOptixHitgroupRecords(OptixState &state, std::vector<HitGroupRecord> &hitgroupRecords) = 0;
	void getOptixHitgroupRecordsShape(HitGroupRecord& rec);
#endif

    /**
     * \brief Return the type of object (i.e. Mesh/BSDF/etc.)
     * provided by this instance
     * */
    virtual EClassType getClassType() const override { return EShape; }

#ifdef NORI_USE_IMGUI
	NORI_OBJECT_IMGUI_NAME("Shape");
	virtual bool getImGuiNodes() override;
#endif

		/**
	 * Has the shape only been moved/transformed. Only IAS needs to be reconstructed
	 */
	mutable bool transformTouched = true;

	/**
	 * Has the shape geometry been modified that the BVH, specifically GAS, needs to be reconstructed?
	 */
	mutable bool geometryTouched = true;
    Emitter *m_emitter = nullptr; ///< Associated emitter, if any
protected:
	Texture<Normal3f> *m_normalMap = nullptr;
	BSDF *m_bsdf = nullptr;       ///< BSDF of the surface
    BoundingBox3f m_bbox;         ///< Bounding box of the mesh
	Medium * m_medium = nullptr;  ///< Associated medium inside the shape, vacuum if null

	float m_volume = 0.f;
	virtual void updateVolume();  ///< Recalculate the volume, uses the bbox if not overridden
};

NORI_NAMESPACE_END

#endif /* __NORI_SHAPE_H */
