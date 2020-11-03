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

#include <nori/integrator.h>
#include <nori/sampler.h>
#include <nori/emitter.h>
#include <nori/bsdf.h>
#include <nori/scene.h>
#include <nori/photon.h>

NORI_NAMESPACE_BEGIN

class PhotonMapper : public Integrator
{
public:
    /// Photon map data structure
    typedef PointKDTree<Photon> PhotonMap;

    PhotonMapper(const PropertyList &props)
    {
        /* Lookup parameters */
        m_photonCount = props.getInteger("photonCount", 1000000);
        m_photonRadius = props.getFloat("photonRadius", 0.0f /* Default: automatic */);
    }

    virtual void preprocess(const Scene *scene) override
    {
        cout << "Gathering " << m_photonCount << " photons .. ";
        cout.flush();

        /* Create a sample generator for the preprocess step */
        Sampler *sampler = static_cast<Sampler *>(
            NoriObjectFactory::createInstance("independent", PropertyList()));

        /* Allocate memory for the photon map */
        m_photonMap = std::unique_ptr<PhotonMap>(new PhotonMap());
        m_photonMap->reserve(m_photonCount);

        /* Estimate a default photon radius */
        if (m_photonRadius == 0)
            m_photonRadius = scene->getBoundingBox().getExtents().norm() / 500.0f;

        /* How to add a photon?
		 * m_photonMap->push_back(Photon(
		 *	Point3f(0, 0, 0),  // Position
		 *	Vector3f(0, 0, 1), // Direction
		 *	Color3f(1, 2, 3)   // Power
		 * ));
		 */
        m_emittedPhotons = 0;
        while (m_photonMap->size() < m_photonCount)
        {
            Color3f t(1.f); // throughput value
            const Emitter *emitter = scene->getRandomEmitter(sampler->next1D());
            Ray3f sampleRay;

            Color3f W = emitter->samplePhoton(sampleRay, sampler->next2D(), sampler->next2D()) * scene->getLights().size();

            m_emittedPhotons++;
            while (true)
            {
                Intersection its;
                if (!scene->rayIntersect(sampleRay, its))
                {
                    break;
                }

                const Shape *shape = its.mesh;
                const BSDF *bsdf = shape->getBSDF();

                // check if diffuse
                if (bsdf->isDiffuse())
                {
                    // --> add photon to map

                    // check if still place for a new photon
                    if (m_photonMap->size() >= m_photonCount)
                    {
                        break;
                    }
                    else
                    {
                        m_photonMap->push_back(Photon(its.p, -sampleRay.d, t * W));
                    }
                }

                // russian roulette
                auto successprob = t.maxCoeff();
                if ((sampler->next1D()) > successprob)
                    break;
                else
                    t /= successprob;

                // calculate next ray by sampling BSDF
                BSDFQueryRecord bqr(its.toLocal(-sampleRay.d));
                Color3f bsdf_col = bsdf->sample(bqr, sampler->next2D());

                // break if sampling failed
                if (bsdf_col.isZero(Epsilon))
                {
                    break;
                }

                sampleRay = Ray3f(its.p, its.toWorld(bqr.wo));
                t = t * bsdf_col;
            }
        }

        /* Build the photon map */
        m_photonMap->build();
    }

    virtual Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &_ray) const override
    {
        /* How to find photons?
		 * std::vector<uint32_t> results;
		 * m_photonMap->search(Point3f(0, 0, 0), // lookup position
		 *                     m_photonRadius,   // search radius
		 *                     results);
		 *
		 * for (uint32_t i : results) {
		 *    const Photon &photon = (*m_photonMap)[i];
		 *    cout << "Found photon!" << endl;
		 *    cout << " Position  : " << photon.getPosition().toString() << endl;
		 *    cout << " Power     : " << photon.getPower().toString() << endl;
		 *    cout << " Direction : " << photon.getDirection().toString() << endl;
		 * }
		 */

        Color3f Li = Color3f(0.f); // initial radiance§
        Color3f t = Color3f(1.0);  // initial throughput
        Ray3f traceRay(_ray);
        while (true)
        {
            Intersection its;
            if (!scene->rayIntersect(traceRay, its))
            {
                break;
            }

            // get colliding object and shape
            const Shape *shape = its.mesh;
            const BSDF *bsdf = shape->getBSDF();

            // if shape is emitter, add eval to result
            if (shape->isEmitter())
            {
                auto emitter = shape->getEmitter();
                EmitterQueryRecord eqr(traceRay.o, its.p, its.shFrame.n);
                Li += t * emitter->eval(eqr);
            }

            if (bsdf->isDiffuse())
            {
                // estimate photon density at the intersection point
                // and terminate the recursion!

                // query all nearby photons from the photon map,
                // summing the product of eachs power with the bsdf value
                // given the photons direction
                std::vector<uint32_t> results;
                m_photonMap->search(its.p,          // lookup position
                                    m_photonRadius, // search radius
                                    results);

                Color3f photonColor(0.f);
                for (uint32_t i : results)
                {
                    const Photon &photon = (*m_photonMap)[i];

                    // create a bsdf query record given the direction of the photon
                    BSDFQueryRecord bqr_p(its.toLocal(-traceRay.d), its.toLocal(photon.getDirection()), EMeasure::ESolidAngle);
                    photonColor += photon.getPower() * bsdf->eval(bqr_p);
                }

                // dividing the product by the area queried (pi * r * r) and by the emitted photon number
                photonColor /= M_PI * m_photonRadius * m_photonRadius * m_emittedPhotons;

                Li += t * photonColor;
                // and terminate the recursion!
                break;
            }

            float succ_prob = std::min(t.maxCoeff(), 0.99f);
            if (sampler->next1D() > succ_prob)
            {
                break;
            }
            else
            {
                t = t / succ_prob;
            }

            BSDFQueryRecord bRec(its.toLocal(-traceRay.d));

            // Sample BSDF
            Color3f bsdf_col = bsdf->sample(bRec, sampler->next2D());
            if (bsdf_col.isZero(Epsilon))
            {
                break; // BSDF sampling failed
            }

            t = t * bsdf_col;

            // create next ray to trace
            traceRay = Ray3f(its.p, its.toWorld(bRec.wo));
        }

        return Li;
    }

    virtual std::string toString() const override
    {
        return tfm::format(
            "PhotonMapper[\n"
            "  photonCount = %i,\n"
            "  photonRadius = %f\n"
            "]",
            m_photonCount,
            m_photonRadius);
    }

private:
    /* 
     * Important: m_photonCount is the total number of photons deposited in the photon map,
     * NOT the number of emitted photons. You will need to keep track of those yourself.
     */
    int m_photonCount;
    float m_photonRadius;
    std::unique_ptr<PhotonMap> m_photonMap;

    int m_emittedPhotons = 0;
};

NORI_REGISTER_CLASS(PhotonMapper, "photonmapper");
NORI_NAMESPACE_END
