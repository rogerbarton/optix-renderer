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

#include <nori/scene.h>
#include <nori/bsdf.h>
#include <nori/camera.h>
#include <nori/integrator.h>
#include <nori/sampler.h>
#include <hypothesis.h>
#include <pcg32.h>

/*
 * =======================================================================
 *   WARNING    WARNING    WARNING    WARNING    WARNING    WARNING
 * =======================================================================
 *   Remember to put on SAFETY GOGGLES before looking at this file. You
 *   are most certainly not expected to read or understand any of it.
 * =======================================================================
 */

NORI_NAMESPACE_BEGIN

/**
 * Student's t-test for the equality of means
 *
 * This test analyzes whether the expected value of a random variable matches a
 * certain known value. When there is significant statistical "evidence"
 * against this hypothesis, the test fails.
 *
 * This is useful in checking whether a Monte Carlo method method converges
 * against the right value. Because statistical tests are able to handle the
 * inherent noise of these methods, they can be used to construct statistical
 * test suites not unlike the traditional unit tests used in software engineering.
 *
 * This implementation can be used to test two things:
 *
 * 1. that the illumination scattered by a BRDF model under uniform illumination
 *    into a certain direction matches a given value (modulo noise).
 *
 * 2. that the average radiance received by a camera within some scene
 *    matches a given value (modulo noise).
 *
 * TODO: No idea if scenes as children will work!!
 */
class StudentsTTest : public NoriObject {
public:
    explicit StudentsTTest(const PropertyList &propList) {
        /* The null hypothesis will be rejected when the associated
           p-value is below the significance level specified here. */
        m_significanceLevel = propList.getFloat("significanceLevel", 0.01f);

        /* This parameter specifies a list of incidence angles that will be tested */
        std::vector<std::string> angles = tokenize(propList.getString("angles", ""));
        for (auto angle : angles)
            m_angles.push_back(toFloat(angle));

        /* This parameter specifies a list of reference values, one for each angle */
        std::vector<std::string> references = tokenize(propList.getString("references", ""));
        for (auto angle : references)
            m_references.push_back(toFloat(angle));

        /* Number of BSDF samples that should be generated (default: 100K) */
        m_sampleCount = propList.getInteger("sampleCount", 100000);
    }

	NoriObject *cloneAndInit() override
	{
		auto clone = new StudentsTTest(*this);

		assert(clone->m_bsdfs.size() == m_bsdfs.size());
		for (int i = 0; i < m_bsdfs.size(); ++i)
			clone->m_bsdfs[i] = dynamic_cast<BSDF *>(m_bsdfs[i]->cloneAndInit());

		assert(clone->m_scenes.size() == m_scenes.size());
		for (int i = 0; i < m_scenes.size(); ++i)
			clone->m_scenes[i] = dynamic_cast<Scene *>(m_scenes[i]->cloneAndInit());

		clone->execute();
		return clone;
	}

	void update(const NoriObject *guiObject) override
	{
		const auto *gui = dynamic_cast<const StudentsTTest *>(guiObject);
		if (!gui->touched)return;
		gui->touched = false;


		// -- Copy properties
		m_angles            = gui->m_angles;
		m_references        = gui->m_references;
		m_significanceLevel = gui->m_significanceLevel;
		m_sampleCount       = gui->m_sampleCount;

		// -- Update sub-objects
		assert(m_bsdfs.size() == gui->m_bsdfs.size());
		for (int i = 0; i < gui->m_bsdfs.size(); i++)
			m_bsdfs[i]->update(gui->m_bsdfs[i]);

		// TODO: No idea if this will work
		assert(m_scenes.size() == gui->m_scenes.size());
		for (int i = 0; i < gui->m_scenes.size(); i++)
			m_scenes[i]->update(gui->m_scenes[i]);

		execute();
	}

	virtual ~StudentsTTest() {
        for (auto bsdf : m_bsdfs)
            delete bsdf;
        for (auto scene : m_scenes)
            delete scene;
    }

    virtual void addChild(NoriObject *obj) override {
        switch (obj->getClassType()) {
            case EBSDF:
                m_bsdfs.push_back(static_cast<BSDF *>(obj));
                break;

            case EScene:
                m_scenes.push_back(static_cast<Scene *>(obj));
                break;

            default:
                throw NoriException("StudentsTTest::addChild(<%s>) is not supported!",
                    classTypeName(obj->getClassType()));
        }
    }

    /// Invoke a series of t-tests on the provided input
    void execute() {
        int total = 0, passed = 0;
        pcg32 random;

        if (!m_bsdfs.empty()) {
            if (m_references.size() * m_bsdfs.size() != m_angles.size())
                throw NoriException("Specified a different number of angles and reference values!");
            if (!m_scenes.empty())
                throw NoriException("Cannot test BSDFs and scenes at the same time!");

            /* Test each registered BSDF */
            int ctr = 0;
            for (auto bsdf : m_bsdfs) {
                for (size_t i=0; i<m_references.size(); ++i) {
                    float angle = m_angles[i], reference = m_references[ctr++];

                    cout << "------------------------------------------------------" << endl;
                    cout << "Testing (angle=" << angle << "): " << bsdf->toString() << endl;
                    ++total;

                    BSDFQueryRecord bRec(sphericalDirection(degToRad(angle), 0));

                    cout << "Drawing " << m_sampleCount << " samples .. " << endl;
                    double mean=0, variance = 0;
                    for (int k=0; k<m_sampleCount; ++k) {
                        Point2f sample(random.nextFloat(), random.nextFloat());
                        double result = (double) bsdf->sample(bRec, sample).getLuminance();

                        /* Numerically robust online variance estimation using an
                           algorithm proposed by Donald Knuth (TAOCP vol.2, 3rd ed., p.232) */
                        double delta = result - mean;
                        mean += delta / (double) (k+1);
                        variance += delta * (result - mean);
                    }
                    variance /= m_sampleCount - 1;
                    std::pair<bool, std::string>
                        result = hypothesis::students_t_test(mean, variance, reference,
                            m_sampleCount, m_significanceLevel, (int) m_references.size());

                    if (result.first)
                        ++passed;
                    cout << result.second << endl;
                }
            }
        } else {
            if (m_references.size() != m_scenes.size())
                throw NoriException("Specified a different number of scenes and reference values!");

            Sampler *sampler = static_cast<Sampler *>(
                NoriObjectFactory::createInstance("independent", PropertyList()));

            int ctr = 0;
            for (auto scene : m_scenes) {
                const Integrator *integrator = scene->getIntegrator();
                const Camera *camera = scene->getCamera();
                float reference = m_references[ctr++];

                cout << "------------------------------------------------------" << endl;
                cout << "Testing scene: " << scene->toString() << endl;
                ++total;

                cout << "Generating " << m_sampleCount << " paths.. " << endl;

                double mean = 0, variance = 0;
                for (int k=0; k<m_sampleCount; ++k) {
                    /* Sample a ray from the camera */
                    Ray3f ray;
                    Point2f pixelSample = (sampler->next2D().array()
                        * camera->getOutputSize().cast<float>().array()).matrix();
                    Color3f value = camera->sampleRay(ray, pixelSample, sampler->next2D());

                    /* Compute the incident radiance */
                    value *= integrator->Li(scene, sampler, ray);

                    /* Numerically robust online variance estimation using an
                       algorithm proposed by Donald Knuth (TAOCP vol.2, 3rd ed., p.232) */
                    double result = (double) value.getLuminance();
                    double delta = result - mean;
                    mean += delta / (double) (k+1);
                    variance += delta * (result - mean);
                }
                variance /= m_sampleCount - 1;

                std::pair<bool, std::string>
                    result = hypothesis::students_t_test(mean, variance, reference,
                        m_sampleCount, m_significanceLevel, (int) m_references.size());

                if (result.first)
                    ++passed;
                cout << result.second << endl;
            }
        }
        cout << "Passed " << passed << "/" << total << " tests." << endl;
    }

    virtual std::string toString() const override {
        return tfm::format(
            "StudentsTTest[\n"
            "  significanceLevel = %f,\n"
            "  sampleCount= %i\n"
            "]",
            m_significanceLevel,
            m_sampleCount
        );
    }

    virtual EClassType getClassType() const override { return ETest; }
#ifndef NORI_USE_NANOGUI
    virtual const char *getImGuiName() const override { return "TTest"; }
    virtual bool getImGuiNodes() override { return false; }
#endif
private:
    std::vector<BSDF *> m_bsdfs;
    std::vector<Scene *> m_scenes;
    std::vector<float> m_angles;
    std::vector<float> m_references;
    float m_significanceLevel;
    int m_sampleCount;
};

NORI_REGISTER_CLASS(StudentsTTest, "ttest");
NORI_NAMESPACE_END
