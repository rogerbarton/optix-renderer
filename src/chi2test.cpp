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

#include <nori/bsdf.h>
#include <nori/warp.h>
#include <pcg32.h>
#include <hypothesis.h>
#include <fstream>
#include <memory>
#include <cassert>

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
 * \brief Statistical test for validating that an importance sampling routine
 * (e.g. from a BSDF) produces a distribution that agrees with what the
 * implementation claims via its associated density function.
 */
class ChiSquareTest : public NoriObject {
public:
    explicit ChiSquareTest(const PropertyList &propList) {
        /* The null hypothesis will be rejected when the associated
           p-value is below the significance level specified here. */
        m_significanceLevel = propList.getFloat("significanceLevel", 0.01f);

        /* Number of cells along the latitudinal axis. The azimuthal
           resolution is twice this value. */
        m_cosThetaResolution = propList.getInteger("resolution", 10);

        /* Minimum expected bin frequency. The chi^2 test does not
           work reliably when the expected frequency in a cell is
           low (e.g. less than 5), because normality assumptions
           break down in this case. Therefore, the implementation
           will merge such low-frequency cells when they fall below
           the threshold specified here. */
        m_minExpFrequency = propList.getInteger("minExpFrequency", 5);

        /* Number of samples that should be taken (-1: automatic) */
        m_sampleCount = propList.getInteger("sampleCount", -1);

        /* Each provided BSDF will be tested for a few different
           incident directions. The value specified here determines
           how many tests will be executed per BSDF */
        m_testCount = propList.getInteger("testCount", 5);

        m_phiResolution = 2 * m_cosThetaResolution;

        if (m_sampleCount < 0) // ~5K samples per bin
            m_sampleCount = m_cosThetaResolution * m_phiResolution * 5000;
    }

	NoriObject *cloneAndInit() override
	{
		auto clone = new ChiSquareTest(*this);
		assert(clone->m_bsdfs.size() == m_bsdfs.size());
		for (int i = 0; i < m_bsdfs.size(); ++i)
			clone->m_bsdfs[i] = static_cast<BSDF *>(m_bsdfs[i]->cloneAndInit());

		clone->execute();
		return clone;
	}

	void update(const NoriObject *guiObject) override
	{
		const auto* gui = static_cast<const ChiSquareTest *>(guiObject);
		if (!gui->touched)return;
		gui->touched = false;

		// -- Copy properties
		m_significanceLevel = gui->m_significanceLevel;
		m_cosThetaResolution = gui->m_cosThetaResolution;
		m_minExpFrequency = gui->m_minExpFrequency;
		m_sampleCount = gui->m_sampleCount;
		m_testCount = gui->m_testCount;

		// -- Update sub-objects
		assert(m_bsdfs.size() == gui->m_bsdfs.size());
		for (int i = 0; i < gui->m_bsdfs.size(); i++)
			m_bsdfs[i]->update(gui->m_bsdfs[i]);

		// -- Update derived properties
		m_phiResolution = 2 * m_cosThetaResolution;
		if (m_sampleCount < 0) // ~5K samples per bin
			m_sampleCount = m_cosThetaResolution * m_phiResolution * 5000;

		execute();
	}

    virtual ~ChiSquareTest() {
        for (auto bsdf : m_bsdfs)
            delete bsdf;
    }

    virtual void addChild(NoriObject *obj) override {
        switch (obj->getClassType()) {
            case EBSDF:
                m_bsdfs.push_back(static_cast<BSDF *>(obj));
                break;

            default:
                throw NoriException("ChiSquareTest::addChild(<%s>) is not supported!",
                    classTypeName(obj->getClassType()));
        }
    }

    /// Execute the chi-square test
    void execute() {

	    int passed = 0, total = 0, res = m_cosThetaResolution*m_phiResolution;
	    pcg32 random; /* Pseudorandom number generator */

	    std::unique_ptr<double[]> obsFrequencies(new double[res]);
	    std::unique_ptr<double[]> expFrequencies(new double[res]);


	    /* Test each registered BSDF */
	    for (auto bsdf : m_bsdfs) {
		    /* Run several tests per BSDF to be on the safe side */
		    for (int l = 0; l<m_testCount; ++l) {
			    memset(obsFrequencies.get(), 0, res*sizeof(double));
			    memset(expFrequencies.get(), 0, res*sizeof(double));

			    cout << "------------------------------------------------------" << endl;
			    cout << "Testing: " << bsdf->toString() << endl;
			    ++total;

			    float cosTheta = random.nextFloat();
			    float sinTheta = std::sqrt(std::max((float) 0, 1-cosTheta*cosTheta));
			    float sinPhi, cosPhi;
			    sincosf(2.0f * M_PI * random.nextFloat(), &sinPhi, &cosPhi);
			    Vector3f wi(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);

			    cout << "Accumulating " << m_sampleCount << " samples into a " << m_cosThetaResolution
			         << "x" << m_phiResolution << " contingency table .. ";
			    cout.flush();

			    /* Generate many samples from the BSDF and create
				   a histogram / contingency table */
			    BSDFQueryRecord bRec(wi);
			    for (int i=0; i<m_sampleCount; ++i) {
				    Point2f sample(random.nextFloat(), random.nextFloat());
				    Color3f result = bsdf->sample(bRec, sample);

				    if ((result.array() == 0).all())
					    continue;

				    int cosThetaBin = std::min(std::max(0, (int) std::floor((bRec.wo.z()*0.5f+0.5f)
				                                                            * m_cosThetaResolution)), m_cosThetaResolution-1);

				    float scaledPhi = std::atan2(bRec.wo.y(), bRec.wo.x()) * INV_TWOPI;
				    if (scaledPhi < 0)
					    scaledPhi += 1;

				    int phiBin = std::min(std::max(0,
				                                   (int) std::floor(scaledPhi * m_phiResolution)), m_phiResolution-1);
				    obsFrequencies[cosThetaBin * m_phiResolution + phiBin] += 1;
			    }
			    cout << "done." << endl;

			    /* Numerically integrate the probability density
				   function over rectangles in spherical coordinates. */
			    double *ptr = expFrequencies.get();
			    cout << "Integrating expected frequencies .. ";
			    cout.flush();
			    for (int i=0; i<m_cosThetaResolution; ++i) {
				    double cosThetaStart = -1.0 + i     * 2.0 / m_cosThetaResolution;
				    double cosThetaEnd   = -1.0 + (i+1) * 2.0 / m_cosThetaResolution;
				    for (int j=0; j<m_phiResolution; ++j) {
					    double phiStart = j     * 2*M_PI / m_phiResolution;
					    double phiEnd   = (j+1) * 2*M_PI / m_phiResolution;

					    auto integrand = [&](double cosTheta, double phi) -> double {
						    double sinTheta = std::sqrt(1 - cosTheta * cosTheta);
						    double sinPhi = std::sin(phi), cosPhi = std::cos(phi);

						    Vector3f wo((float) (sinTheta * cosPhi),
						                (float) (sinTheta * sinPhi),
						                (float) cosTheta);

						    BSDFQueryRecord bRec(wi, wo, ESolidAngle);
						    return bsdf->pdf(bRec);
					    };

					    double integral = hypothesis::adaptiveSimpson2D(
							    integrand, cosThetaStart, phiStart, cosThetaEnd,
							    phiEnd);

					    *ptr++ = integral * m_sampleCount;
				    }
			    }
			    cout << "done." << endl;

			    /* Write the test input data to disk for debugging */
			    hypothesis::chi2_dump(m_cosThetaResolution, m_phiResolution, obsFrequencies.get(), expFrequencies.get(),
			                          tfm::format("chi2test_%i.m", total));

			    /* Perform the Chi^2 test */
			    std::pair<bool, std::string> result =
					                                 hypothesis::chi2_test(m_cosThetaResolution*m_phiResolution, obsFrequencies.get(), expFrequencies.get(),
					                                                       m_sampleCount, m_minExpFrequency, m_significanceLevel, m_testCount * (int) m_bsdfs.size());

			    if (result.first)
				    ++passed;

			    cout << result.second << endl;
		    }
	    }

	    cout << "Passed " << passed << "/" << total << " tests." << endl;
    }

    virtual std::string toString() const override {
        return tfm::format("ChiSquareTest[\n"
            "  thetaResolution = %i,\n"
            "  phiResolution = %i,\n"
            "  minExpFrequency = %i,\n"
            "  sampleCount = %i,\n"
            "  testCount = %i,\n"
            "  significanceLevel = %f\n"
            "]",
            m_cosThetaResolution,
            m_phiResolution,
            m_minExpFrequency,
            m_sampleCount,
            m_testCount,
            m_significanceLevel
        );
    }

    virtual EClassType getClassType() const override { return ETest; }
#ifndef NORI_USE_NANOGUI
	NORI_OBJECT_IMGUI_NAME("Chi^2 Test");
	virtual bool getImGuiNodes() override { return false; }
#endif
private:
    int m_cosThetaResolution;
    int m_phiResolution;
    int m_minExpFrequency;
    int m_sampleCount;
    int m_testCount;
    float m_significanceLevel;
    std::vector<BSDF *> m_bsdfs;
};

NORI_REGISTER_CLASS(ChiSquareTest, "chi2test");
NORI_NAMESPACE_END
