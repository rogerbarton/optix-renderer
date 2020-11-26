#include <nori/emitter.h>
#include <nori/texture.h>
#include <Eigen/Geometry>
#include <nori/warp.h>

NORI_NAMESPACE_BEGIN

class EnvMap : public Emitter
{
public:
	explicit EnvMap(const PropertyList &props) {}

	NoriObject *cloneAndInit() override
	{
		// Use constant texture as a fallback
		if (!m_map)
		{
			PropertyList l;
			l.setColor("value", Color3f(0.5f));
			m_map = static_cast<Texture<Color3f> *>(NoriObjectFactory::createInstance("constant_color", l));
		}

		auto clone = new EnvMap(*this);
		clone->m_map = static_cast<Texture<Color3f> *>(m_map->cloneAndInit());

		return clone;
	}

	void update(const NoriObject *guiObject) override
	{
		const auto *gui = static_cast<const EnvMap *>(guiObject);
		if (!gui->touched)
			return;
		gui->touched = false;

		m_map->update(gui->m_map);

		calculateProbs();
	}

	~EnvMap()
	{
		delete m_map;
	}

	void addChild(NoriObject *obj) override
	{
		switch (obj->getClassType())
		{
		case ETexture:
			if (obj->getIdName() == "albedo")
			{
				if (m_map)
					throw NoriException("There is already an albedo defined!");
				m_map = static_cast<Texture<Color3f> *>(obj);
			}
			else
			{
				throw NoriException("The name of this texture does not match any field!");
			}
			break;

		default:
			throw NoriException("EnvMap::addChild(<%s>) is not supported!",
								classTypeName(obj->getClassType()));
		}
	}

	std::string toString() const override
	{
		return tfm::format("PngEnvMap[\n"
						   "  texture = %s\n"
						   "]",
						   m_map->toString());
	};

	Color3f sample(EmitterQueryRecord &lRec,
				   const Point2f &sample) const
	{
		// sample any point based on the probabilities
		Histogram::upair result;

		Histogram::elem_type it = histogram.getElement(sample.x());
		if (it != histogram.map.end())
		{
			result = it->second;
		}
		else
		{
			throw NoriException("Histogram could not find a data point...");
		}

		// convert result (kind of uv coords) into direction
		float i = result.first / (float)m_map->getHeight();
		float j = result.second / (float)m_map->getWidth();

		Vector3f v;
		if (m_map->getHeight() == 1 && m_map->getWidth() == 1)
		{
			// sample a uniform direction from the scene (this only happens if the texture is constant)
			v = Warp::squareToUniformSphere(sample);
		}
		else
		{
			v = sphericalDirection(j * M_PI, i * 2.0f * M_PI);
		}
		Vector3f v_inf = v * 1.f / Epsilon; // divide by epsilon = * inf
		lRec.n = -v;						// the normal points inwards
		lRec.p = v_inf;
		lRec.wi = (lRec.p - lRec.ref).normalized();
		lRec.shadowRay = Ray3f(lRec.p, -lRec.wi, Epsilon, (lRec.p - lRec.ref).norm() - Epsilon);

		lRec.pdf = pdf(lRec);
		return eval(lRec) / lRec.pdf;
	}

	float pdf(const EmitterQueryRecord &lRec) const
	{
		// adaptive sampling based on the brightness of each pixel

		if (m_map->getHeight() == 1 && m_map->getWidth() == 1)
		{
			return Warp::squareToUniformSpherePdf(Vector3f(1.f, 0.f, 0.f));
		}

		Vector3f target = lRec.p;
		Point2f uv = sphericalCoordinates(target);

		// convert these uv coords into x and y for the probability
		unsigned int i = uv.y() / (2.f * M_PI) * m_map->getWidth();
		unsigned int j = uv.x() * INV_PI * m_map->getHeight();

		i = i + m_map->getHeight() % m_map->getHeight();
		j = j + m_map->getWidth() % m_map->getWidth();

		// second and third part is the probability of sampling one pixel (in solid angles)
		return probabilities(i, j) / Warp::squareToUniformSpherePdf(Vector3f(1.f, 0.f, 0.f)) * m_map->getHeight() * m_map->getWidth();
	}

	Color3f eval(const EmitterQueryRecord &lRec) const override
	{
		// ref does not have to be set, because the env map has inf size
		// --> the distance of ref/center of envmap is neglectable

		// eval texture based on lRec.wi
		Point2f uv_coords = sphericalCoordinates(lRec.wi);
		Point2f uv;

		// switch coordinates and map to [0,1]
		uv.x() = uv_coords.y() / (2.f * M_PI);
		uv.y() = uv_coords.x() / M_PI;

		return m_map->eval(uv);
	}

	Color3f samplePhoton(Ray3f &ray, const Point2f &sample1, const Point2f &sample2) const override
	{
		EmitterQueryRecord EQR;
		auto Li = this->sample(EQR, sample1);

		//set shadowray
		ray = EQR.shadowRay;

		//compute pdf of sampling random point
		auto pdf = this->pdf(EQR);

		return Li / pdf;
	}

#ifdef NORI_USE_IMGUI
	NORI_OBJECT_IMGUI_NAME("Environment Map");
	virtual bool getImGuiNodes() override
	{
		touched |= Emitter::getImGuiNodes();

		if (m_map)
		{
			bool node_open = ImGui::TreeNode("Texture");
			ImGui::NextColumn();
			ImGui::AlignTextToFramePadding();

			ImGui::Text(m_map->getImGuiName().c_str());
			ImGui::NextColumn();
			if (node_open)
			{
				touched |= m_map->getImGuiNodes();
				ImGui::TreePop();
			}
		}

		return touched;
	}
#endif

	bool isEnvMap() const override
	{
		return true;
	}

private:
	/// calculate the histogram based on the probabilities
	void calculateProbs()
	{
		probabilities = MatrixXf(m_map->getHeight(), m_map->getWidth());
		for (unsigned int i = 0; i < m_map->getHeight(); i++)
		{
			for (unsigned int j = 0; j < m_map->getWidth(); j++)
			{
				Color3f col = m_map->eval(Point2f(i / (float)m_map->getHeight(), j / (float)m_map->getWidth()));
				probabilities(i, j) = col.getLuminance(); // bias to possibly select every one once
			}
		}
		//probabilities.normalize();
		for (unsigned int i = 0; i < m_map->getHeight(); i++)
		{
			for (unsigned int j = 0; j < m_map->getWidth(); j++)
			{
				histogram.add_element(i, j, probabilities(i, j));
			}
		}
	}

	Texture<Color3f> *m_map = nullptr;
	Histogram histogram;
	MatrixXf probabilities;
};

NORI_REGISTER_CLASS(EnvMap, "envmap");
NORI_NAMESPACE_END