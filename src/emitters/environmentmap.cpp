#include <nori/emitter.h>
#include <nori/texture.h>
#include <Eigen/Geometry>
#include <nori/warp.h>
#include <nori/dpdf.h>

NORI_NAMESPACE_BEGIN

class EnvMap : public Emitter
{
public:
	explicit EnvMap(const PropertyList &props) {
		lightProb = props.getFloat("lightWeight", 1.f);
		m_radiance = props.getColor("radiance", 1.f);
	}

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

		Emitter::cloneAndInit(clone);

		return clone;
	}

	void update(const NoriObject *guiObject) override
	{
		const auto *gui = static_cast<const EnvMap *>(guiObject);
		if (!gui->touched)
			return;
		gui->touched = false;

		m_map->update(gui->m_map);

		Emitter::update(gui);

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
				   const Point2f &sample) const override
	{
		// sample any point based on the probabilities
		size_t elem = dpdf.sample(sample.x());

		// convert result (kind of uv coords) into direction
		float i = int(elem / m_map->getWidth()) / (float)m_map->getHeight();
		float j = int(elem % m_map->getWidth()) / (float)m_map->getWidth();

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

		if (lRec.pdf < Epsilon)
			return Color3f(0.f);

		return eval(lRec) / lRec.pdf;
	}

	float pdf(const EmitterQueryRecord &lRec) const override
	{
		// adaptive sampling based on the brightness of each pixel

		if (m_map->getHeight() == 1 && m_map->getWidth() == 1)
		{
			return Warp::squareToUniformSpherePdf(Vector3f(1.f, 0.f, 0.f));
		}

		// second and third part is the probability of sampling one pixel (in solid angles)
		return eval(lRec).getLuminance() * dpdf.getNormalization() / Warp::squareToUniformSpherePdf(Vector3f(1.f, 0.f, 0.f)) * m_map->getHeight() * m_map->getWidth();
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

		return m_map->eval(uv) * m_radiance;
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
	void calculateProbs()
	{
		dpdf.clear();
		dpdf.reserve(m_map->getHeight() * m_map->getWidth());
		for (unsigned int i = 0; i < m_map->getHeight(); i++)
		{
			for (unsigned int j = 0; j < m_map->getWidth(); j++)
			{
				Color3f col = m_map->eval(Point2f(i / (float)m_map->getHeight(), j / (float)m_map->getWidth()));
				dpdf.append(std::abs(col.getLuminance()) /*+ Epsilon */); // add epsilon to add prob to select every pixel once
			}
		}

		dpdf.normalize();
	}

	Texture<Color3f> *m_map = nullptr;
	DiscretePDF dpdf;
};

NORI_REGISTER_CLASS(EnvMap, "envmap");
NORI_NAMESPACE_END