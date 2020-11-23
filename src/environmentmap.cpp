#include <nori/emitter.h>
#include <nori/texture.h>
#include <Eigen/Geometry>
#include <nori/warp.h>

NORI_NAMESPACE_BEGIN

class EnvMap : public Emitter
{
public:
	EnvMap(const PropertyList &props)
	{
		if (props.has("albedo"))
		{
			PropertyList l;
			l.setColor("value", props.getColor("albedo"));
			m_map = static_cast<Texture<Color3f> *>(NoriObjectFactory::createInstance("constant_color", l));
		}

		scaleU = props.getFloat("scaleU", 1.f);
		scaleV = props.getFloat("scaleV", 1.f);
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

	void activate() override
	{
		if (!m_map)
		{
			PropertyList l;
			l.setColor("value", Color3f(0.5f));
			m_map = static_cast<Texture<Color3f> *>(NoriObjectFactory::createInstance("constant_color", l));
			m_map->activate();
		}

		calculateProbs();
	}

	std::string toString() const override
	{
		return tfm::format("PngEnvMap[\n"
						   "  texture = %s,\n"
						   "  scaleU = %f,\n"
						   "  scaleV = %f,\n"
						   "]",
						   m_map->toString(), scaleU, scaleV);
	};

	Color3f sample(EmitterQueryRecord &lRec,
				   const Point2f &sample) const
	{
		// sample any point based on the probabilities

		Histogram::upair result;
		while (true)
		{
			Histogram::elem_type it = histogram.getElement(sample.x());
			if (it != histogram.map.end())
			{
				result = it->second;
				break;
			}
			else
			{
				throw NoriException("Histogram could not find a data point...");
			}
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
		lRec.n = -(v_inf - m_position).normalized(); // the normal points inwards to m_position
		lRec.p = v_inf;
		lRec.wi = (lRec.p - lRec.ref).normalized();
		lRec.shadowRay = Ray3f(lRec.p, -lRec.wi, Epsilon, (lRec.p - lRec.ref).norm() - Epsilon);

		lRec.pdf = pdf(lRec);

		if (lRec.pdf < Epsilon)
			return Color3f(0.f);

		Color3f col = eval(lRec) / lRec.pdf;
		return col;
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

		return probabilities(i, j) / Warp::squareToUniformSpherePdf(Vector3f(1.f, 0.f, 0.f));
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
#ifndef NORI_USE_NANOGUI
	virtual const char *getImGuiName() const override
	{
		return "Environment Map";
	}
	virtual bool getImGuiNodes() override
	{
		bool ret = Emitter::getImGuiNodes();

		ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen |
								   ImGuiTreeNodeFlags_Bullet;

		if (m_map)
		{
			bool node_open = ImGui::TreeNode("Texture");
			ImGui::NextColumn();
			ImGui::AlignTextToFramePadding();

			ImGui::Text(m_map->getImGuiName());
			ImGui::NextColumn();
			if (node_open)
			{
				ret |= m_map->getImGuiNodes();
				ImGui::TreePop();
			}
		}

		ImGui::AlignTextToFramePadding();
		ImGui::PushID(1);
		ImGui::TreeNodeEx("scale U", flags, "Scale U");
		ImGui::NextColumn();
		ImGui::SetNextItemWidth(-1);
		ret |= ImGui::DragFloat("##value", &scaleU, 0.01, 0, 10.f, "%f%", ImGuiSliderFlags_AlwaysClamp);
		ImGui::NextColumn();
		ImGui::PopID();

		ImGui::AlignTextToFramePadding();
		ImGui::PushID(2);
		ImGui::TreeNodeEx("scale V", flags, "Scale V");
		ImGui::NextColumn();
		ImGui::SetNextItemWidth(-1);
		ret |= ImGui::DragFloat("##value", &scaleV, 0.01, 0, 10.f, "%f%", ImGuiSliderFlags_AlwaysClamp);
		ImGui::NextColumn();
		ImGui::PopID();

		return ret;
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
		for (int i = 0; i < m_map->getHeight(); i++)
		{
			for (int j = 0; j < m_map->getWidth(); j++)
			{
				Color3f col = m_map->eval(Point2f(i / (float)m_map->getHeight(), j / (float)m_map->getWidth()));
				probabilities(i, j) = col.getLuminance(); // bias to possibly select every one once
			}
		}
		probabilities.normalize();
		for (int i = 0; i < m_map->getHeight(); i++)
		{
			for (int j = 0; j < m_map->getWidth(); j++)
			{
				histogram.add_element(i, j, probabilities(i, j));
			}
		}
	}

	Texture<Color3f> *m_map = nullptr;
	float scaleU, scaleV;
	Histogram histogram;
	MatrixXf probabilities;
};

NORI_REGISTER_CLASS(EnvMap, "envmap");
NORI_NAMESPACE_END