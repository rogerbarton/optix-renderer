#include <nori/environmentmap.h>
#include <nori/texture.h>
#include <Eigen/Geometry>

NORI_NAMESPACE_BEGIN

class PNGEnvMap : public EnvironmentMap
{
public:
	explicit PNGEnvMap(const PropertyList &props)
	{
		PropertyList l;
		l.setColor("value", props.has("envmap") ? props.getColor("albedo") : Color3f(0.5f));
		m_map = dynamic_cast<Texture<Color3f> *>(NoriObjectFactory::createInstance("constant_color", l));

		scaleU = props.getFloat("scaleU", 1.f);
		scaleV = props.getFloat("scaleV", 1.f);

		sphereTexture = props.getBoolean("sphereTexture", false);
	}

	NoriObject *cloneAndInit() override {
		auto clone = new PNGEnvMap(*this);
		clone->m_map = dynamic_cast<Texture<Color3f>*>(m_map->cloneAndInit());
		return clone;
	}

	void update(const NoriObject* guiObject) override
	{
		const auto* gui = dynamic_cast<const PNGEnvMap*>(guiObject);
		scaleU = gui->scaleU;
		scaleV = gui->scaleV;
		sphereTexture = gui->sphereTexture;
		m_map->update(gui->m_map);
	}

	~PNGEnvMap()
	{
		delete m_map;
	}

	void addChild(NoriObject *obj) override
	{
		switch (obj->getClassType())
		{
		case ETexture:
			if (obj->getIdName() == "envmap")
			{
				if (m_map)
					throw NoriException("There is already an envmap defined!");
				m_map = dynamic_cast<Texture<Color3f> *>(obj);
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
						   "  envmap = %s,\n"
						   "  scaleU = %f,\n"
						   "  scaleV = %f,\n"
						   "  sphereTexture = %d\n"
						   "]",
						   m_map->toString(), scaleU, scaleV, sphereTexture);
	};

	//use wi as the escaping ray
	Color3f eval(const Vector3f &_wi) override
	{
		if (!sphereTexture)
		{
			const float uw = 0.25f;
			const float uh = 1.f / 3.f;

			const Vector3f &r = _wi;
			Point2f uv;

			float u, v;
			int index;
			convert_xyz_to_cube_uv(r[0], r[1], r[2], &index, &u, &v);

			v = 1.f - v;

			u *= scaleU;
			v *= scaleV;

			u *= uw;
			v *= uh;

			switch (index)
			{
			case 0: //X+
				u += 2 * uw;
				v += uh;
				break;
			case 1: //X-
				v += uh;
				break;
			case 3: //Y+
				u += uw;
				v += 2 * uh;
				break;
			case 2: //Y-
				u += uw;
				break;
			case 4: //Z+
				u += uw;
				v += uh;
				break;
			case 5: //Z-
				u += 3 * uw;
				v += uh;
				break;
			}
			return m_map->eval(Point2f(u, v));
		}
		else
		{
			// eval texture based on _wi
			Point2f uv_coords = sphericalCoordinates(_wi);
			Point2f uv;

			// switch coordinates and map to [0,1]
			uv.x() = uv_coords.y() / (2.f * M_PI);
			uv.y() = uv_coords.x() / M_PI;

			return m_map->eval(uv);
		}
	}
#ifndef NORI_USE_NANOGUI
	virtual const char *getImGuiName() const override { return "PNG Environment Map"; }
	virtual bool getImGuiNodes() override
	{
		bool ret = EnvironmentMap::getImGuiNodes();

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

		ImGui::AlignTextToFramePadding();
        ImGui::PushID(3);
        ImGui::TreeNodeEx("SphereTexture", flags, "Sphere Texture");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        ret |= ImGui::Checkbox("##value", &sphereTexture);
        ImGui::NextColumn();
        ImGui::PopID();

		return ret;
	}
#endif
private:
	Texture<Color3f> *m_map = nullptr;
	bool sphereTexture = false;
	float scaleU, scaleV;

	void convert_xyz_to_cube_uv(float x, float y, float z, int *index, float *u, float *v)
	{
		float absX = fabs(x);
		float absY = fabs(y);
		float absZ = fabs(z);

		int isXPositive = x > 0 ? 1 : 0;
		int isYPositive = y > 0 ? 1 : 0;
		int isZPositive = z > 0 ? 1 : 0;

		float maxAxis = 0, uc = 0, vc = 0;

		// POSITIVE X
		if (isXPositive && absX >= absY && absX >= absZ)
		{
			// u (0 to 1) goes from +z to -z
			// v (0 to 1) goes from -y to +y
			maxAxis = absX;
			uc = -z;
			vc = y;
			*index = 0;
		}
		// NEGATIVE X
		if (!isXPositive && absX >= absY && absX >= absZ)
		{
			// u (0 to 1) goes from -z to +z
			// v (0 to 1) goes from -y to +y
			maxAxis = absX;
			uc = z;
			vc = y;
			*index = 1;
		}
		// POSITIVE Y
		if (isYPositive && absY >= absX && absY >= absZ)
		{
			// u (0 to 1) goes from -x to +x
			// v (0 to 1) goes from +z to -z
			maxAxis = absY;
			uc = x;
			vc = -z;
			*index = 2;
		}
		// NEGATIVE Y
		if (!isYPositive && absY >= absX && absY >= absZ)
		{
			// u (0 to 1) goes from -x to +x
			// v (0 to 1) goes from -z to +z
			maxAxis = absY;
			uc = x;
			vc = z;
			*index = 3;
		}
		// POSITIVE Z
		if (isZPositive && absZ >= absX && absZ >= absY)
		{
			// u (0 to 1) goes from -x to +x
			// v (0 to 1) goes from -y to +y
			maxAxis = absZ;
			uc = x;
			vc = y;
			*index = 4;
		}
		// NEGATIVE Z
		if (!isZPositive && absZ >= absX && absZ >= absY)
		{
			// u (0 to 1) goes from +x to -x
			// v (0 to 1) goes from -y to +y
			maxAxis = absZ;
			uc = -x;
			vc = y;
			*index = 5;
		}

		// Convert range from -1 to 1 to 0 to 1
		*u = 0.5f * (uc / maxAxis + 1.0f);
		*v = 0.5f * (vc / maxAxis + 1.0f);
	}
};

NORI_REGISTER_CLASS(PNGEnvMap, "png_env");
NORI_NAMESPACE_END