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

#include <nori/mesh.h>
#include <nori/timer.h>
#include <filesystem/resolver.h>
#include <unordered_map>
#include <fstream>
#include <filesystem>
#include <imgui/filebrowser.h>
#include <Eigen/Geometry>

NORI_NAMESPACE_BEGIN

/**
 * \brief Loader for Wavefront OBJ triangle meshes
 */
class WavefrontOBJ : public Mesh
{
public:
    explicit WavefrontOBJ(const PropertyList &propList)
    {
	    filename = getFileResolver()->resolve(propList.getString("filename")).str();
	    trafo = propList.getTransform("toWorld", Transform());
    }

	NoriObject *cloneAndInit() override
	{
		auto clone = new WavefrontOBJ(*this);
		Shape::cloneAndInit(clone);
		return clone;
	}

	void update(const NoriObject *guiObject) override
	{
    	const auto* gui = static_cast<const WavefrontOBJ*>(guiObject);
		if (!gui->touched) return;
		gui->touched = false;

    	if(gui->fileTouched || gui->geometryTouched || gui->transformTouched)
	    {
		    gui->fileLastReadTime = std::filesystem::last_write_time(gui->filename);
		    filename = gui->filename;
		    loadFromFile();
	    }

    	// Update sub-object regardless of transformTouched
	    trafo.update(gui->trafo);

		Mesh::update(guiObject);

		gui->geometryTouched  = false;
		gui->transformTouched = false;
		gui->fileTouched      = false;
	}

	/**
	 * Load the .obj file
	 * Sets and also resets: m_V, m_F, m_N, m_UV
	 * This will also handle if a file has already been loaded.
	 */
	void loadFromFile() {
	    typedef std::unordered_map<OBJVertex, uint32_t, OBJVertexHash> VertexMap;

	    std::ifstream is(filename);
	    if (is.fail())
		    throw NoriException("Unable to open OBJ file \"%s\"!", filename);

	    cout << "Loading \"" << filename << "\" .. ";
	    cout.flush();
	    Timer timer;

	    std::vector<Vector3f> positions;
	    std::vector<Vector2f> texcoords;
	    std::vector<Vector3f> normals;
	    std::vector<uint32_t> indices;
	    std::vector<OBJVertex> vertices;
	    VertexMap vertexMap;

	    std::string line_str;
	    while (std::getline(is, line_str))
	    {
		    std::istringstream line(line_str);

		    std::string prefix;
		    line >> prefix;

		    if (prefix == "v")
		    {
			    Point3f p;
			    line >> p.x() >> p.y() >> p.z();
			    p = trafo * p;
			    m_bbox.expandBy(p);
			    positions.push_back(p);
		    }
		    else if (prefix == "vt")
		    {
			    Point2f tc;
			    line >> tc.x() >> tc.y();
			    texcoords.push_back(tc);
		    }
		    else if (prefix == "vn")
		    {
			    Normal3f n;
			    line >> n.x() >> n.y() >> n.z();
			    normals.push_back((trafo * n).normalized());
		    }
		    else if (prefix == "f")
		    {
			    std::string v1, v2, v3, v4;
			    line >> v1 >> v2 >> v3 >> v4;
			    OBJVertex verts[6];
			    int nVertices = 3;

			    verts[0] = OBJVertex(v1);
			    verts[1] = OBJVertex(v2);
			    verts[2] = OBJVertex(v3);

			    if (!v4.empty())
			    {
				    /* This is a quad, split into two triangles */
				    verts[3] = OBJVertex(v4);
				    verts[4] = verts[0];
				    verts[5] = verts[2];
				    nVertices = 6;
			    }
			    /* Convert to an indexed vertex list */
			    for (int i = 0; i < nVertices; ++i)
			    {
				    const OBJVertex &v = verts[i];
				    VertexMap::const_iterator it = vertexMap.find(v);
				    if (it == vertexMap.end())
				    {
					    vertexMap[v] = (uint32_t)vertices.size();
					    indices.push_back((uint32_t)vertices.size());
					    vertices.push_back(v);
				    }
				    else
				    {
					    indices.push_back(it->second);
				    }
			    }
		    }
	    }

	    m_F.resize(3, indices.size() / 3);
	    memcpy(m_F.data(), indices.data(), sizeof(uint32_t) * indices.size());

	    m_V.resize(3, vertices.size());
	    for (uint32_t i = 0; i < vertices.size(); ++i)
		    m_V.col(i) = positions.at(vertices[i].p - 1);

	    m_N.resize(3, normals.empty() ? 0 : vertices.size());
	    if (!normals.empty())
	    {
		    for (uint32_t i = 0; i < vertices.size(); ++i)
			    m_N.col(i) = normals.at(vertices[i].n - 1);
	    }

	    m_UV.resize(2, texcoords.empty() ? 0 : vertices.size());
	    if (!texcoords.empty())
	    {
		    for (uint32_t i = 0; i < vertices.size(); ++i)
			    m_UV.col(i) = texcoords.at(vertices[i].uv - 1);
	    }

		//tangents
			if(!normals.empty() && !texcoords.empty()){
				m_T.resize(3, vertices.size());
				m_BT.resize(3, vertices.size());
				
				uint32_t triaCount = indices.size() / 3;
				
				for (uint32_t i = 0; i < triaCount; ++i) {

					uint32_t i1 = indices[3 * i];
					uint32_t i2 = indices[3 * i + 1];
					uint32_t i3 = indices[3 * i + 2];
			
					const Vector3f& v0 = m_V.col(i1);
					const Vector3f& v1 = m_V.col(i2);
					const Vector3f& v2 = m_V.col(i3);

					const Point2f& uv0 = m_UV.col(i1);
					const Point2f& uv1 = m_UV.col(i2);
					const Point2f& uv2 = m_UV.col(i3);
					
					const Vector3f Edge1 = v1 - v0;
					const Vector3f Edge2 = v2 - v0;

					float DeltaU1 = uv1[0] - uv0[0];
					float DeltaV1 = uv1[1] - uv0[1];
					float DeltaU2 = uv2[0] - uv0[0];
					float DeltaV2 = uv2[1] - uv0[1];

					float frac = DeltaU1 * DeltaV2 - DeltaU2 * DeltaV1;
					float f = 1.f / (frac == 0.f ? Epsilon : frac); //fixes an edge case

					Vector3f Tangent(
						f * (DeltaV2 * Edge1.x() - DeltaV1 * Edge2.x()),
						f * (DeltaV2 * Edge1.y() - DeltaV1 * Edge2.y()),
						f * (DeltaV2 * Edge1.z() - DeltaV1 * Edge2.z())
					);
							
					m_T.col(i1) += trafo * Tangent;
					m_T.col(i2) += trafo * Tangent;
					m_T.col(i3) += trafo * Tangent;

					m_BT.col(i1) += trafo * Vector3f(Vector3f(m_N.col(i1)).cross(Tangent));
					m_BT.col(i2) += trafo * Vector3f(Vector3f(m_N.col(i2)).cross(Tangent));
					m_BT.col(i3) += trafo * Vector3f(Vector3f(m_N.col(i3)).cross(Tangent));
				}
			} 

	    m_name = filename.string();
	    cout << "done. (V=" << m_V.cols() << ", F=" << m_F.cols() << ", N=" << m_N.cols() << ", UV=" << m_UV.cols() << ", took "
	         << timer.elapsedString() << " and "
	         << memString(m_F.size() * sizeof(uint32_t) +
	                      sizeof(float) * (m_V.size() + m_N.size() + m_UV.size()))
	         << ")" << endl;
    }

#ifdef NORI_USE_IMGUI
	NORI_OBJECT_IMGUI_NAME("Mesh");
	virtual bool getImGuiNodes() override
	{
		ImGui::PushID(EShape);
		touched |= Mesh::getImGuiNodes();

		ImGui::AlignTextToFramePadding();
		ImGui::TreeNodeEx("name", ImGuiLeafNodeFlags, "Filename");
		ImGui::NextColumn();
		ImGui::SetNextItemWidth(-1);
		ImGui::Text(tfm::format("%s%s", filename.filename().string().c_str(), (fileTouched ? "*" : "")).c_str());
		ImGui::NextColumn();

		// -- Change filename
		ImGui::NextColumn(); // skip column
		static ImGui::FileBrowser fileBrowser;
		if (ImGui::Button("Open"))
		{
			fileBrowser.Open();
			fileBrowser.SetTitle("Open Mesh File");
			fileBrowser.SetTypeFilters({".obj"});
			if(filename.has_parent_path())
				fileBrowser.SetPwd(filename.parent_path());
		}

		ImGui::SameLine();
		if (ImGui::Button("Refresh"))
			geometryTouched |= std::filesystem::last_write_time(filename) > fileLastReadTime;
		ImGui::NextColumn();

		fileBrowser.Display();
		if (fileBrowser.HasSelected())
		{
			filename = fileBrowser.GetSelected();
			fileTouched = true;
			fileBrowser.ClearSelected();
		}

		// -- Remaining Properties
		ImGui::AlignTextToFramePadding();
		ImGui::PushID(0);
		bool node_open = ImGui::TreeNode("Transform");
		ImGui::NextColumn();
		ImGui::SetNextItemWidth(-1);
		ImGui::Text("To World");
		ImGui::NextColumn();
		if(node_open) {
			transformTouched |= trafo.getImGuiNodes();
			ImGui::TreePop();
		}
		ImGui::PopID();

		ImGui::PopID();

		geometryTouched |= fileTouched;
		geometryTouched |= transformTouched; // Because obj bakes the transform into the mesh, we must reload the mesh
		touched |= transformTouched | geometryTouched;
		return touched;
	}
#endif

protected:
    /// Vertex indices used by the OBJ format
    struct OBJVertex
    {
        uint32_t p = (uint32_t)-1;
        uint32_t n = (uint32_t)-1;
        uint32_t uv = (uint32_t)-1;

        inline OBJVertex() {}

        inline OBJVertex(const std::string &string)
        {
            std::vector<std::string> tokens = tokenize(string, "/", true);

            if (tokens.size() < 1 || tokens.size() > 3)
                throw NoriException("Invalid vertex data: \"%s\"", string);

            p = toUInt(tokens[0]);

            if (tokens.size() >= 2 && !tokens[1].empty())
                uv = toUInt(tokens[1]);

            if (tokens.size() >= 3 && !tokens[2].empty())
                n = toUInt(tokens[2]);
        }

        inline bool operator==(const OBJVertex &v) const
        {
            return v.p == p && v.n == n && v.uv == uv;
        }
    };

    /// Hash function for OBJVertex
    struct OBJVertexHash : std::function<size_t(OBJVertex)>
    {
        std::size_t operator()(const OBJVertex &v) const
        {
            size_t hash = std::hash<uint32_t>()(v.p);
            hash = hash * 37 + std::hash<uint32_t>()(v.uv);
            hash = hash * 37 + std::hash<uint32_t>()(v.n);
            return hash;
        }
    };
private:
	std::filesystem::path filename;
	Transform trafo;

	mutable std::filesystem::file_time_type fileLastReadTime;
	mutable bool fileTouched = true;
};

NORI_REGISTER_CLASS(WavefrontOBJ, "obj");
NORI_NAMESPACE_END
