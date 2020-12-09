#include <nori/perspective.h>

NORI_NAMESPACE_BEGIN

nori::PerspectiveCamera::PerspectiveCamera(const nori::PropertyList &propList) {
	/* Width and height in pixels. Default: 720p */
	m_outputSize.x() = propList.getInteger("width", 1280);
	m_outputSize.y() = propList.getInteger("height", 720);
	m_invOutputSize = m_outputSize.cast<float>().cwiseInverse();

	/* Specifies an optional camera-to-world transformation. Default: none */
	m_cameraToWorld = propList.getTransform("toWorld", Transform());

	/* Horizontal field of view in degrees */
	m_fov = propList.getFloat("fov", 30.0f);

	/* Near and far clipping planes in world-space units */
	m_nearClip = propList.getFloat("nearClip", 1e-4f);
	m_farClip  = propList.getFloat("farClip", 1e4f);

	// Depth of Field
	m_focalDistance = propList.getFloat("focalDistance", 10.f);
	m_fstop         = propList.getFloat("fstop", 0.f);
	m_lensRadius    = propList.getFloat("lensRadius", 0.f);

	m_rfilter = nullptr;
}
nori::NoriObject *nori::PerspectiveCamera::cloneAndInit() {
	// If no reconstruction filter was assigned, instantiate a Gaussian filter
	if (!m_rfilter)
		m_rfilter = static_cast<ReconstructionFilter *>(
				NoriObjectFactory::createInstance("gaussian"));

	if (m_fstop == 0.f)
		m_fstop = m_focalDistance / m_lensRadius;
	else
		m_lensRadius = m_focalDistance / m_fstop;

	auto clone = new PerspectiveCamera(*this);
	clone->m_rfilter = static_cast<ReconstructionFilter *>(m_rfilter->cloneAndInit());
	return clone;
}
void nori::PerspectiveCamera::update(const nori::NoriObject *guiObject) {
	const auto *gui = static_cast<const PerspectiveCamera *>(guiObject);
	if (!gui->touched) return;

	gui->touched = false;

	// -- Copy properties
	m_outputSize    = gui->m_outputSize;
	m_fov           = gui->m_fov;
	m_nearClip      = gui->m_nearClip;
	m_farClip       = gui->m_farClip;
	m_focalDistance = gui->m_focalDistance;
	m_fstop         = gui->m_fstop;
	m_lensRadius    = gui->m_lensRadius;

	// -- Update sub-objects
	m_rfilter->update(gui->m_rfilter);
	m_cameraToWorld.update(gui->m_cameraToWorld);

	// -- Recalculate derived properties
	m_invOutputSize = m_outputSize.cast<float>().cwiseInverse();
	float aspect = m_outputSize.x() / (float) m_outputSize.y();

	/** Project vectors in camera space onto a plane at z=1:
	 *
	 *  xProj = cot * x / z
	 *  yProj = cot * y / z
	 *  zProj = (far * (z - near)) / (z * (far-near))
	 *  The cotangent factor ensures that the field of view is
	 *  mapped to the interval [-1, 1].
	 */
	float recip = 1.0f / (m_farClip - m_nearClip),
	      cot   = 1.0f / std::tan(degToRad(m_fov / 2.0f));

	Eigen::Matrix4f perspective;
	perspective <<
	            cot, 0,   0,   0,
			0, cot,   0,   0,
			0,   0,   m_farClip * recip, -m_nearClip * m_farClip * recip,
			0,   0,   1,   0;

	/**
	 * Translation and scaling to shift the clip coordinates into the
	 * range from zero to one. Also takes the aspect ratio into account.
	 */
	m_sampleToCamera = Transform(
			Eigen::DiagonalMatrix<float, 3>(Vector3f(0.5f, -0.5f * aspect, 1.0f)) *
			Eigen::Translation<float, 3>(1.0f, -1.0f / aspect, 0.0f) * perspective).inverse();
}
nori::Color3f nori::PerspectiveCamera::sampleRay(nori::Ray3f &ray, const nori::Point2f &samplePosition,
                                                 const nori::Point2f &apertureSample) const {
	/* Compute the corresponding position on the
	   near plane (in local camera space) */
	const Point3f nearP = m_sampleToCamera * Point3f(
			samplePosition.x() * m_invOutputSize.x(),
			samplePosition.y() * m_invOutputSize.y(), 0.0f);

	/* Turn into a normalized ray direction, and
	   adjust the ray interval accordingly */
	const Vector3f d = nearP.normalized();

	// Create local space ray
	ray.d = d;
	ray.o = Point3f(0, 0, 0);

	// Depth of field, adjusts ray in local space
	if (m_lensRadius > Epsilon)
	{
		ray.update();

		static Sampler *const sampler = static_cast<Sampler *>(
				NoriObjectFactory::createInstance("independent"));

		// TODO: Also use squareToUniformTriangle for other bokeh effects
		const Point2f pLens  = m_lensRadius * Warp::squareToUniformDisk(sampler->next2D());
		const float   ft     = m_focalDistance / ray.d.z();
		// position of ray at time of intersection with the focal plane
		const Point3f pFocus = ray(ft);

		ray.o = Point3f(pLens.x(), pLens.y(), 0.f);
		// direction connecting aperture and focal plane points
		ray.d = (pFocus - ray.o).normalized();
	}

	ray.o = m_cameraToWorld * ray.o;
	ray.d = m_cameraToWorld * ray.d;

	const float invZ = 1.0f / d.z();
	ray.mint = m_nearClip * invZ;
	ray.maxt = m_farClip * invZ;
	ray.update();

	return Color3f(1.0f);
}
void nori::PerspectiveCamera::addChild(nori::NoriObject *obj) {
	switch (obj->getClassType())
	{
		case EReconstructionFilter:
			if (m_rfilter)
				throw NoriException("Camera: tried to register multiple reconstruction filters!");
			m_rfilter = static_cast<ReconstructionFilter *>(obj);
			break;

		default:
			throw NoriException("Camera::addChild(<%s>) is not supported!",
			                    classTypeName(obj->getClassType()));
	}
}
std::string nori::PerspectiveCamera::toString() const {
	return tfm::format(
			"PerspectiveCamera[\n"
			"  cameraToWorld = %s,\n"
			"  outputSize = %s,\n"
			"  fov = %f,\n"
			"  clip = [%f, %f],\n"
			"  rfilter = %s\n"
			"]",
			indent(m_cameraToWorld.toString(), 18),
			m_outputSize.toString(),
			m_fov,
			m_nearClip,
			m_farClip,
			indent(m_rfilter->toString())
	);
}
#ifdef NORI_USE_IMGUI

bool nori::PerspectiveCamera::getImGuiNodes() {
	touched |= Camera::getImGuiNodes();

	ImGui::AlignTextToFramePadding();
	ImGui::PushID(0);
	bool node_open = ImGui::TreeNode("Transform");
	ImGui::NextColumn();
	ImGui::SetNextItemWidth(-1);
	ImGui::Text("To World");
	ImGui::NextColumn();
	if (node_open)
	{
		touched |= m_cameraToWorld.getImGuiNodes();
		ImGui::TreePop();
	}
	ImGui::PopID();

	ImGui::AlignTextToFramePadding();
	ImGui::PushID(1);
	ImGui::TreeNodeEx("fov", ImGuiLeafNodeFlags, "Field Of View");
	ImGui::NextColumn();
	ImGui::SetNextItemWidth(-1);
	touched |= ImGui::DragFloat("##value", &m_fov, 1, 0, 360, "%.3f", ImGuiSliderFlags_AlwaysClamp);
	ImGui::NextColumn();
	ImGui::PopID();

	ImGui::AlignTextToFramePadding();
	ImGui::PushID(2);
	ImGui::TreeNodeEx("nearCLip", ImGuiLeafNodeFlags, "Near Clip");
	ImGui::NextColumn();
	ImGui::SetNextItemWidth(-1);
	touched |= ImGui::DragFloat("##value", &m_nearClip, 1, 0, SLIDER_MAX_FLOAT, "%.3f",
	                            ImGuiSliderFlags_AlwaysClamp);
	ImGui::NextColumn();
	ImGui::PopID();

	ImGui::AlignTextToFramePadding();
	ImGui::PushID(3);
	ImGui::TreeNodeEx("fov", ImGuiLeafNodeFlags, "Far Clip");
	ImGui::NextColumn();
	ImGui::SetNextItemWidth(-1);
	touched |= ImGui::DragFloat("##value", &m_farClip, 1, 0, SLIDER_MAX_FLOAT, "%.3f",
	                            ImGuiSliderFlags_AlwaysClamp);
	ImGui::NextColumn();
	ImGui::PopID();

	ImGui::AlignTextToFramePadding();
	ImGui::PushID(4);
	ImGui::TreeNodeEx("focalDistance", ImGuiLeafNodeFlags, "Focus Distance (m)");
	ImGui::SameLine();
	ImGui::HelpMarker("Not the same as the focal length of the lens.");
	ImGui::NextColumn();
	ImGui::SetNextItemWidth(-1);
	touched |= ImGui::DragFloat("##value", &m_focalDistance, 0.01f, 0, SLIDER_MAX_FLOAT, "%.3f",
	                            ImGuiSliderFlags_AlwaysClamp);
	ImGui::NextColumn();
	ImGui::PopID();

	ImGui::AlignTextToFramePadding();
	ImGui::PushID(5);
	ImGui::TreeNodeEx("fstop", ImGuiLeafNodeFlags, "F-Stop");
	ImGui::NextColumn();
	ImGui::SetNextItemWidth(-1);
	touched |= ImGui::DragFloat("##value", &m_fstop, 0.001f, 0, SLIDER_MAX_FLOAT,
	                            "%.3f", ImGuiSliderFlags_AlwaysClamp);
	ImGui::NextColumn();

	if (touched)
		m_lensRadius = m_focalDistance / m_fstop;

	ImGui::AlignTextToFramePadding();
	ImGui::PushID(6);
	ImGui::TreeNodeEx("lensRadius", ImGuiLeafNodeFlags, "Lens Radius");
	ImGui::NextColumn();
	ImGui::SetNextItemWidth(-1);
	const bool touchedLensRadius = ImGui::DragFloat("##value", &m_lensRadius, 0.001f, 0, SLIDER_MAX_FLOAT,
	                                                "%.3f", ImGuiSliderFlags_AlwaysClamp);
	ImGui::NextColumn();
	ImGui::PopID();

	touched |= touchedLensRadius;
	if (touchedLensRadius)
		m_fstop                  = m_focalDistance / m_lensRadius;

	return touched;
}
#endif

NORI_REGISTER_CLASS(PerspectiveCamera, "perspective");
NORI_NAMESPACE_END