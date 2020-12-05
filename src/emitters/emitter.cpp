#include <nori/emitter.h>

NORI_NAMESPACE_BEGIN

	void Emitter::cloneAndInit(Emitter *clone)
	{
		// Shape already cloned by scene
		// if(m_shape)
		// 	clone->m_shape = static_cast<Shape *>(m_shape->cloneAndInit());
	}

	void Emitter::update(const NoriObject *guiObject)
	{
		const auto *gui = static_cast<const Emitter *>(guiObject);
		m_position = gui->m_position;
		// Shape already updated by scene
		// if(m_shape)
		// 	m_shape->update(gui->m_shape);
	}

#ifdef NORI_USE_IMGUI
	bool Emitter::getImGuiNodes()
	{
		ImGui::PushID(EEmitter);

		ImGui::AlignTextToFramePadding();
		ImGui::TreeNodeEx("Position", ImGuiLeafNodeFlags, "Position");
		ImGui::NextColumn();
		ImGui::SetNextItemWidth(-1);
		touched |= ImGui::DragPoint3f("##value", &m_position, 0.1f);
		ImGui::NextColumn();
		ImGui::PopID();

		// Also set shape dirty in case this is called by the scene directly
		if (m_shape)
			m_shape->touched |= touched;
		return touched;
	}
#endif

NORI_NAMESPACE_END