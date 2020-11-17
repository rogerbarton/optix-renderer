#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <imgui/filebrowser.h>

#include <nori/common.h>
#include <nori/render.h>
#include <nori/glutil.h>
#include <map>

NORI_NAMESPACE_BEGIN

class MouseState
{
public:
	// keep track of the last mouse position
	double lastMouseX = 0, lastMouseY = 0;
	double mouseMoveX = 0, mouseMoveY = 0;

	bool rButtonPressed = false, lButtonPressed = false, mButtonPressed = false;
	bool dragging = false;

	int mods = 0;

public:
	MouseState() {}
	~MouseState() {}
	void onMouseClick(double xPos, double yPos, int button, int action,
					  int mods)
	{
		this->mods = mods;

		lastMouseX = xPos;
		lastMouseY = yPos;

		dragging = (action == GLFW_PRESS);

		if (button == GLFW_MOUSE_BUTTON_LEFT)
			lButtonPressed = (action != GLFW_RELEASE);
		if (button == GLFW_MOUSE_BUTTON_MIDDLE)
			mButtonPressed = (action != GLFW_RELEASE);
		if (button == GLFW_MOUSE_BUTTON_RIGHT)
			rButtonPressed = (action != GLFW_RELEASE);
	}
	void onMouseMove(double xPos, double yPos)
	{
		mouseMoveX = lastMouseX - xPos;
		mouseMoveY = -lastMouseY + yPos;
		lastMouseX = xPos;
		lastMouseY = yPos;
	}
};

typedef std::map<int, bool> KeyboardState;

inline float get_pixel_ratio();

class ImguiScreen
{
public:
	ImguiScreen(ImageBlock &block);
	~ImguiScreen();

	void initGlfw(const char *windowTitle, int width, int height);
	void initGl();
	void initImGui();
	void setCallbacks();

	void resizeWindow(int width, int height);

	void drawAll();

	void draw();

	void mainloop();

	void render();

	void keyPressed(int key, int mods);
	void keyReleased(int key, int mods);

	// -- Scene loading
	void openXML(const std::string &filename);
	void openEXR(const std::string &filename);

private:
	// -- Window state, this must be public for the main.cpp file
	GLFWwindow *glfwWindow;
	bool uiShowSceneWindow = true;

	bool _minimizedWindow = false; // is this actually needed?

	int width;
	int height;

	// -- Input State
	int32_t _mouseButton = -1;

	// -- Render State
	ImageBlock &m_block;
	RenderThread m_renderThread;

	float clearColor[3] = {0.8f, 0.8f, 0.8f};

	KeyboardState keyboardState;
	MouseState mouseState;

	ImGui::FileBrowser filebrowser;
	ImGui::FileBrowser filebrowserSave = ImGui::FileBrowser(ImGuiFileBrowserFlags_::ImGuiFileBrowserFlags_EnterNewFilename);

	uint32_t m_texture = 0;
	float m_scale = 1.f;
	GLShader *m_shader;

	/**
	 * Draws the editable scene tree with imgui
	 */
	void drawSceneTree();
};

NORI_NAMESPACE_END
