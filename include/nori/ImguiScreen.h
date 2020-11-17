#pragma once
//#include <GL/glew.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <nori/common.h>
#include <nori/render.h>

NORI_NAMESPACE_BEGIN

/**
 * Glfw and dear gui window, based on Nvidia OptiX samples so far.
 * It creates a glfw window, registers callbacks and sets up imgui.
 * WIP!
 */

inline float get_pixel_ratio();

class ImguiScreen
{
public:
	ImguiScreen(ImageBlock &block);
	bool uiShowDemoWindow = false;
	bool uiShowDebugWindow = true;

	void initGlfw(const char *windowTitle, int width, int height);
	void initGl();
	void initImGui();

	void drawAll();

	void draw();

	void mainloop();

	void render();

	// -- Scene loading
	void openXML(const std::string &filename) {}
	void openEXR(const std::string &filename) {}

	// -- GLFW window callbacks
	void mouseButtonCallback(int button, int action, int mods);
	void cursorPosCallback(double xpos, double ypos);
	void windowSizeCallback(int32_t xres, int32_t yres);
	void windowIconifyCallback(int32_t iconified);
	void keyCallback(int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/);
	void scrollCallback(double xscroll, double yscroll);

private:
	// -- Window state, this must be public for the main.cpp file
	GLFWwindow *glfwWindow;

	bool _minimizedWindow = false;

	int width;
	int height;

	// -- Input State
	int32_t _mouseButton = -1;

	// -- Render State
	ImageBlock &m_block;
	//unsigned int &_imageWidth;
	//unsigned int &_imageHeight;
	RenderThread m_renderThread;

	float clearColor[3] = {0.8f, 0.8f, 0.8f};
};

NORI_NAMESPACE_END
