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
#include <map>

NORI_NAMESPACE_BEGIN

/*class MouseState {
public:
    // keep track of the last mouse position
    double lastMouseX = 0, lastMouseY = 0;
    double mouseMoveX = 0, mouseMoveY = 0;

    bool rButtonPressed = false, lButtonPressed = false, mButtonPressed = false;
    bool dragging = false;

    int mods = 0;

public:
    MouseState();
    ~MouseState();
    void onMouseClick(double xPos, double yPos, int button, int action,
                      int mods);
    void onMouseMove(double xPos, double yPos);
};
*/
typedef std::map<int, bool> KeyboardState;

inline float get_pixel_ratio();

class ImguiScreen
{
public:
	ImguiScreen(ImageBlock &block);
	bool uiShowDemoWindow  = false;
	bool uiShowSceneWindow = true;

	void initGlfw(const char *windowTitle, int width, int height);
	void initGl();
	void initImGui();

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

	KeyboardState keyboardState;

	ImGui::FileBrowser filebrowser;

	bool renderImage = false;
};

NORI_NAMESPACE_END
