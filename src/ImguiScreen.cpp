#include <nori/ImguiScreen.h>

#include <nori/block.h>
#include <nori/parser.h>
#include <nori/bitmap.h>
#include <map>
#include <algorithm>

NORI_NAMESPACE_BEGIN

float get_pixel_ratio()
{
#if RETINA_SCREEN == 1
	return 1.f;
#endif
	GLFWmonitor *monitor = glfwGetPrimaryMonitor();
	if (monitor == nullptr)
		throw "Primary monitor not found.";
	float xscale, yscale;
	glfwGetMonitorContentScale(monitor, &xscale, &yscale);
	return xscale;
}

ImguiScreen::ImguiScreen(ImageBlock &block) : m_block(block), m_renderThread(m_block)
{
	width = block.cols();
	height = block.rows();
	initGlfw("ENori - Enhanced Nori", width, height);
	initGl();
	initImGui();
}

void ImguiScreen::mainloop()
{
	while (!glfwWindowShouldClose(glfwWindow))
	{
		drawAll();
#ifdef SINGLE_BUFFER
		glFlush();
#else
		glfwSwapBuffers(glfwWindow);
#endif
		glfwPollEvents();
	}
	glfwTerminate();
}

void ImguiScreen::drawAll()
{
	glClearColor(clearColor[0], clearColor[1], clearColor[2], 1.00f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT |
			GL_STENCIL_BUFFER_BIT);

	// draw scene here
	render();

	// draw ImGui menu
	{
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		draw();

		ImGui::EndFrame();
		ImGui::UpdatePlatformWindows();
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}
}

void ImguiScreen::render() {
	// draws the tonemapped image to screen
}

void ImguiScreen::draw()
{
	
	if (uiShowDemoWindow)
		ImGui::ShowDemoWindow(&uiShowDemoWindow);

	ImGui::DockSpaceOverViewport(NULL, ImGuiDockNodeFlags_PassthruCentralNode |
										   ImGuiDockNodeFlags_NoDockingInCentralNode);

	if (ImGui::BeginMainMenuBar())
	{
		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::MenuItem("Open...", "Ctrl+O"))
			{
			}
			if (ImGui::MenuItem("Save"))
			{
			}
			if (ImGui::MenuItem("Save As..."))
			{
			}
			if (ImGui::MenuItem("Settings..."))
			{
			}
			ImGui::MenuItem("Gui Demo", "Alt+D", &uiShowDemoWindow);
			ImGui::EndMenu();
		}
		ImGui::MenuItem("Debug", "D", &uiShowDebugWindow);
		ImGui::EndMainMenuBar();
	}
}

static void errorCallback(int error, const char *description)
{
	std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

// -- GLFW window callbacks
static void keyCallbackStub(GLFWwindow *window, int32_t key, int32_t scancode, int32_t action, int32_t mods)
{
	static_cast<ImguiScreen *>(glfwGetWindowUserPointer(window))->keyCallback(key, scancode, action, mods);
}

void ImguiScreen::keyCallback(int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{
	if (action == GLFW_PRESS)
	{
		if (key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE)
			glfwSetWindowShouldClose(glfwWindow, true);
	}
}
// -- End of GLFW window callbacks

void ImguiScreen::initGlfw(const char *windowTitle, int width, int height)
{
	glfwWindow = nullptr;
	glfwSetErrorCallback(errorCallback);
	if (!glfwInit())
		throw NoriException("Failed to initialize GLFW.");

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

#ifdef SINGLE_BUFFER
	glfwWindowHint(GLFW_DOUBLEBUFFER, GL_FALSE); // turn off framerate limit
#endif

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make Apple happy -- should not be needed
#endif
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	glfwWindow = glfwCreateWindow(width, height, windowTitle, nullptr, nullptr);
	if (!glfwWindow)
		throw NoriException("Failed to create GLFW window.");

	glfwMakeContextCurrent(glfwWindow);
	glfwSwapInterval(0); // No vsync

	// -- window Callbacks
	glfwSetWindowUserPointer(glfwWindow, this);
	glfwSetKeyCallback(glfwWindow, keyCallbackStub);
}

void ImguiScreen::initGl()
{
	// #if defined(WIN32)
	// 	static bool glewInitialized = false;
	// 	if (!glewInitialized)
	// 	{
	// 		glewExperimental = GL_TRUE;
	// 		glewInitialized = true;
	// 		if (glewInit() != GLEW_NO_ERROR)
	// 			throw std::runtime_error("Could not initialize GLEW.");
	// 	}
	// GL_CHECK(glClearColor(0.212f, 0.271f, 0.31f, 1.0f));
	// GL_CHECK(glClear(GL_COLOR_BUFFER_BIT));
	// #endif
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		throw std::runtime_error("Failed to initialize GLAD");
	}
}

void ImguiScreen::initImGui()
{
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO &imGuiIo = ImGui::GetIO();
	(void)imGuiIo;
	imGuiIo.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	imGuiIo.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
	imGuiIo.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

	ImGui::StyleColorsDark();

	ImGuiStyle &imGuiStyle = ImGui::GetStyle();
	if (imGuiIo.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
	{
		imGuiStyle.WindowRounding = 0.f;
		imGuiStyle.Colors[ImGuiCol_WindowBg].w = 1.f;
	}

	ImGui_ImplGlfw_InitForOpenGL(glfwWindow, true);
	const char *glsl_version = "#version 150";
	ImGui_ImplOpenGL3_Init(glsl_version);

	//    imGuiIo.Fonts->AddFontFromFileTTF("imgui/misc/fonts/Roboto-Medium.ttf", 16.f);
}

NORI_NAMESPACE_END