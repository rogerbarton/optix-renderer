#include <nori/DebugGui.h>

#include <nori/block.h>
#include <nori/parser.h>
#include <nori/bitmap.h>

/*#include <imgui/imconfig.h>
#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
*/
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

DebugGui::DebugGui(ImageBlock &block) : m_block(block), m_renderThread(m_block)
{
	width = block.cols();
	height = block.rows();
	initGlfw("ENori - Enhanced Nori", width, height);
	initGl();
	initImGui();
}

void DebugGui::newFrame()
{
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

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

void DebugGui::endFrame()
{
	if (uiShowDebugWindow)
	{
		if (ImGui::Begin("Hello There", &uiShowDebugWindow))
		{
			// if (ImGui::Button("Reset Camera"))
			// 	scene.resetCamera();

			// uint32_t max = 1000u;
			// ImGui::DragInt("Samples per Launch", reinterpret_cast<int*>(&params.samplesPerLaunch), 1.f, 1, max);
			//
			// ImGui::DragInt("Max Launches", reinterpret_cast<int*>(&state.guiParams.maxLaunches), 1.f, 1, max);
			// ImGui::ColorEdit3("Background", reinterpret_cast<float*>(&state.params.bgColor));

			if (ImGui::CollapsingHeader("Stats", ImGuiTreeNodeFlags_DefaultOpen))
			{
				// ImGui::DragInt("Subframes", reinterpret_cast<int*>(&state.params.subframeIndex), 1.f, 1, 1000,
				//                "%d", ImGuiSliderFlags_ReadOnly | ImGuiSliderFlags_NoInput);
				// displayStats(stateUpdateTime, renderTime, displayTime);
			}
		}
		ImGui::End();
	}

	ImGui::Render();
	int framebufResx, framebufResy;
	glfwGetFramebufferSize(glfwWindow, &framebufResx, &framebufResy);
	glViewport(0, 0, framebufResx, framebufResy);
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	ImGuiIO &imGuiIo = ImGui::GetIO();
	if (imGuiIo.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
	{
		ImGui::UpdatePlatformWindows();
		ImGui::RenderPlatformWindowsDefault();
		glfwMakeContextCurrent(glfwWindow);
	}
}

static void errorCallback(int error, const char *description)
{
	std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

// -- GLFW window callbacks
static void keyCallbackStub(GLFWwindow *window, int32_t key, int32_t scancode, int32_t action, int32_t mods)
{
	static_cast<DebugGui *>(glfwGetWindowUserPointer(window))->keyCallback(key, scancode, action, mods);
}

void DebugGui::keyCallback(int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{
	if (action == GLFW_PRESS)
	{
		if (key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE)
			glfwSetWindowShouldClose(glfwWindow, true);
	}
}
// -- End of GLFW window callbacks

void DebugGui::initGlfw(const char *windowTitle, int width, int height)
{
	glfwWindow = nullptr;
	glfwSetErrorCallback(errorCallback);
	if (!glfwInit())
		throw NoriException("Failed to initialize GLFW.");

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
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

void DebugGui::initGl()
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
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        throw std::runtime_error("Failed to initialize GLAD");
    }
}

void DebugGui::initImGui()
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