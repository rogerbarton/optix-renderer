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

void ImguiScreen::resizeWindow(int width, int height)
{
	this->width = width;
	this->height = height;

#ifdef RETINA_SCREEN
	this->width /= 2.0;
	this->height /= 2.0;
#endif

	glViewport(0, 0, width, height);
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

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());


		if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
		{
			ImGui::UpdatePlatformWindows();
			ImGui::RenderPlatformWindowsDefault();
		}

		
	}
}

void ImguiScreen::render()
{
	// draws the tonemapped image to screen
}

void ImguiScreen::draw()
{

	if (uiShowDemoWindow)
	{
		ImGui::ShowDemoWindow(&uiShowDemoWindow);

		//ImGui::ShowUserGuide();
	}

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

	glfwSetKeyCallback(glfwWindow, [](GLFWwindow *window, int key, int scancode,
									  int action, int mods) {
		auto app = static_cast<ImguiScreen *>(glfwGetWindowUserPointer(window));
		app->keyboardState[key] = (action != GLFW_RELEASE);

		if (ImGui::GetIO().WantCaptureKeyboard ||
			ImGui::GetIO().WantTextInput)
		{
			ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
			return;
		}

		if (key == GLFW_KEY_ESCAPE)
		{
			glfwSetWindowShouldClose(window, GL_TRUE);
			return;
		}

		if (action == GLFW_PRESS)
			app->keyPressed(key, mods);
		if (action == GLFW_RELEASE)
			app->keyReleased(key, mods);
	});

	glfwSetFramebufferSizeCallback(glfwWindow, [](GLFWwindow *window, int width,
												  int height) {
		auto app = static_cast<ImguiScreen *>(glfwGetWindowUserPointer(window));
		app->resizeWindow(width, height);
	});
}

void ImguiScreen::keyPressed(int key, int mods)
{
	std::cout << "Key pressed" << std::endl;
}

void ImguiScreen::keyReleased(int key, int mods)
{
	std::cout << "Key released" << std::endl;
}

void ImguiScreen::initGl()
{
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		throw std::runtime_error("Failed to initialize GLAD");
	}

	glEnable(GL_MULTISAMPLE);
}

void ImguiScreen::initImGui()
{
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO &imGuiIo = ImGui::GetIO();

	ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
	// ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // do not enable viewports for now

	ImGui::StyleColorsDark();

	ImGuiStyle &imGuiStyle = ImGui::GetStyle();

	ImGui_ImplGlfw_InitForOpenGL(glfwWindow, true);
	const char *glsl_version = "#version 150";
	ImGui_ImplOpenGL3_Init(glsl_version);

	//    imGuiIo.Fonts->AddFontFromFileTTF("imgui/misc/fonts/Roboto-Medium.ttf", 16.f);
}

NORI_NAMESPACE_END