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

	filebrowser.SetTitle("Open File");
	filebrowser.SetTypeFilters({".xml", ".exr"});
}

void ImguiScreen::openXML(const std::string& filename) {
	// TODO
}

void ImguiScreen::openEXR(const std::string &filename)
{
	if (m_renderThread.isBusy())
	{
		cerr << "Error: rendering in progress, you need to wait until it's done" << endl;
		return;
	}

	Bitmap bitmap(filename);

	m_block.lock();
	m_block.init(Vector2i(bitmap.cols(), bitmap.rows()), nullptr);
	m_block.fromBitmap(bitmap);
	Vector2i bsize = m_block.getSize();
	m_block.unlock();

	renderImage = true;
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
		// Potentially needed when displaying the image
		// int framebufResx, framebufResy;
		// glfwGetFramebufferSize(glfwWindow, &framebufResx, &framebufResy);
		// glViewport(0, 0, framebufResx, framebufResy);
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
		{
			ImGui::UpdatePlatformWindows();
			ImGui::RenderPlatformWindowsDefault();
			glfwMakeContextCurrent(glfwWindow);
		}
	}
}

void ImguiScreen::render()
{
	// draws the tonemapped image to screen
	if(renderImage) {
		std::cout << "Render imge" << std::endl;
	}
}

void ImguiScreen::draw()
{
	// Enable docking in the main window, do not clear it so we can see the image behind
	ImGui::DockSpaceOverViewport(NULL, ImGuiDockNodeFlags_PassthruCentralNode |
	                                   ImGuiDockNodeFlags_NoDockingInCentralNode);

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
				filebrowser.Open();
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

	// handle filedialog
	filebrowser.Display();

	if (filebrowser.HasSelected())
	{
		std::string extension = filebrowser.GetSelected().extension();
		if(extension == ".xml") {
			openXML(filebrowser.GetSelected().string());
		} else if(extension == ".exr") {
			openEXR(filebrowser.GetSelected().string());
		}
 		filebrowser.ClearSelected();
	}
}

static void errorCallback(int error, const char *description)
{
	std::cerr << "GLFW Error " << error << ": " << description << std::endl;
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
	if (key == GLFW_KEY_D && mods == GLFW_MOD_ALT)
		uiShowDemoWindow = !uiShowDemoWindow;
}

void ImguiScreen::keyReleased(int key, int mods)
{
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
	ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

	ImGui::StyleColorsDark();

	ImGuiStyle &imGuiStyle = ImGui::GetStyle();

	ImGui_ImplGlfw_InitForOpenGL(glfwWindow, true);
	const char *glsl_version = "#version 150";
	ImGui_ImplOpenGL3_Init(glsl_version);

	//    imGuiIo.Fonts->AddFontFromFileTTF("imgui/misc/fonts/Roboto-Medium.ttf", 16.f);
}

NORI_NAMESPACE_END