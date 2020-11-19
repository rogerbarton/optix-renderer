#include <nori/ImguiScreen.h>

#include <nori/block.h>
#include <nori/parser.h>
#include <nori/bitmap.h>
#include <nori/scene.h>
#include <map>
#include <algorithm>
#include <filesystem/path.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

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
	windowWidth = block.cols();
	windowHeight = block.rows();
	initGlfw("ENori - Enhanced Nori", windowWidth, windowHeight);
	initGl();
	initImGui();
	setCallbacks();

	filebrowser.SetTitle("Open File");
	filebrowser.SetTypeFilters({".xml", ".exr"});
	filebrowser.SetPwd(std::filesystem::relative("../scenes/project"));

	filebrowser.SetTitle("Save as");
	filebrowserSave.SetPwd(std::filesystem::relative("../scenes/project"));

	glGenTextures(1, &m_texture);
	glBindTexture(GL_TEXTURE_2D, m_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	// init shader
	m_shader = new GLShader();
	m_shader->init("Tonemapper", "#version 330\n"
								 "in vec2 position;\n"
								 "out vec2 uv;\n"
								 "void main() {\n"
								 "    gl_Position = vec4(position.x*2-1, position.y*2-1, 0.0, 1.0);\n"
								 "    uv = vec2(position.x, 1-position.y);\n"
								 "}",
				   "#version 330\n"
				   "uniform sampler2D source;\n"
				   "uniform float scale;\n"
				   "in vec2 uv;\n"
				   "out vec4 out_color;\n"
				   "float toSRGB(float value) {\n"
				   "    if (value < 0.0031308)\n"
				   "        return 12.92 * value;\n"
				   "    return 1.055 * pow(value, 0.41666) - 0.055;\n"
				   "}\n"
				   "void main() {\n"
				   "    vec4 color = texture(source, uv);\n"
				   "    color *= scale / color.w;\n"
				   "    out_color = vec4(toSRGB(color.r), toSRGB(color.g), toSRGB(color.b), 1);\n"
				   "}");

	MatrixXu indices(3, 2); /* Draw 2 triangles */
	indices.col(0) << 0, 1, 2;
	indices.col(1) << 2, 3, 0;

	MatrixXf positions(2, 4);
	positions.col(0) << 0, 0;
	positions.col(1) << 1, 0;
	positions.col(2) << 1, 1;
	positions.col(3) << 0, 1;

	m_shader->bind();
	m_shader->uploadIndices(indices);
	m_shader->uploadAttrib("position", positions);
}

void ImguiScreen::drop(const std::string &filename)
{
	filesystem::path path = filesystem::path(filename);

	if (path.extension() == "xml")
	{
		/* Render the XML scene file */
		openXML(filename);
	}
	else if (path.extension() == "exr")
	{
		/* Alternatively, provide a basic OpenEXR image viewer */
		openEXR(filename);
	}
	else
	{
		cerr << "Error: unknown file \"" << filename
			 << "\", expected an extension of type .xml or .exr" << endl;
	}
}

void ImguiScreen::openXML(const std::string &filename)
{
	if (m_renderThread.isBusy())
		m_renderThread.stopRendering();

	try
	{
		renderingFilename = filename;
		m_renderThread.renderScene(filename);

		imageZoom = 1.f;
		centerImage(true);
	}
	catch (const std::exception &e)
	{
		cerr << "Fatal error: " << e.what() << endl;
	}
}

ImguiScreen::~ImguiScreen()
{
	if (m_shader)
		delete m_shader;
}

void ImguiScreen::openEXR(const std::string &filename)
{
	if (m_renderThread.isBusy())
		m_renderThread.stopRendering();

	if (m_renderThread.m_scene)
		delete m_renderThread.m_scene;
	m_renderThread.m_scene = nullptr; // nullify scene for Tree viewer

	Bitmap bitmap(filename);

	m_block.lock();
	m_block.init(Vector2i(bitmap.cols(), bitmap.rows()), nullptr);
	m_block.fromBitmap(bitmap);
	m_block.unlock();

	imageZoom = 1.f;
	centerImage(true);
}

void ImguiScreen::windowResized(int width, int height)
{
	this->windowWidth = width;
	this->windowHeight = height;

#ifdef RETINA_SCREEN
	this->windowWidth /= 2.0;
	this->windowHeight /= 2.0;
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
			glfwMakeContextCurrent(glfwWindow);
		}
	}
}

void ImguiScreen::render()
{
	// draws the tonemapped image to screen
	m_block.lock();
	int borderSize = m_block.getBorderSize();
	const Vector2i &size = m_block.getSize();
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, m_texture);
	glPixelStorei(GL_UNPACK_ROW_LENGTH, m_block.cols());
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, size.x(), size.y(),
				 0, GL_RGBA, GL_FLOAT, (uint8_t *)m_block.data() + (borderSize * m_block.cols() + borderSize) * sizeof(Color4f));
	glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
	m_block.unlock();

	glViewport(imageOffset[0], imageOffset[1], get_pixel_ratio() * size[0] * imageZoom, get_pixel_ratio() * size[1] * imageZoom);
	m_shader->bind();
	m_shader->setUniform("scale", m_scale);
	m_shader->setUniform("source", 0);
	m_shader->drawIndexed(GL_TRIANGLES, 0, 2);
	glViewport(0, 0, windowWidth, windowHeight); // reset viewport
}

void ImguiScreen::draw()
{
	// Enable docking in the main window, do not clear it so we can see the image behind
	ImGui::DockSpaceOverViewport(NULL, ImGuiDockNodeFlags_PassthruCentralNode |
										   ImGuiDockNodeFlags_NoDockingInCentralNode);

	if (ImGui::BeginMainMenuBar())
	{
		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::MenuItem("Open...", "Ctrl+O"))
				filebrowser.Open();
			if (ImGui::MenuItem("Export Image"))
			{
				filebrowserSave.Open();
			}
			if (ImGui::MenuItem("Settings..."))
			{
			}
			ImGui::EndMenu();
		}
		ImGui::MenuItem("Scene", "D", &uiShowSceneWindow);

		if (ImGui::BeginMenu("View"))
		{
			if (ImGui::MenuItem("Center Image", "0"))
				centerImage();
			if (ImGui::MenuItem("Reset Zoom", "1"))
				setZoom(1.f, false);
			if (ImGui::MenuItem("2x Zoom", "2"))
				setZoom(2.f, false);
			ImGui::EndMenu();
		}

		ImGui::EndMainMenuBar();
	}

	// handle filedialog
	filebrowser.Display();
	filebrowserSave.Display();

	if (filebrowser.HasSelected())
	{
		std::string extension = filebrowser.GetSelected().extension().string();
		if (extension == ".xml")
		{
			openXML(filebrowser.GetSelected().string());
		}
		else if (extension == ".exr")
		{
			openEXR(filebrowser.GetSelected().string());
		}
		filebrowser.ClearSelected();
	}

	if (filebrowserSave.HasSelected())
	{
		std::string extension = filebrowserSave.GetSelected().extension().string();
		if (extension == ".png")
		{
			// write tonemapped png
			Bitmap *bm = m_block.toBitmap();
			bm->saveToLDR(filebrowserSave.GetSelected().string());
			std::cout << "PNG file saved to " << filebrowserSave.GetSelected().string() << std::endl;
		}
		else if (extension == ".exr")
		{
			// write exr
			Bitmap *bm = m_block.toBitmap();
			bm->save(filebrowserSave.GetSelected().string());
			std::cout << "EXR file saved to " << filebrowserSave.GetSelected().string() << std::endl;
		}
		filebrowserSave.ClearSelected();
	}

	if (uiShowSceneWindow)
	{
		ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
		if (ImGui::Begin("Scene Controller", &uiShowSceneWindow))
		{
			ImGui::Text("Render");
			ImGui::SameLine();
			ImGui::ProgressBar(m_renderThread.getProgress());

			if (ImGui::Button("Stop Render"))
				m_renderThread.stopRendering();

			// show restart button if m_scene is valid
			if (m_renderThread.m_scene && ImGui::Button("Restart Render"))
				m_renderThread.rerenderScene(renderingFilename);

			if (ImGui::Button("Reset Camera"))
			{
			}

			static float exposureLog = 0.5f;
			ImGui::SliderFloat("Exposure", &exposureLog, 0.01f, 1.f);
			ImGui::SameLine();
			if (ImGui::Button("Reset"))
				exposureLog = 0.5f;
			m_scale = std::pow(2.f, (exposureLog - 0.5f) * 20);

			if (ImGui::CollapsingHeader("Scene Tree", ImGuiTreeNodeFlags_DefaultOpen))
			{
				ImGui::BeginChild(42);
				drawSceneTree();
				ImGui::EndChild();
			}
		}
		ImGui::End();
	}
}

void ImguiScreen::setCallbacks()
{
	glfwSetKeyCallback(glfwWindow, [](GLFWwindow *window, int key, int scancode,
									  int action, int mods) {
		auto app = static_cast<ImguiScreen *>(glfwGetWindowUserPointer(window));

		if (ImGui::GetIO().WantCaptureKeyboard ||
			ImGui::GetIO().WantTextInput)
		{
			ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
			return;
		}

		app->keyboardState[key] = (action != GLFW_RELEASE);

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

	glfwSetMouseButtonCallback(glfwWindow, [](GLFWwindow *window, int button,
											  int action, int mods) {
		double xPos, yPos;
		glfwGetCursorPos(window, &xPos, &yPos);
#if RETINA_SCREEN == 1
		xPos *= 2;
		yPos *= 2;
#endif
		auto app = static_cast<ImguiScreen *>(glfwGetWindowUserPointer(window));
		app->mouseState.onMouseClick(xPos, yPos, button, action, mods);

		if (ImGui::GetIO().WantCaptureMouse)
		{
			ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
			return;
		}

		if (action == GLFW_PRESS)
			app->mouseButtonPressed(button, mods);

		if (action == GLFW_RELEASE)
			app->mouseButtonReleased(button, mods);
	});

	glfwSetCursorPosCallback(glfwWindow, [](GLFWwindow *window, double xpos,
											double ypos) {
		auto app = static_cast<ImguiScreen *>(glfwGetWindowUserPointer(window));
#if RETINA_SCREEN == 1
		xpos *= 2;
		ypos *= 2;
#endif
		app->mouseState.onMouseMove(xpos, ypos);

		if (ImGui::GetIO().WantCaptureMouse)
			return;

		app->mouseMove(xpos, ypos);
	});

	glfwSetScrollCallback(glfwWindow, [](GLFWwindow *window, double xoffset,
										 double yoffset) {
		if (ImGui::GetIO().WantCaptureMouse)
		{
			ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
			return;
		}

		auto app = static_cast<ImguiScreen *>(glfwGetWindowUserPointer(window));
		app->scrollWheel(xoffset, yoffset);
	});

	glfwSetDropCallback(glfwWindow, [](GLFWwindow *window, int count,
									   const char **filenames) {
		auto app = static_cast<ImguiScreen *>(glfwGetWindowUserPointer(window));
		std::string file(filenames[0]);
		app->drop(file);
	});
}

static void errorCallback(int error, const char *description)
{
	std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

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

	glfwSetFramebufferSizeCallback(glfwWindow, [](GLFWwindow *window, int width,
												  int height) {
		auto app = static_cast<ImguiScreen *>(glfwGetWindowUserPointer(window));
		app->windowResized(width, height);
	});

	GLFWimage image;
	image.pixels = stbi_load(std::filesystem::relative("../logo.png").string().c_str(), &image.width, &image.height, nullptr, 4);
	glfwSetWindowIcon(glfwWindow, 1, &image);
	stbi_image_free(image.pixels);
}

void ImguiScreen::keyPressed(int key, int mods)
{
	if (key == GLFW_KEY_O && mods & GLFW_MOD_CONTROL)
		filebrowser.Open();
	else if (key == GLFW_KEY_D)
	{
		uiShowSceneWindow = !uiShowSceneWindow;
	}
	else if (key == GLFW_KEY_Z && GLFW_MOD_CONTROL)
		m_renderThread.stopRendering();
	else if (key == GLFW_KEY_E && mods & GLFW_MOD_CONTROL)
	{
		filebrowserSave.Open();
	}
	else if (key == GLFW_KEY_0)
		centerImage();
	else if (key == GLFW_KEY_1)
		setZoom(1.f);
	else if (key == GLFW_KEY_2)
		setZoom(2.f);
}

void ImguiScreen::keyReleased(int key, int mods)
{
}

void ImguiScreen::mouseButtonPressed(int button, int mods) {}
void ImguiScreen::mouseButtonReleased(int button, int mods) {}
void ImguiScreen::mouseMove(double xpos, double ypos)
{
	if (mouseState.dragging)
	{
		imageOffset(0) -= (int)mouseState.mouseMoveX;
		imageOffset(1) -= (int)mouseState.mouseMoveY;
	}
}

void ImguiScreen::setZoom(float value, bool centerOnMouse)
{

	Vector2f zoomCenter{windowWidth / 2.f, windowHeight / 2.f};
	if (centerOnMouse)
	{
		zoomCenter.x() = mouseState.lastMouseX;
		zoomCenter.y() = windowHeight - mouseState.lastMouseY;
	}

	float scale = imageZoom / value;
	imageOffset(0) = (imageOffset(0) - zoomCenter.x()) / scale + zoomCenter.x();
	imageOffset(1) = (imageOffset(1) - zoomCenter.y()) / scale + zoomCenter.y();

	imageZoom = value;
}

void ImguiScreen::centerImage(bool autoExpandWindow)
{
	m_block.lock();
	const Vector2i bsize = m_block.getSize();
	m_block.unlock();

	if (autoExpandWindow && (windowWidth < bsize.x() || windowHeight < bsize.y()))
	{
		// bit hacky but it works, should have a resize function
		glfwSetWindowSize(glfwWindow, std::max(windowWidth, bsize.x()), std::max(windowHeight, bsize.y()));
		windowResized(bsize.x(), bsize.y());
	}

	imageOffset(0) = (windowWidth - imageZoom * bsize(0)) / 2;
	imageOffset(1) = (windowHeight - imageZoom * bsize(1)) / 2;
}

void ImguiScreen::scrollWheel(double xoffset, double yoffset)
{
	float scale = 1.f - 0.05f * yoffset;
	setZoom(imageZoom / scale);
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

void ImguiScreen::drawSceneTree()
{
	// check if a scene exists
	if (!m_renderThread.m_scene)
	{
		ImGui::Text("No scene loaded...");
		return;
	}

	bool renderThreadBusy = m_renderThread.isBusy();

	if (renderThreadBusy)
	{
		ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
		ImGui::PushStyleVar(ImGuiStyleVar_Alpha,
							ImGui::GetStyle().Alpha * 0.5f);
	}

	// Start columns
	ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 2));
	ImGui::Columns(2);

	ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen |
							   ImGuiTreeNodeFlags_Bullet;

	ImGui::AlignTextToFramePadding();
	ImGui::TreeNodeEx("fileName", flags, "Filename");
	ImGui::NextColumn();
	ImGui::SetNextItemWidth(-1);
	ImGui::Text(filesystem::path(renderingFilename).filename().c_str());
	ImGui::NextColumn();

	// Start recursion
	m_renderThread.m_scene->getImGuiNodes();

	// end columns
	ImGui::Columns(1);
	ImGui::Separator();
	ImGui::PopStyleVar();

	// pop disable flags
	if (renderThreadBusy)
	{
		ImGui::PopItemFlag();
		ImGui::PopStyleVar();
	}
}

NORI_NAMESPACE_END