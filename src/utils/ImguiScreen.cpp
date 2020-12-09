#include <nori/ImguiScreen.h>

#include <nori/block.h>
#include <nori/parser.h>
#include <nori/bitmap.h>
#include <nori/scene.h>
#include <nori/sampler.h>
#include <map>
#include <algorithm>
#include <filesystem/path.h>
#include <filesystem>

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

ImguiScreen::ImguiScreen() : m_renderThread{}
{
	windowWidth = 1000;
	windowHeight = 800;
	initGlfw("ENori - Enhanced Nori", windowWidth, windowHeight);
	initGl();
	initImGui();
	setCallbacks();

	m_renderThread.initBlocks();

	filebrowser.SetTitle("Open File");
	filebrowser.SetTypeFilters({".xml", ".exr"});
	filebrowser.SetPwd(std::filesystem::relative("../scenes/project"));

	filebrowser.SetTitle("Save as");
	filebrowserSave.SetPwd(std::filesystem::relative("../scenes/project"));

	GL_CHECK(glGenTextures(1, &m_texture));
	GL_CHECK(glBindTexture(GL_TEXTURE_2D, m_texture));
	GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
	GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
	GL_CHECK(glGenTextures(1, &m_textureGpu));
	GL_CHECK(glBindTexture(GL_TEXTURE_2D, m_textureGpu));
	GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
	GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));

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
				   "uniform sampler2D sourceCpu;\n"
				   "uniform sampler2D sourceGpu;\n"
				   "uniform float scale;\n"
				   "uniform float samplesCpu;\n"
				   "uniform float samplesGpu;\n"
				   "in vec2 uv;\n"
				   "out vec4 out_color;\n"
				   "float toSRGB(float value) {\n"
				   "    if (value < 0.0031308)\n"
				   "        return 12.92 * value;\n"
				   "    return 1.055 * pow(value, 0.41666) - 0.055;\n"
				   "}\n"
				   "void main() {\n"
				   "    vec4 color = samplesCpu * texture(sourceCpu, uv) + samplesGpu * texture(sourceGpu, uv);\n"
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

	fpsTimer.reset();
}

void ImguiScreen::drop(const std::string &filename)
{
	if(filename.empty()) return;

	std::filesystem::path path(filename);

	if (path.extension() == ".xml")
	{
		/* Render the XML scene file */
		openXML(filename);
	}
	else if (path.extension() == ".exr")
	{
		/* Alternatively, provide a basic OpenEXR image viewer */
		openEXR(filename);

		// auto tone map
		ImageBlock& compositeBlock = m_renderThread.getBlock();
		compositeBlock.lock();
		Bitmap *bm = compositeBlock.toBitmap();
		compositeBlock.unlock();

		const std::string outfile = std::filesystem::path(path).replace_extension(".png").string();
		bm->saveToLDR(outfile);
		std::cout << "PNG file saved to " << outfile << std::endl;
	}
	else
	{
		cerr << "Error: unknown file \"" << filename
			 << "\", expected an extension of type .xml or .exr" << endl;
	}
}

void ImguiScreen::openXML(const std::string &filename)
{
	try
	{
		m_renderThread.loadScene(filename);

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
	delete m_shader;
}

void ImguiScreen::openEXR(const std::string &filename)
{
	if (m_renderThread.isBusy())
		m_renderThread.stopRendering();

	if (m_renderThread.m_guiScene)
	{
		delete m_renderThread.m_guiScene;
		m_renderThread.m_guiScene = nullptr; // nullify scene for Tree viewer
	}

	Bitmap bitmap(filename);

	ImageBlock& compositeBlock = m_renderThread.getBlock();
	compositeBlock.lock();
	compositeBlock.init(Vector2i(bitmap.cols(), bitmap.rows()), nullptr);
	compositeBlock.fromBitmap(bitmap);
	compositeBlock.unlock();
	m_renderThread.m_visibleRenderLayer = ERenderLayer::Composite;

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

	GL_CHECK(glViewport(0, 0, width, height));
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

		// wait until we reach the FPS limit
		while (fpsTimer.elapsed() < (1.0f / targetFramerate * 1000.f)){}

		fpsTimer.reset();
	}
	glfwTerminate();
}

void ImguiScreen::drawAll()
{
	GL_CHECK(glClearColor(clearColor[0], clearColor[1], clearColor[2], 1.00f));
	GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT));

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
	ImageBlock& block = m_renderThread.getCurrentBlock();
	block.lock();
	int borderSize = block.getBorderSize();
	const Vector2i &size = block.getSize();

	// cpu -> tex0
	{
		GL_CHECK(glActiveTexture(GL_TEXTURE0));
		GL_CHECK(glBindTexture(GL_TEXTURE_2D, m_texture));
		GL_CHECK(glPixelStorei(GL_UNPACK_ROW_LENGTH, block.cols()));
		GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, size.x(), size.y(),
		             0, GL_RGBA, GL_FLOAT,
		             (uint8_t *) block.data() + (borderSize * block.cols() + borderSize) * sizeof(Color4f)));
		GL_CHECK(glPixelStorei(GL_UNPACK_ROW_LENGTH, 0));
	}

	//gpu image -> tex1
#ifdef NORI_USE_OPTIX
	{
		GL_CHECK(glActiveTexture(GL_TEXTURE1));
		GL_CHECK(glBindTexture(GL_TEXTURE_2D, m_textureGpu));
		GL_CHECK(glPixelStorei(GL_UNPACK_ROW_LENGTH, block.cols()));
		GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_renderThread.m_optixBlock->getPBO()));
		GL_CHECK(glPixelStorei(GL_UNPACK_ALIGNMENT, 4));
		GL_CHECK(glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, size.x(), size.y(), 0, GL_RGBA, GL_FLOAT, nullptr ));
		GL_CHECK(glPixelStorei(GL_UNPACK_ROW_LENGTH, 0));
	}
#endif

	block.unlock();

	GL_CHECK(glViewport(imageOffset[0], imageOffset[1], get_pixel_ratio() * size[0] * imageZoom, get_pixel_ratio() * size[1] * imageZoom));
	m_shader->bind();
	m_shader->setUniform("scale", m_scale);
	m_shader->setUniform("sourceCpu", static_cast<int>(m_texture));
	// m_shader->setUniform("sourceGpu", static_cast<int>(m_textureGpu));
	m_shader->setUniform("sourceGpu", static_cast<int>(m_texture));// TODO: DEBUGGGGGG
#ifdef NORI_USE_OPTIX
	m_shader->setUniform("samplesCpu", 0.5f); // TODO: cpu / cpu+gpu
	m_shader->setUniform("samplesGpu", 0.5f);
#else
	m_shader->setUniform("samplesCpu", 1.f);
	m_shader->setUniform("samplesGpu", 0.f);
#endif
	m_shader->drawIndexed(GL_TRIANGLES, 0, 2);
	GL_CHECK(glViewport(0, 0, windowWidth, windowHeight)); // reset viewport
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
			openXML(filebrowser.GetSelected().string());
		else if (extension == ".exr")
			openEXR(filebrowser.GetSelected().string());
		filebrowser.ClearSelected();
	}

	if (filebrowserSave.HasSelected())
	{
		std::string extension = filebrowserSave.GetSelected().extension().string();
		if (extension == ".png")
		{
			// write tonemapped png
			Bitmap *bm = m_renderThread.getBlock().toBitmap();
			bm->saveToLDR(filebrowserSave.GetSelected().string());
			std::cout << "PNG file saved to " << filebrowserSave.GetSelected().string() << std::endl;
		}
		else if (extension == ".exr")
		{
			// write exr
			Bitmap *bm = m_renderThread.getBlock().toBitmap();
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
			ImGui::Text("Render time: %s", (char*)m_renderThread.getRenderTime().c_str());

			if (ImGui::Button("Stop Render"))
				m_renderThread.stopRendering();

			// show restart button if m_scene is valid
			if (m_renderThread.m_guiScene)
			{
				ImGui::SameLine();
				if (ImGui::Button("Restart Render"))
					m_renderThread.restartRender();
			}

			m_renderThread.drawRenderGui();

			ImGui::NewLine();

			ImGui::Text("Image Offset");
			ImGui::SameLine();
			ImGui::PushID(1);
			ImGui::DragVector2i("##value", &imageOffset);
			ImGui::PopID();

			ImGui::Text("Zoom Level");
			ImGui::SameLine();
			ImGui::PushID(2);
			ImGui::SliderFloat("##value", &imageZoom, 1.f / 30.f, 30.f, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
			ImGui::PopID();

			ImGui::NewLine();

			ImGui::Text("Target Framerate");
			ImGui::SameLine();
			ImGui::PushID(3);
			ImGui::SliderFloat("##value", &targetFramerate, 1.f, 100.f, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
			ImGui::PopID();
			ImGui::NewLine();

			static float exposureLog = 0.5f;
			ImGui::SliderFloat("Exposure", &exposureLog, 0.01f, 1.f);
			ImGui::SameLine();
			if (ImGui::Button("Reset"))
				exposureLog = 0.5f;
			m_scale = std::pow(2.f, (exposureLog - 0.5f) * 20);

			if (ImGui::CollapsingHeader("Scene Tree", ImGuiTreeNodeFlags_DefaultOpen))
			{
				ImGui::BeginChild(42);
				m_renderThread.drawSceneGui();
				if (!m_renderThread.m_guiScene && ImGui::Button("Open Scene"))
					filebrowser.Open();

				ImGui::EndChild();
			}
		}
		ImGui::End();
	}

	if (uiShowDemoWindow)
		ImGui::ShowDemoWindow(&uiShowDemoWindow);
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

		if (key == GLFW_KEY_Q)
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
	if (key == GLFW_KEY_O || key == GLFW_KEY_GRAVE_ACCENT)
		filebrowser.Open();
	else if (key == GLFW_KEY_D)
		uiShowSceneWindow = !uiShowSceneWindow;
	else if (key == GLFW_KEY_ESCAPE)
	{
		if (filebrowser.IsOpened())
			filebrowser.Close();
		else if (filebrowserSave.IsOpened())
			filebrowserSave.Close();
		else
			m_renderThread.stopRendering();
	}
	else if (key == GLFW_KEY_F5)
		m_renderThread.restartRender();
	else if (key == GLFW_KEY_E)
		filebrowserSave.Open();
	else if (key == GLFW_KEY_0)
		centerImage();
	else if (key == GLFW_KEY_1)
		setZoom(1.f);
	else if (key == GLFW_KEY_2)
		setZoom(2.f);
	else if (key == GLFW_KEY_F1)
		uiShowDemoWindow = !uiShowDemoWindow;
	else if (key == GLFW_KEY_R)
		drop(m_renderThread.getFilename());
}

void ImguiScreen::keyReleased(int key, int mods)
{
}

void ImguiScreen::mouseButtonPressed(int button, int mods)
{
	if (!m_renderThread.isBusy() && !m_renderThread.m_guiScene)
		filebrowser.Open();
}
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
	ImageBlock& compositeBlock = m_renderThread.getBlock();
	compositeBlock.lock();
	const Vector2i bsize = compositeBlock.getSize();
	compositeBlock.unlock();

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
	float scale = 1.f - 0.05f * static_cast<float>(yoffset);
	setZoom(clamp(imageZoom / scale, 1.f / 30.f, 30.f));
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

	imGuiIo.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
	imGuiIo.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

	ImGui::StyleColorsDark();

	// ImGuiStyle &imGuiStyle = ImGui::GetStyle();
	// imGuiStyle.FrameRounding = 3.f;

	ImGui_ImplGlfw_InitForOpenGL(glfwWindow, true);
	const char *glsl_version = "#version 150";
	ImGui_ImplOpenGL3_Init(glsl_version);

	//    imGuiIo.Fonts->AddFontFromFileTTF("imgui/misc/fonts/Roboto-Medium.ttf", 16.f);
}


void nori::MouseState::onMouseClick(double xPos, double yPos, int button, int action, int _mods) {
	mods = _mods;

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

void nori::MouseState::onMouseMove(double xPos, double yPos) {
	mouseMoveX = lastMouseX - xPos;
	mouseMoveY = -lastMouseY + yPos;
	lastMouseX = xPos;
	lastMouseY = yPos;
}

NORI_NAMESPACE_END
