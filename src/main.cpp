/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob, Romain Prévost

    Nori is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Nori is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <nori/block.h>
#ifndef DISABLE_NORI_GUI
//#  define USE_NANOGUI   // Toggles betweeen the standard nanogui nori viewer and imgui viewer
#  ifdef USE_NANOGUI
#    include <nori/gui.h>
#  else
#    include <nori/DebugGui.h>
#    include <GLFW/glfw3.h>
#  endif
#else
  #include <nori/render.h>
#endif
#include <filesystem/path.h>

int main(int argc, char **argv) {
    using namespace nori;

    try {
#ifndef DISABLE_NORI_GUI
#  ifdef USE_NANOGUI
        nanogui::init();

        // Open the UI with a dummy image
        ImageBlock block(Vector2i(720, 720), nullptr);
        NoriScreen *screen = new NoriScreen(block);

        // if file is passed as argument, handle it
        if (argc == 2) {
            std::string filename = argv[1];
            filesystem::path path(filename);

            if (path.extension() == "xml") {
                /* Render the XML scene file */
                screen->openXML(filename);
            } else if (path.extension() == "exr") {
                /* Alternatively, provide a basic OpenEXR image viewer */
                screen->openEXR(filename);
            } else {
                cerr << "Error: unknown file \"" << filename
                << "\", expected an extension of type .xml or .exr" << endl;
            }
        }

        nanogui::mainloop();
        delete screen;
        nanogui::shutdown();
#  else
        ImageBlock block(Vector2i(720, 720), nullptr);
		DebugGui gui{block};

	    // if file is passed as argument, handle it
	    if (argc == 2) {
		    std::string filename = argv[1];
		    filesystem::path path(filename);

		    if (path.extension() == "xml") {
			    /* Render the XML scene file */
			    gui.openXML(filename);
		    } else if (path.extension() == "exr") {
			    /* Alternatively, provide a basic OpenEXR image viewer */
			    gui.openEXR(filename);
		    } else {
			    cerr << "Error: unknown file \"" << filename
			         << "\", expected an extension of type .xml or .exr" << endl;
		    }
	    }

	    // TODO: render and ui draw loop
	    while(!glfwWindowShouldClose(gui.glfwWindow))
        {
            gui.newFrame();

            gui.endFrame();
        }
#  endif
#else
        if (argc == 2) {
            ImageBlock block(Vector2i(720, 720), nullptr);
            RenderThread m_renderThread(block);
            std::string filename = argv[1];
            filesystem::path path(filename);

            if (path.extension() == "xml") {
                /* Render the XML scene file */
                m_renderThread.renderScene(filename);
                // waiting until rendere is finished
                while(m_renderThread.isBusy());
            } else {
                cerr << "Error: unknown file \"" << filename
                << "\", expected an extension of type .xml" << endl;
            }
        }
#endif

    } catch (const std::exception &e) {
        cerr << "Fatal error: " << e.what() << endl;
        return -1;
    }
    return 0;
}
