# OptiX renderer based on Nori

![demo](./reports/optix-demo.gif)

Extended course project with GPU hardware acceleration. Features:

1. Combined CPU and GPU path tracing.
2. OptiX GPU image denoising.
3. [nvrtc](https://docs.nvidia.com/cuda/nvrtc/index.html) runtime compilation of CUDA shaders.
4. [ImGui](https://github.com/ocornut/imgui) for editing scene parameters with immediate feedback

## Nori

Nori is a simple ray tracer written in C++. It runs on Windows, Linux, and
Mac OS and provides basic functionality that is required to complete the
assignments in the course Computer Graphics taught at ETH Zürich.

### Course information and framework documentation

For access to course information including slides and reading material, visit the main [Computer Graphics website](https://graphics.ethz.ch/teaching/cg20/home.php). The Nori 2 framework and coding assignments will be described on the [Nori website](https://graphics.ethz.ch/teaching/cg20/nori.php).
