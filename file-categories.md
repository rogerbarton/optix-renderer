# helpers
- color.h
- common.h
- hdrloader.h
- parser.h, object.h, proplist.h
- timer.h
- vector.h
- warp.h

# testing
- chi2, ttest, warptest

# gui/display
- bitmap.h
- block.h
- glutil.h
- gui.h, imguiScreen, imguiHelpers
- ~render.h
- rfilter.h
- hdrToLdr.cpp

## post process
- denoiser.h

# renderer (cuda part)
- ~render.h
- scene.h
- integrator.h: direct_{ems,mats,mis}.cpp, path_{mats,mis}.cpp, av.cpp, normals.cpp, photonmapper.cpp, preview
- sampler.h: adaptive.cpp, independent.cpp
- bsdf.h: diffuse.cpp, microfacet.cpp, dielectric.cpp
- emitter.h, arealight, pointlight, envmap
- texture.h: checkerboard, pngtexture, consttexture
- shape.h
-   transform.h
-   mesh.h (sampling surface), sphere.cpp, objloader
-   volume.h

## helpers
- dpdf.h (i.e. discretepdf cuda?)
- frame.h
- ray.h

# scene (params part, raygen)
- camera.h: perspective.cpp
- mesh.h

# acceleration structure
- bvh.h
- bbox.h
- kdtree.h
- photon.h
