#pragma once

#include <nori/object.h>
#include <filesystem>
#include <nanovdb/util/IO.h>
#include <filesystem/resolver.h>

NORI_NAMESPACE_BEGIN

class Volume : public NoriObject
{
public:
    Volume(const PropertyList &props);
    EClassType getClassType() const override { return EVolume; }
    std::string toString() const override;
    filesystem::path filename;
    nanovdb::GridHandle<nanovdb::HostBuffer> densityHandle, heatHandle;
    nanovdb::NanoGrid<float> *grid;
    nanovdb::NanoGrid<float> *heat;

private:
    void loadNanoVdb();
    void loadOpenVdbAndCacheNanoVdb(const std::filesystem::path &cacheFilename);
    void writeToNanoVdb(std::string filename);
};

NORI_NAMESPACE_END