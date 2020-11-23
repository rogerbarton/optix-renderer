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

#if !defined(__NORI_OBJECT_H)
#define __NORI_OBJECT_H

#include <nori/proplist.h>
#include <imgui/imgui.h>
#include <nori/ImguiHelpers.h>

NORI_NAMESPACE_BEGIN

/*
 * This struct is used to store a cumulative function distribution (CFD) of samples
 */
struct Histogram
{
    using upair = std::pair<int, int>;
    using map_type = std::map<float, upair>;
    using elem_type = map_type::const_iterator;

    float cumulative = 0.f;
    map_type map;

    elem_type getElement(float prob) const {
        return map.lower_bound(prob * cumulative);
    }

    void add_element(int i, int j, float value)
    {
        cumulative += value;
        map[cumulative] = upair(i, j);
    }
};

/**
 * \brief Base class of all objects
 *
 * A Nori object represents an instance that is part of
 * a scene description, e.g. a scattering model or emitter.
 */
class NoriObject
{
public:
    enum EClassType
    {
	    EScene = 0,
	    EShape,
	    ETexture,
	    EVolume,
	    EBSDF,
	    EPhaseFunction,
	    EEmitter,
	    EMedium,
	    ECamera,
	    EIntegrator,
	    ESampler,
        EPixelSampler,
        EDenoiser,
        ETest,
        EReconstructionFilter,
        EClassTypeCount
    };

    /// Turn a class type into a human-readable string
    static std::string classTypeName(EClassType type)
    {
        switch (type)
        {
        case EScene:
            return "scene";
        case EShape:
            return "shape";
        case ETexture:
            return "texture";
        case EVolume:
            return "volume";
        case EBSDF:
            return "bsdf";
        case EEmitter:
            return "emitter";
        case ECamera:
            return "camera";
        case EIntegrator:
            return "integrator";
        case ESampler:
            return "sampler";
        case ETest:
            return "test";
        case EDenoiser:
            return "denoiser";
        case EPixelSampler:
            return "pixelsampler";
        default:
            return "<unknown>";
        }
    }

    /// Virtual destructor
    virtual ~NoriObject() {}

    /**
     * \brief Return the type of object (i.e. Mesh/BSDF/etc.) 
     * provided by this instance
     * */
    virtual EClassType getClassType() const = 0;

    /**
     * \brief Add a child object to the current instance
     *
     * The default implementation does not support children and
     * simply throws an exception
     */
    virtual void addChild(NoriObject *child)
    {
        throw NoriException(
            "NoriObject::addChild() is not implemented for objects of type '%s'!",
            classTypeName(getClassType()));
    }

    /**
     * \brief Set the parent object
     *
     * Subclasses may choose to override this method to be
     * notified when they are added to a parent object. The
     * default implementation does nothing.
     */
    virtual void setParent(NoriObject *parent)
    { /* Do nothing */
    }

    /**
     * Clones the object, initializes the copy and returns the new copy. Does a deep copy.
     */
	virtual NoriObject *cloneAndInit() = 0;

	/**
	 * Creates a copy constructor for the derived class.
	 * Can be used in most cases where there are no pointer members, as this function should perform a deep copy.
	 * See also: https://stackoverflow.com/questions/12255546/c-deep-copying-a-base-class-pointer
	 */
#   define NORI_OBJECT_DEFAULT_CLONE(cls) \
	NoriObject *cloneAndInit() override { \
		return new cls(*this);            \
	}                                     \

	/**
	 * If the gui has modified the object, optionally add for children.
	 * This can be modified for const objects.
	 */
    mutable bool touched = true;

    /**
     * Initialize the object when the scene has changed before rendering.
     * You can use the NORI_OBJECT_DEFAULT_UPDATE macro.
     */
    virtual void update(const NoriObject *guiObject) = 0;

    /**
     * Implements the NoriObject::update() by copying ALL members if touched.
     * Use this if all members are (xml) properties.
     */
#   define NORI_OBJECT_DEFAULT_UPDATE(cls)                   \
	void update(const NoriObject *guiObject) override {      \
        const auto *gui = static_cast<const cls *>(guiObject);   \
		if(!gui->touched) return;                            \
        gui->touched = false;                                \
        *this = *gui;                                        \
	}

    /// Return a brief string summary of the instance (for debugging purposes)
    virtual std::string toString() const = 0;

    /// Allow to assign a name to the object
    void setIdName(const std::string &name) { m_idname = name; }
    const std::string &getIdName() const { return m_idname; }
#ifndef NORI_USE_NANOGUI
	virtual std::string getImGuiName() const  {
		return tfm::format("%s%s", "Object", (touched ? "*" : ""));
	}

	/**
	 * Sets the display name in the scene tree. Indicates if the object was touched
	 */
#   define NORI_OBJECT_IMGUI_NAME(cls)                         \
	std::string getImGuiName() const override {                \
		return tfm::format("%s%s", cls, (touched ? "*" : "")); \
	}

    virtual bool getImGuiNodes() = 0;
#endif

protected:
    std::string m_idname;
};

/**
 * \brief Factory for Nori objects
 *
 * This utility class is part of a mini-RTTI framework and can 
 * instantiate arbitrary Nori objects by their name.
 */
class NoriObjectFactory
{
public:
    typedef std::function<NoriObject *(const PropertyList &)> Constructor;

    /**
     * \brief Register an object constructor with the object factory
     *
     * This function is called by the macro \ref NORI_REGISTER_CLASS
     *
     * \param name
     *     An internal name that is associated with this class. This is the
     *     'type' field found in the scene description XML files
     *
     * \param constr
     *     A function pointer to an anonymous function that is
     *     able to call the constructor of the class.
     */
    static void registerClass(const std::string &name, const Constructor &constr);

    /**
     * \brief Construct an instance from the class of the given name
     *
     * \param name
     *     An internal name that is associated with this class. This is the
     *     'type' field found in the scene description XML files
     *
     * \param propList
     *     A list of properties that will be passed to the constructor
     *     of the class.
     */
    static NoriObject *createInstance(const std::string &name,
                                      const PropertyList &propList)
    {
        if (!m_constructors || m_constructors->find(name) == m_constructors->end())
            throw NoriException("A constructor for class \"%s\" could not be found!", name);
        return (*m_constructors)[name](propList);
    }

    static void printRegisteredClasses()
    {
        if (m_constructors)
            for (auto v : *m_constructors)
                std::cout << v.first << std::endl;
    }

private:
    static std::map<std::string, Constructor> *m_constructors;
};

/// Macro for registering an object constructor with the \ref NoriObjectFactory
#define NORI_REGISTER_CLASS(cls, name)                            \
    cls *cls##_create(const PropertyList &list)                   \
    {                                                             \
        return new cls(list);                                     \
    }                                                             \
    static struct cls##_                                          \
    {                                                             \
        cls##_()                                                  \
        {                                                         \
            NoriObjectFactory::registerClass(name, cls##_create); \
        }                                                         \
    } cls##__NORI_;

/// Macro for registering an object constructor with the \ref NoriObjectFactory
#define NORI_REGISTER_TEMPLATED_CLASS(cls, T, name)                     \
    cls<T> *cls##_##T##_create(const PropertyList &list)                \
    {                                                                   \
        return new cls<T>(list);                                        \
    }                                                                   \
    static struct cls##_##T##_                                          \
    {                                                                   \
        cls##_##T##_()                                                  \
        {                                                               \
            NoriObjectFactory::registerClass(name, cls##_##T##_create); \
        }                                                               \
    } cls##T##__NORI_;

NORI_NAMESPACE_END

#endif /* __NORI_OBJECT_H */
