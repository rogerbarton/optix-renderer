/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Romain Prévost

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

#if !defined(__NORI_TEXTURE_H)
#define __NORI_TEXTURE_H

#include <nori/object.h>

NORI_NAMESPACE_BEGIN

/**
 * \brief Superclass of all texture
 */
template <typename T>
class Texture : public NoriObject {
public:
    Texture() {}
    virtual ~Texture() {}

    /**
     * \brief Return the type of object (i.e. Mesh/Emitter/etc.) 
     * provided by this instance
     * */
    virtual EClassType getClassType() const override { return ETexture; }

    /**
     * @brief Eval the texture at the current point (in uv-coordinates)
     */
    virtual T eval(const Point2f & uv) = 0;

    /**
     * @brief Get Width of image, defaults to 1, used for PNG textures
     */
    virtual unsigned int getWidth() {
		return 1;
	}

    /**
     * @brief Get height of image, defaults to 1, used for PNG textures
     */
	virtual unsigned int getHeight() {
		return 1;
	}
#ifndef NORI_USE_NANOGUI
    virtual const char *getImGuiName() const override { return "Texture Base"; }
    virtual void getImGuiNodes() override {}
#endif
};

NORI_NAMESPACE_END

#endif /* __NORI_TEXTURE_H */
