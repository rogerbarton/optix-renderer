/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob

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

#include <nori/frame.h>
#include <nori/vector.h>
#include <nori/warp.h>

NORI_NAMESPACE_BEGIN

Vector3f Warp::sampleUniformHemisphere(Sampler *sampler, const Normal3f &pole) {
  // Naive implementation using rejection sampling
  Vector3f v;
  do {
    v.x() = 1.f - 2.f * sampler->next1D();
    v.y() = 1.f - 2.f * sampler->next1D();
    v.z() = 1.f - 2.f * sampler->next1D();
  } while (v.squaredNorm() > 1.f);

  if (v.dot(pole) < 0.f)
    v = -v;
  v /= v.norm();

  return v;
}

Point2f Warp::squareToUniformSquare(const Point2f &sample) { return sample; }

float Warp::squareToUniformSquarePdf(const Point2f &sample) {
  return ((sample.array() >= 0).all() && (sample.array() <= 1).all()) ? 1.0f
                                                                      : 0.0f;
}

Point2f Warp::squareToUniformDisk(const Point2f &sample) {
  float rho = sqrt(sample.x());
  float theta = sample.y() * 2.0f * M_PI;
  return Point2f(rho * cos(theta), rho * sin(theta));
}

float Warp::squareToUniformDiskPdf(const Point2f &p) {
  return (p.squaredNorm() <= 1.0f) ? 1.0f / M_PI : 0.0f;
}

Vector3f Warp::squareToUniformSphereCap(const Point2f &sample,
                                        float cosThetaMax) {
  float z = sample.x() * (1.0f - cosThetaMax) + cosThetaMax;
  float r = sqrt(1.0f - z * z);
  float theta = sample.y() * 2.0f * M_PI;
  float x = r * cos(theta);
  float y = r * sin(theta);
  return Vector3f(x, y, z);
}

float Warp::squareToUniformSphereCapPdf(const Vector3f &v, float cosThetaMax) {
  return (std::abs(v.squaredNorm() - 1.0f) < Epsilon && v.z() <= cosThetaMax)
             ? 0.0f
             : 1.0f / (2.0f * M_PI * (1.0f - cosThetaMax));
}

Vector3f Warp::squareToUniformSphere(const Point2f &sample) {
  Vector3f w;
  w.z() = 2.0f * sample.x() - 1.0f;
  float r = sqrt(1.0f - w.z() * w.z());
  float sigma = 2.0f * M_PI * sample.y();
  w.x() = r * cos(sigma);
  w.y() = r * sin(sigma);
  return w.normalized();
}

float Warp::squareToUniformSpherePdf(const Vector3f &v) {
  return (std::abs(v.squaredNorm() - 1.0f) < Epsilon) ? 0.25f / M_PI : 0.0f;
}

Vector3f Warp::squareToUniformSphereVolume(const Point3f &sample)
{
	const float r     = sqrt(sample.x());
	const float theta = 1.f * M_PI * sample.y();
	const float phi   = 2.f * M_PI * sample.z();
	return Vector3f(
			r * sin(theta) * cos(phi),
			r * sin(theta) * sin(phi),
			r * cos(theta));
}

float Warp::squareToUniformSphereVolumePdf(const Point3f &sample)
{
	return 4.f / 3.f * M_PI;
}

Vector3f Warp::squareToUniformHemisphere(const Point2f &sample) {
  Vector3f w = squareToUniformSphere(sample);
  w.z() = abs(w.z()); // z to absolute value
  return w;
}

float Warp::squareToUniformHemispherePdf(const Vector3f &v) {
  return (std::abs(v.squaredNorm() - 1) < Epsilon && v.z() > 0)
             ? 2.0f * squareToUniformSpherePdf(v)
             : 0.0f;
}

Vector3f Warp::squareToCosineHemisphere(const Point2f &sample) {
  // squareToUniformDisk and project onto hemisphere (x,y stay, z projected)
  Point2f diskPoint = squareToUniformDisk(sample);
  // map this point (z coordinate) to hemisphere
  Vector3f ret;
  ret.x() = diskPoint.x();
  ret.y() = diskPoint.y();

  ret.z() = sqrt(1.f - diskPoint.squaredNorm());

  return ret;
}

float Warp::squareToCosineHemispherePdf(const Vector3f &v) {
  // angle between v and the z axis divided by pi
  // this results in v_z / PI
  return (std::abs(v.squaredNorm() - 1) < Epsilon && v.z() > 0) ? v.z() / M_PI
                                                                : 0.0f;
}

Vector3f Warp::squareToBeckmann(const Point2f &sample, float alpha) {
  // http://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Reflection_Functions.html
  float logSample = log(1.f - sample.x());
  if (std::isinf(logSample))
    logSample = 0;
  float tan2Theta = -alpha * alpha * logSample;
  float phi = sample.y() * 2.f * M_PI;
  float cosTheta = 1.f / sqrt(1 + tan2Theta);
  float sinTheta = sqrt(1.f - cosTheta * cosTheta);

  // spherical direction
  Vector3f res =
      Vector3f(sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta);

  if (res.z() < 0) {
    res = -res;
  }

  return res;
}

float Warp::squareToBeckmannPdf(const Vector3f &m, float alpha) {
  float c_t = m.z();
  float r = std::sqrt(m.x() * m.x() + m.y() * m.y());
  float tantheta = r / m.z();
  if (std::abs(m.squaredNorm() - 1) > Epsilon || m.z() < 0)
    return 0.f;
  return exp(-tantheta * tantheta / alpha / alpha) /
         (M_PI * alpha * alpha * c_t * c_t * c_t);
}

Vector3f Warp::squareToUniformTriangle(const Point2f &sample) {
  float su1 = sqrtf(sample.x());
  float u = 1.f - su1, v = sample.y() * su1;
  return Vector3f(u, v, 1.f - u - v);
}

Vector3f Warp::squareToHenyeyGreenstein(const Point2f &sample, float g)
{
	/**
	 * See https://www.oceanopticsbook.info/view/scattering/level-2/the-henyey-greenstein-phase-function
	 * phase function integral phi=[pi/2, pi = *partial* cdf =
	 *   B_HG = (1 - g) / 2g *((1 + g)/sqrt(1 + g^2) - 1)
	 */
	float cosTheta;
	if (std::abs(g) < Epsilon)
		cosTheta = 1 - 2 * sample.x();
	else
	{
		const float factor = (1 - g * g) / (1 - g + 2 * g * sample.x());
		cosTheta = (1 + g * g - factor * factor) / (2 * g);
	}

	// random rotation about wi axis
	const float phi = 2 * M_PI * sample.y();

	// Direction from spherical coords
	const float sinTheta = std::sqrt(1 - cosTheta * cosTheta);
	float sinPhi, cosPhi;
	sincosf(phi, &sinPhi, &cosPhi);

	return Vector3f(
			sinTheta * cosPhi,
			sinTheta * sinPhi,
			cosTheta);
}

float Warp::squareToHenyeyGreensteinPdf(const Vector3f &m, float g)
{
	const float cosTheta = Frame::cosTheta(m);
	const float g2 = g * g;
	return 0.25f / M_PI * (1 - g2) / std::pow(1 + g2 - 2 * g * cosTheta, 1.5f);
}

Vector3f Warp::squareToSchlick(const Point2f &sample, float k)
{
	float cosTheta;
	if (std::abs(k) < Epsilon)
		cosTheta = 1;
	else
	{
		cosTheta = 1 / k * (1 - 1 / (2 * k * (sample.x() - 1 + k * k + 1 / (2 * k * (1 - k)))));
	}

	// random rotation about wi axis
	const float phi = 2 * M_PI * sample.y();

	// Direction from spherical coords
	const float sinTheta = std::sqrt(1 - cosTheta * cosTheta);
	float sinPhi, cosPhi;
	sincosf(phi, &sinPhi, &cosPhi);

	return Vector3f(
			sinTheta * cosPhi,
			sinTheta * sinPhi,
			cosTheta);
}

float Warp::squareToSchlickPdf(const Vector3f &m, float k)
{
	// const float k = 1.55f * g - 0.55f * std::pow(g, 3);
	const float factor = 1 - k * Frame::cosTheta(m);
	return 0.25f / M_PI * (1 - k * k) / factor;
}

NORI_NAMESPACE_END
