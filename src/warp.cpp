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
  return (p.norm() <= 1.0f) ? 1.0f / M_PI : 0.0f;
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
  return (v.norm() <= 1.0f && v.z() <= cosThetaMax)
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
  return (v.norm() <= 1.0f) ? 0.25f / M_PI : 0.0f;
}

Vector3f Warp::squareToUniformHemisphere(const Point2f &sample) {
  Vector3f w = squareToUniformSphere(sample);
  w.z() = abs(w.z()); // z to absolute valu
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

NORI_NAMESPACE_END
