// Copyright 2023, DragonflyDB authors.  All rights reserved.
// See LICENSE for licensing terms.
//

#include "core/search/vector_utils.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>

#include "base/logging.h"

namespace dfly::search {

using namespace std;

namespace {

#ifdef WITH_SIMSIMD
#include <simsimd/simsimd.h>
#endif

#if defined(__GNUC__) && !defined(__clang__)
#define FAST_MATH __attribute__((optimize("fast-math")))
#else
#define FAST_MATH
#endif

OwnedFtVector ConvertToFtVector(string_view value) {
  size_t size = value.size() / sizeof(float);
  auto out = make_unique<std::byte[]>(value.size());
  memcpy(out.get(), value.data(), value.size());

  return OwnedFtVector{std::move(out), size};
}

}  // namespace

// Euclidean vector distance: sqrt( sum: (u[i] - v[i])^2  )
FAST_MATH float L2Distance(const float* u, const float* v, size_t dims) {
#ifdef WITH_SIMSIMD
  simsimd_distance_t distance = 0;
  simsimd_l2_f32(u, v, dims, &distance);
  return static_cast<float>(distance);
#else
  float sum = 0;
  for (size_t i = 0; i < dims; i++)
    sum += (u[i] - v[i]) * (u[i] - v[i]);
  return sqrt(sum);
#endif
}

// Inner product distance: 1 - dot_product(u, v)
// For normalized vectors, this is equivalent to cosine distance
FAST_MATH float IPDistance(const float* u, const float* v, size_t dims) {
#ifdef WITH_SIMSIMD
  // Use SimSIMD dot product and convert to inner product distance: 1 - dot(u, v).
  simsimd_distance_t dot = 0;
  simsimd_dot_f32(u, v, dims, &dot);
  return 1.0f - static_cast<float>(dot);
#else
  float sum_uv = 0;
  for (size_t i = 0; i < dims; i++)
    sum_uv += u[i] * v[i];
  return 1.0f - sum_uv;
#endif
}

// Cosine distance: 1 - (dot_product(u, v) / (||u|| * ||v||))
FAST_MATH float CosineDistance(const float* u, const float* v, size_t dims) {
#ifdef WITH_SIMSIMD
  simsimd_distance_t distance = 0;
  simsimd_cos_f32(u, v, dims, &distance);
  return static_cast<float>(distance);
#else
  float sum_uv = 0, sum_uu = 0, sum_vv = 0;
  for (size_t i = 0; i < dims; i++) {
    sum_uv += u[i] * v[i];
    sum_uu += u[i] * u[i];
    sum_vv += v[i] * v[i];
  }

  if (float denom = sum_uu * sum_vv; denom != 0.0f)
    return 1 - sum_uv / sqrt(denom);
  return 0.0f;
#endif
}

OwnedFtVector BytesToFtVector(string_view value) {
  DCHECK_EQ(value.size() % sizeof(float), 0u) << value.size();
  return ConvertToFtVector(value);
}

std::optional<OwnedFtVector> BytesToFtVectorSafe(string_view value) {
  if (value.size() % sizeof(float)) {
    return std::nullopt;
  }
  return ConvertToFtVector(value);
}

float VectorDistance(const float* u, const float* v, size_t dims, VectorSimilarity sim) {
  switch (sim) {
    case VectorSimilarity::L2:
      return L2Distance(u, v, dims);
    case VectorSimilarity::IP:
      return IPDistance(u, v, dims);
    case VectorSimilarity::COSINE:
      return CosineDistance(u, v, dims);
  };
  return 0.0f;
}

float HalfToFloat(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;  // +/- zero
    } else {
      exp = 127 - 15 + 1;  // normalize the subnormal
      while ((mant & 0x400) == 0) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x3FF;
      bits = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1F) {
    bits = sign | 0x7F800000u | (mant << 13);  // inf / nan
  } else {
    bits = sign | ((exp + 112) << 23) | (mant << 13);  // rebias 127 - 15 = 112
  }
  float out;
  memcpy(&out, &bits, sizeof(out));
  return out;
}

float Bf16ToFloat(uint16_t b) {
  uint32_t bits = static_cast<uint32_t>(b) << 16;
  float out;
  memcpy(&out, &bits, sizeof(out));
  return out;
}

uint16_t FloatToHalf(float f) {
  uint32_t x;
  memcpy(&x, &f, sizeof(x));
  const uint32_t sign = (x >> 16) & 0x8000;
  const uint32_t exp_field = (x >> 23) & 0xFF;
  uint32_t mant = x & 0x7FFFFF;

  if (exp_field == 0xFF)  // inf / nan
    return static_cast<uint16_t>(sign | 0x7C00 | (mant ? 0x200 : 0));

  int32_t exp = static_cast<int32_t>(exp_field) - 127 + 15;
  if (exp >= 0x1F)  // overflow -> inf
    return static_cast<uint16_t>(sign | 0x7C00);

  if (exp <= 0) {  // subnormal or zero
    if (exp < -10)
      return static_cast<uint16_t>(sign);
    mant |= 0x800000;
    const int shift = 14 - exp;
    uint32_t half_mant = mant >> shift;
    const uint32_t remainder = mant & ((1u << shift) - 1);
    const uint32_t halfway = 1u << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (half_mant & 1)))
      half_mant++;
    return static_cast<uint16_t>(sign | half_mant);
  }

  uint16_t half = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
  const uint32_t remainder = mant & 0x1FFF;  // round to nearest even
  if (remainder > 0x1000 || (remainder == 0x1000 && (half & 1)))
    half++;
  return half;
}

uint16_t FloatToBf16(float f) {
  uint32_t bits;
  memcpy(&bits, &f, sizeof(bits));
  if ((bits & 0x7FFFFFFF) > 0x7F800000)  // nan -> quiet nan
    return static_cast<uint16_t>((bits >> 16) | 0x0040);
  bits += 0x7FFF + ((bits >> 16) & 1);  // round to nearest even
  return static_cast<uint16_t>(bits >> 16);
}

std::vector<std::byte> EncodeOnesVector(size_t dim, VectorDataType dt) {
  const size_t width = ElementSize(dt);
  std::byte elem[sizeof(double)] = {};
  switch (dt) {
    case VectorDataType::FLOAT32: {
      float v = 1.0f;
      memcpy(elem, &v, sizeof(v));
    } break;
    case VectorDataType::FLOAT64: {
      double v = 1.0;
      memcpy(elem, &v, sizeof(v));
    } break;
    case VectorDataType::FLOAT16: {
      uint16_t v = FloatToHalf(1.0f);
      memcpy(elem, &v, sizeof(v));
    } break;
    case VectorDataType::BFLOAT16: {
      uint16_t v = FloatToBf16(1.0f);
      memcpy(elem, &v, sizeof(v));
    } break;
    case VectorDataType::INT8: {
      int8_t v = 1;
      memcpy(elem, &v, sizeof(v));
    } break;
    case VectorDataType::UINT8: {
      uint8_t v = 1;
      memcpy(elem, &v, sizeof(v));
    } break;
  }
  std::vector<std::byte> out(dim * width);
  for (size_t i = 0; i < dim; i++)
    memcpy(out.data() + i * width, elem, width);
  return out;
}

namespace {

uint16_t LoadU16(const void* base, size_t i) {
  uint16_t v;
  memcpy(&v, static_cast<const char*>(base) + i * sizeof(uint16_t), sizeof(v));
  return v;
}

// Byte-safe reads: native-width blobs may be unaligned (borrowed keyspace storage), so a
// typed deref would be UB. memcpy lowers to a single load on the target ISAs.
float LoadF32(const void* base, size_t i) {
  float v;
  memcpy(&v, static_cast<const char*>(base) + i * sizeof(float), sizeof(v));
  return v;
}

double LoadF64(const void* base, size_t i) {
  double v;
  memcpy(&v, static_cast<const char*>(base) + i * sizeof(double), sizeof(v));
  return v;
}

// Reads element i of a native-width blob, widened to Acc (double for FLOAT64, else float).
template <VectorDataType DT> struct Reader;
template <> struct Reader<VectorDataType::FLOAT32> {
  using Acc = float;
  static float Get(const void* p, size_t i) {
    return LoadF32(p, i);
  }
};
template <> struct Reader<VectorDataType::FLOAT64> {
  using Acc = double;
  static double Get(const void* p, size_t i) {
    return LoadF64(p, i);
  }
};
template <> struct Reader<VectorDataType::FLOAT16> {
  using Acc = float;
  static float Get(const void* p, size_t i) {
    return HalfToFloat(LoadU16(p, i));
  }
};
template <> struct Reader<VectorDataType::BFLOAT16> {
  using Acc = float;
  static float Get(const void* p, size_t i) {
    return Bf16ToFloat(LoadU16(p, i));
  }
};
template <> struct Reader<VectorDataType::INT8> {
  using Acc = float;
  static float Get(const void* p, size_t i) {
    return static_cast<const int8_t*>(p)[i];
  }
};
template <> struct Reader<VectorDataType::UINT8> {
  using Acc = float;
  static float Get(const void* p, size_t i) {
    return static_cast<const uint8_t*>(p)[i];
  }
};

template <VectorDataType DT> float L2Typed(const void* u, const void* v, size_t dims) {
  using Acc = typename Reader<DT>::Acc;
  Acc sum = 0;
  for (size_t i = 0; i < dims; i++) {
    Acc d = Reader<DT>::Get(u, i) - Reader<DT>::Get(v, i);
    sum += d * d;
  }
  return static_cast<float>(std::sqrt(sum));
}

template <VectorDataType DT> float IPTyped(const void* u, const void* v, size_t dims) {
  using Acc = typename Reader<DT>::Acc;
  Acc sum = 0;
  for (size_t i = 0; i < dims; i++)
    sum += Reader<DT>::Get(u, i) * Reader<DT>::Get(v, i);
  return static_cast<float>(Acc(1) - sum);
}

template <VectorDataType DT> float CosineTyped(const void* u, const void* v, size_t dims) {
  using Acc = typename Reader<DT>::Acc;
  Acc uv = 0, uu = 0, vv = 0;
  for (size_t i = 0; i < dims; i++) {
    Acc a = Reader<DT>::Get(u, i), b = Reader<DT>::Get(v, i);
    uv += a * b;
    uu += a * a;
    vv += b * b;
  }
  if (Acc denom = uu * vv; denom != Acc(0))
    return static_cast<float>(Acc(1) - uv / std::sqrt(denom));
  return 0.0f;
}

template <VectorDataType DT>
float DistByMetric(const void* u, const void* v, size_t dims, VectorSimilarity sim) {
  switch (sim) {
    case VectorSimilarity::L2:
      return L2Typed<DT>(u, v, dims);
    case VectorSimilarity::IP:
      return IPTyped<DT>(u, v, dims);
    case VectorSimilarity::COSINE:
      return CosineTyped<DT>(u, v, dims);
  }
  return 0.0f;
}

}  // namespace

float VectorDistance(const void* u, const void* v, size_t dims, VectorSimilarity sim,
                     VectorDataType dt) {
  switch (dt) {
    case VectorDataType::FLOAT32:
      return DistByMetric<VectorDataType::FLOAT32>(u, v, dims, sim);
    case VectorDataType::FLOAT64:
      return DistByMetric<VectorDataType::FLOAT64>(u, v, dims, sim);
    case VectorDataType::FLOAT16:
      return DistByMetric<VectorDataType::FLOAT16>(u, v, dims, sim);
    case VectorDataType::BFLOAT16:
      return DistByMetric<VectorDataType::BFLOAT16>(u, v, dims, sim);
    case VectorDataType::INT8:
      return DistByMetric<VectorDataType::INT8>(u, v, dims, sim);
    case VectorDataType::UINT8:
      return DistByMetric<VectorDataType::UINT8>(u, v, dims, sim);
  }
  return 0.0f;
}

std::string_view VectorSimilarityToString(VectorSimilarity sim) {
  switch (sim) {
    case VectorSimilarity::L2:
      return "L2";
    case VectorSimilarity::IP:
      return "IP";
    case VectorSimilarity::COSINE:
      return "COSINE";
  }
  DCHECK(false) << "Unhandled VectorSimilarity enum value: " << static_cast<int>(sim);
  return "L2";
}

std::string_view VectorDataTypeToString(VectorDataType dt) {
  switch (dt) {
    case VectorDataType::FLOAT32:
      return "FLOAT32";
    case VectorDataType::FLOAT64:
      return "FLOAT64";
    case VectorDataType::FLOAT16:
      return "FLOAT16";
    case VectorDataType::BFLOAT16:
      return "BFLOAT16";
    case VectorDataType::INT8:
      return "INT8";
    case VectorDataType::UINT8:
      return "UINT8";
  }
  DCHECK(false) << "Unhandled VectorDataType enum value: " << static_cast<int>(dt);
  return "FLOAT32";
}

std::optional<VectorDataType> ParseVectorDataType(std::string_view name) {
  if (name == "FLOAT32")
    return VectorDataType::FLOAT32;
  if (name == "FLOAT64")
    return VectorDataType::FLOAT64;
  if (name == "FLOAT16")
    return VectorDataType::FLOAT16;
  if (name == "BFLOAT16")
    return VectorDataType::BFLOAT16;
  if (name == "INT8")
    return VectorDataType::INT8;
  if (name == "UINT8")
    return VectorDataType::UINT8;
  return std::nullopt;
}

float DistanceToSimilarity(float distance, VectorSimilarity sim) {
  switch (sim) {
    case VectorSimilarity::L2:
      return 1.0f / (1.0f + distance * distance);
    case VectorSimilarity::IP:
    case VectorSimilarity::COSINE:
      return (2.0f - distance) / 2.0f;
  }
  DCHECK(false) << "Unhandled VectorSimilarity enum value: " << static_cast<int>(sim);
  return 0.0f;
}

void InitSimSIMD() {
#if defined(WITH_SIMSIMD)
  (void)simsimd_capabilities();
#endif
}

}  // namespace dfly::search
