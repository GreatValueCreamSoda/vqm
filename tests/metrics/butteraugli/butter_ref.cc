// Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
// BSD-3-Clause; see LICENSE.
// Butteraugli portions: Copyright (c) the JPEG XL Project Authors.
// BSD-3-Clause; see licenses/JPEG-XL-LICENSE.
//
// Test-only double-precision oracle for the libjxl butteraugli comparator.
// This is an independent transliteration of the comparator path of
// .reference/libjxl/lib/jxl/butteraugli/butteraugli.cc (BSD-3-Clause); the line
// citations in the header of butter_ref.h identify every ported block. The
// code below follows the source's pointer arithmetic literally: the 16 Malta
// line kernels keep libjxl's deliberate duplicates (they are part of the
// metric), and the low-frequency terms use the 5-tap variant. Arithmetic is
// double precision throughout, with MIRROR / renormalized borders as in the
// source; it is single-threaded and deterministic.

#include "butter_ref.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace vqm::test {
namespace {

struct Plane {
    int width = 0;
    int height = 0;
    std::vector<double> data;
};

std::size_t Offset(const Plane& plane, int x, int y) {
    return static_cast<std::size_t>(y) * static_cast<std::size_t>(plane.width) +
           static_cast<std::size_t>(x);
}

double Get(const Plane& plane, int x, int y) {
    return plane.data[Offset(plane, x, y)];
}

void Set(Plane& plane, int x, int y, double value) {
    plane.data[Offset(plane, x, y)] = value;
}

double& At(Plane& plane, int x, int y) {
    return plane.data[Offset(plane, x, y)];
}

Plane MakePlane(int width, int height) {
    Plane plane;
    plane.width = width;
    plane.height = height;
    plane.data.assign(
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0.0
    );
    return plane;
}

struct PsychoImage {
    Plane lf[3];
    Plane mf[3];
    Plane hf[2];
    Plane uhf[2];
};

// butteraugli.cc:78-88. The radius, tap values and normalization are all
// computed in double here (the source uses float).
std::vector<double> ComputeKernel(double sigma) {
    constexpr double kRadiusScale = 2.25;
    const double scaler = -1.0 / (2.0 * sigma * sigma);
    const int radius =
        std::max(1, static_cast<int>(kRadiusScale * std::fabs(sigma)));
    std::vector<double> kernel(static_cast<std::size_t>(2 * radius + 1));
    for (int i = -radius; i <= radius; ++i) {
        kernel[static_cast<std::size_t>(i + radius)] =
            std::exp(scaler * static_cast<double>(i * i));
    }
    return kernel;
}

// image_ops.h:184-196.
int Mirror(int x, int size) {
    while (x < 0 || x >= size) {
        x = x < 0 ? -x - 1 : 2 * size - 1 - x;
    }
    return x;
}

// butteraugli.cc:99-265. A 5-tap kernel uses the Separable5 path: taps are
// normalized to sum one and out-of-range samples repeat the border pixel once
// (Mirror). Larger kernels use ConvolutionWithTranspose: interior pixels use
// the globally normalized taps, border pixels renormalize by the sum of the
// taps that actually fall inside the image.
Plane Blur(const Plane& in, double sigma) {
    const std::vector<double> kernel = ComputeKernel(sigma);
    const int radius = static_cast<int>(kernel.size() / 2);
    Plane temp = MakePlane(in.width, in.height);
    if (kernel.size() == 5) {
        double sum = 0.0;
        for (double weight : kernel) {
            sum += weight;
        }
        const double center = kernel[2] / sum;
        const double near = kernel[1] / sum;
        const double far = kernel[0] / sum;
        for (int y = 0; y < in.height; ++y) {
            for (int x = 0; x < in.width; ++x) {
                const double value =
                    center * Get(in, x, y) +
                    near * (Get(in, Mirror(x - 1, in.width), y) +
                            Get(in, Mirror(x + 1, in.width), y)) +
                    far * (Get(in, Mirror(x - 2, in.width), y) +
                           Get(in, Mirror(x + 2, in.width), y));
                Set(temp, x, y, value);
            }
        }
        Plane out = MakePlane(in.width, in.height);
        for (int y = 0; y < in.height; ++y) {
            for (int x = 0; x < in.width; ++x) {
                const double value =
                    center * Get(temp, x, y) +
                    near * (Get(temp, x, Mirror(y - 1, in.height)) +
                            Get(temp, x, Mirror(y + 1, in.height))) +
                    far * (Get(temp, x, Mirror(y - 2, in.height)) +
                           Get(temp, x, Mirror(y + 2, in.height)));
                Set(out, x, y, value);
            }
        }
        return out;
    }

    for (int y = 0; y < in.height; ++y) {
        for (int x = 0; x < in.width; ++x) {
            double sum = 0.0;
            double weight_sum = 0.0;
            for (int i = -radius; i <= radius; ++i) {
                const int xx = x + i;
                if (xx < 0 || xx >= in.width) {
                    continue;
                }
                const double weight =
                    kernel[static_cast<std::size_t>(i + radius)];
                sum += Get(in, xx, y) * weight;
                weight_sum += weight;
            }
            Set(temp, x, y, sum / weight_sum);
        }
    }
    Plane out = MakePlane(in.width, in.height);
    for (int y = 0; y < in.height; ++y) {
        for (int x = 0; x < in.width; ++x) {
            double sum = 0.0;
            double weight_sum = 0.0;
            for (int i = -radius; i <= radius; ++i) {
                const int yy = y + i;
                if (yy < 0 || yy >= in.height) {
                    continue;
                }
                const double weight =
                    kernel[static_cast<std::size_t>(i + radius)];
                sum += Get(temp, x, yy) * weight;
                weight_sum += weight;
            }
            Set(out, x, y, sum / weight_sum);
        }
    }
    return out;
}

// butteraugli.cc:1425-1465. `clamp` selects the clamped absorbance used for
// the sensitivity estimate.
void OpsinAbsorbance(
    double in0, double in1, double in2, bool clamp, double* out0, double* out1,
    double* out2
) {
    constexpr double kMix0 = 0.29956550340058319;
    constexpr double kMix1 = 0.63373087833825936;
    constexpr double kMix2 = 0.077705617820981968;
    constexpr double kMix3 = 1.7557483643287353;
    constexpr double kMix4 = 0.22158691104574774;
    constexpr double kMix5 = 0.69391388044116142;
    constexpr double kMix6 = 0.0987313588422;
    constexpr double kMix7 = 1.7557483643287353;
    constexpr double kMix8 = 0.02;
    constexpr double kMix9 = 0.02;
    constexpr double kMix10 = 0.20480129041026129;
    constexpr double kMix11 = 12.226454707163354;

    *out0 = kMix0 * in0 + (kMix1 * in1 + (kMix2 * in2 + kMix3));
    *out1 = kMix4 * in0 + (kMix5 * in1 + (kMix6 * in2 + kMix7));
    *out2 = kMix8 * in0 + (kMix9 * in1 + (kMix10 * in2 + kMix11));
    if (clamp) {
        *out0 = std::max(*out0, kMix3);
        *out1 = std::max(*out1, kMix7);
        *out2 = std::max(*out2, kMix11);
    }
}

// butteraugli.cc:1407-1422. libjxl evaluates the same expression through
// FastLog2f; std::log is used here.
double Gamma(double v) {
    v = std::max(v, 0.0);
    return 19.245013259874995 * std::log(v + 9.9710635769299145) -
           23.16046239805755;
}

// butteraugli.cc:1468-1540.
void OpsinDynamicsImage(
    const Plane rgb[3], double intensity_target, Plane xyb[3]
) {
    constexpr double kSigma = 1.2;
    Plane blurred[3];
    for (int c = 0; c < 3; ++c) {
        blurred[c] = Blur(rgb[c], kSigma);
    }
    for (int c = 0; c < 3; ++c) {
        xyb[c] = MakePlane(rgb[c].width, rgb[c].height);
    }
    const int width = rgb[0].width;
    const int height = rgb[0].height;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const double br = Get(blurred[0], x, y) * intensity_target;
            const double bg = Get(blurred[1], x, y) * intensity_target;
            const double bb = Get(blurred[2], x, y) * intensity_target;
            double pre0 = 0.0;
            double pre1 = 0.0;
            double pre2 = 0.0;
            OpsinAbsorbance(br, bg, bb, true, &pre0, &pre1, &pre2);
            pre0 = std::max(pre0, 1e-4);
            pre1 = std::max(pre1, 1e-4);
            pre2 = std::max(pre2, 1e-4);
            const double sensitivity0 = std::max(Gamma(pre0) / pre0, 1e-4);
            const double sensitivity1 = std::max(Gamma(pre1) / pre1, 1e-4);
            const double sensitivity2 = std::max(Gamma(pre2) / pre2, 1e-4);

            double cur0 = 0.0;
            double cur1 = 0.0;
            double cur2 = 0.0;
            OpsinAbsorbance(
                Get(rgb[0], x, y) * intensity_target,
                Get(rgb[1], x, y) * intensity_target,
                Get(rgb[2], x, y) * intensity_target, false, &cur0, &cur1, &cur2
            );
            cur0 = std::max(cur0 * sensitivity0, 1.7557483643287353);
            cur1 = std::max(cur1 * sensitivity1, 1.7557483643287353);
            cur2 = std::max(cur2 * sensitivity2, 12.226454707163354);
            Set(xyb[0], x, y, cur0 - cur1);
            Set(xyb[1], x, y, cur0 + cur1);
            Set(xyb[2], x, y, cur2);
        }
    }
}

// butteraugli.cc:296-325.
double RemoveRangeAroundZero(double x, double w) {
    if (x > w) {
        return x - w;
    }
    if (x < -w) {
        return x + w;
    }
    return 0.0;
}

double AmplifyRangeAroundZero(double x, double w) {
    if (x > w) {
        return x + w;
    }
    if (x < -w) {
        return x - w;
    }
    return x + x;
}

double MaximumClamp(double v, double maxval) {
    constexpr double kMul = 0.724216145665;
    if (v >= maxval) {
        return maxval + (v - maxval) * kMul;
    }
    if (v < -maxval) {
        return -maxval + (v + maxval) * kMul;
    }
    return v;
}

// butteraugli.cc:404-558, including XybLowFreqToVals (:327-342) and
// SuppressXByY (:367-390).
PsychoImage SeparateFrequencies(const Plane xyb[3]) {
    PsychoImage ps;
    constexpr double kSigmaLf = 7.15593339443;
    constexpr double kSigmaHf = 3.22489901262;
    constexpr double kSigmaUhf = 1.56416327805;

    // SeparateLFAndMF, then the LF to "vals" conversion. The blue output uses
    // the original (pre-ymul) y value.
    for (int c = 0; c < 3; ++c) {
        ps.lf[c] = Blur(xyb[c], kSigmaLf);
        ps.mf[c] = MakePlane(xyb[c].width, xyb[c].height);
        for (int y = 0; y < xyb[c].height; ++y) {
            for (int x = 0; x < xyb[c].width; ++x) {
                Set(ps.mf[c], x, y, Get(xyb[c], x, y) - Get(ps.lf[c], x, y));
            }
        }
    }
    for (int y = 0; y < xyb[0].height; ++y) {
        for (int x = 0; x < xyb[0].width; ++x) {
            const double xv = Get(ps.lf[0], x, y);
            const double yv = Get(ps.lf[1], x, y);
            const double bv = Get(ps.lf[2], x, y);
            Set(ps.lf[0], x, y, xv * 33.832837186260);
            Set(ps.lf[1], x, y, yv * 14.458268100570);
            Set(ps.lf[2], x, y, (bv + (-0.362267051518) * yv) * 49.87984651440);
        }
    }

    // SeparateMFAndHF.
    for (int c = 0; c < 3; ++c) {
        if (c == 2) {
            ps.mf[2] = Blur(ps.mf[2], kSigmaHf);
            break;
        }
        ps.hf[c] = ps.mf[c];
        const Plane blurred = Blur(ps.mf[c], kSigmaHf);
        const double range = c == 0 ? 0.29 : 0.1;
        for (int y = 0; y < blurred.height; ++y) {
            for (int x = 0; x < blurred.width; ++x) {
                const double mfv = Get(blurred, x, y);
                // The source subtracts before the range modification.
                const double hfv = Get(ps.hf[c], x, y) - mfv;
                Set(ps.mf[c], x, y,
                    c == 0 ? RemoveRangeAroundZero(mfv, range)
                           : AmplifyRangeAroundZero(mfv, range));
                Set(ps.hf[c], x, y, hfv);
            }
        }
    }

    // Suppress red-green by intensity change in the high-frequency channels.
    constexpr double kSuppress = 46.0;
    constexpr double kSuppressScale = 0.653020556257;
    for (int y = 0; y < ps.hf[0].height; ++y) {
        for (int x = 0; x < ps.hf[0].width; ++x) {
            const double yv = Get(ps.hf[1], x, y);
            const double factor =
                kSuppressScale +
                (1.0 - kSuppressScale) * (kSuppress / (yv * yv + kSuppress));
            Set(ps.hf[0], x, y, Get(ps.hf[0], x, y) * factor);
        }
    }

    // SeparateHFAndUHF.
    for (int i = 0; i < 2; ++i) {
        ps.uhf[i] = ps.hf[i];
        ps.hf[i] = Blur(ps.hf[i], kSigmaUhf);
        if (i == 0) {
            for (int y = 0; y < ps.hf[0].height; ++y) {
                for (int x = 0; x < ps.hf[0].width; ++x) {
                    const double hfv = Get(ps.hf[0], x, y);
                    const double uhfv = Get(ps.uhf[0], x, y) - hfv;
                    Set(ps.hf[0], x, y, RemoveRangeAroundZero(hfv, 1.5));
                    Set(ps.uhf[0], x, y, RemoveRangeAroundZero(uhfv, 0.04));
                }
            }
        } else {
            for (int y = 0; y < ps.hf[1].height; ++y) {
                for (int x = 0; x < ps.hf[1].width; ++x) {
                    double hfv =
                        MaximumClamp(Get(ps.hf[1], x, y), 28.4691806922);
                    double uhfv = Get(ps.uhf[1], x, y) - hfv;
                    uhfv = MaximumClamp(uhfv, 5.19175294647) * 2.69313763794;
                    Set(ps.uhf[1], x, y, uhfv);
                    hfv = AmplifyRangeAroundZero(hfv * 2.155, 0.132);
                    Set(ps.hf[1], x, y, hfv);
                }
            }
        }
    }
    return ps;
}

// The Malta line kernels. Offsets are (dy, dx) from the centre. The full unit
// has 16 terms, including libjxl's deliberate duplicates T13..T16
// (butteraugli.cc:579-949); the LF unit has 16 distinct 5-tap terms.
struct Tap {
    int dy;
    int dx;
};

struct TapLine {
    int count;
    Tap taps[9];
};

const TapLine kFullUnit[16] = {
    {9,
     {{0, -4},
      {0, -3},
      {0, -2},
      {0, -1},
      {0, 0},
      {0, 1},
      {0, 2},
      {0, 3},
      {0, 4}}},
    {9,
     {{-4, 0},
      {-3, 0},
      {-2, 0},
      {-1, 0},
      {0, 0},
      {1, 0},
      {2, 0},
      {3, 0},
      {4, 0}}},
    {7, {{-3, -3}, {-2, -2}, {-1, -1}, {0, 0}, {1, 1}, {2, 2}, {3, 3}}},
    {7, {{-3, 3}, {-2, 2}, {-1, 1}, {0, 0}, {1, -1}, {2, -2}, {3, -3}}},
    {9,
     {{-4, 1},
      {-3, 1},
      {-2, 1},
      {-1, 0},
      {0, 0},
      {1, 0},
      {2, -1},
      {3, -1},
      {4, -1}}},
    {9,
     {{-4, -1},
      {-3, -1},
      {-2, -1},
      {-1, 0},
      {0, 0},
      {1, 0},
      {2, 1},
      {3, 1},
      {4, 1}}},
    {9,
     {{-1, -4},
      {-1, -3},
      {-1, -2},
      {0, -1},
      {0, 0},
      {0, 1},
      {1, 2},
      {1, 3},
      {1, 4}}},
    {9,
     {{1, -4},
      {1, -3},
      {1, -2},
      {0, -1},
      {0, 0},
      {0, 1},
      {-1, 2},
      {-1, 3},
      {-1, 4}}},
    {7, {{-3, -2}, {-2, -1}, {-1, -1}, {0, 0}, {1, 1}, {2, 1}, {3, 2}}},
    {7, {{-3, 2}, {-2, 1}, {-1, 1}, {0, 0}, {1, -1}, {2, -1}, {3, -2}}},
    {7, {{-2, -3}, {-1, -2}, {-1, -1}, {0, 0}, {1, 1}, {1, 2}, {2, 3}}},
    {7, {{-2, 3}, {-1, 2}, {-1, 1}, {0, 0}, {1, -1}, {1, -2}, {2, -3}}},
    {9,
     {{1, -4},
      {1, -3},
      {1, -2},
      {0, -1},
      {0, 0},
      {0, 1},
      {-1, 2},
      {-1, 3},
      {-1, 4}}},
    {9,
     {{-1, -4},
      {-1, -3},
      {-1, -2},
      {0, -1},
      {0, 0},
      {0, 1},
      {1, 2},
      {1, 3},
      {1, 4}}},
    {9,
     {{-4, -1},
      {-3, -1},
      {-2, -1},
      {-1, 0},
      {0, 0},
      {1, 0},
      {2, 1},
      {3, 1},
      {4, 1}}},
    {9,
     {{-4, 1},
      {-3, 1},
      {-2, 1},
      {-1, 0},
      {0, 0},
      {1, 0},
      {2, -1},
      {3, -1},
      {4, -1}}},
};

const TapLine kLfUnit[16] = {
    {5, {{0, -4}, {0, -2}, {0, 0}, {0, 2}, {0, 4}}},
    {5, {{-4, 0}, {-2, 0}, {0, 0}, {2, 0}, {4, 0}}},
    {5, {{-3, -3}, {-2, -2}, {0, 0}, {2, 2}, {3, 3}}},
    {5, {{-3, 3}, {-2, 2}, {0, 0}, {2, -2}, {3, -3}}},
    {5, {{-4, 1}, {-2, 1}, {0, 0}, {2, -1}, {4, -1}}},
    {5, {{-4, -1}, {-2, -1}, {0, 0}, {2, 1}, {4, 1}}},
    {5, {{-1, -4}, {-1, -2}, {0, 0}, {1, 2}, {1, 4}}},
    {5, {{1, -4}, {1, -2}, {0, 0}, {-1, 2}, {-1, 4}}},
    {5, {{-3, -2}, {-2, -1}, {0, 0}, {2, 1}, {3, 2}}},
    {5, {{-3, 2}, {-2, 1}, {0, 0}, {2, -1}, {3, -2}}},
    {5, {{-2, -3}, {-1, -2}, {0, 0}, {1, 2}, {2, 3}}},
    {5, {{-2, 3}, {-1, 2}, {0, 0}, {1, -2}, {2, -3}}},
    {5, {{2, -4}, {1, -2}, {0, 0}, {-1, 2}, {-2, 4}}},
    {5, {{-2, -4}, {-1, -2}, {0, 0}, {1, 2}, {2, 4}}},
    {5, {{-4, -2}, {-2, -1}, {0, 0}, {2, 1}, {4, 2}}},
    {5, {{-4, 2}, {-2, 1}, {0, 0}, {2, -1}, {4, -2}}},
};

// butteraugli.cc:953-985. Zero-padded out-of-bounds access; the first term
// initializes the accumulator.
double MaltaUnit(const Plane& diffs, int x0, int y0, const TapLine* lines) {
    double retval = 0.0;
    for (int term = 0; term < 16; ++term) {
        const TapLine& line = lines[term];
        double sum = 0.0;
        for (int t = 0; t < line.count; ++t) {
            const int x = x0 + line.taps[t].dx;
            const int y = y0 + line.taps[t].dy;
            if (x >= 0 && x < diffs.width && y >= 0 && y < diffs.height) {
                sum += Get(diffs, x, y);
            }
        }
        retval += sum * sum;
    }
    return retval;
}

// butteraugli.cc:988-1108.
void MaltaDiffMap(
    const Plane& lum0, const Plane& lum1, double w_0gt1, double w_0lt1,
    double norm1, bool lf, Plane& diffs, Plane& block_diff_ac
) {
    constexpr double kLen = 3.75;
    const double mulli = lf ? 0.611612573796 : 0.39905817637;
    const double w_pre0gt1 =
        mulli * std::sqrt(0.5 * w_0gt1) / (2.0 * kLen + 1.0);
    const double w_pre0lt1 =
        mulli * std::sqrt(0.33 * w_0lt1) / (2.0 * kLen + 1.0);
    const double norm2_0gt1 = w_pre0gt1 * norm1;
    const double norm2_0lt1 = w_pre0lt1 * norm1;
    const int width = lum0.width;
    const int height = lum0.height;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const double val0 = Get(lum0, x, y);
            const double val1 = Get(lum1, x, y);
            const double absval = 0.5 * (std::fabs(val0) + std::fabs(val1));
            const double diff = val0 - val1;
            const double scaler = norm2_0gt1 / (norm1 + absval);

            // Primary symmetric quadratic objective.
            double d = scaler * diff;

            const double scaler2 = norm2_0lt1 / (norm1 + absval);
            const double fabs0 = std::fabs(val0);

            // Secondary half-open quadratic objectives.
            const double too_small = 0.55 * fabs0;
            const double too_big = 1.05 * fabs0;
            if (val0 < 0.0) {
                if (val1 > -too_small) {
                    d -= scaler2 * (val1 + too_small);
                } else if (val1 < -too_big) {
                    d += scaler2 * (-val1 - too_big);
                }
            } else {
                if (val1 < too_small) {
                    d += scaler2 * (too_small - val1);
                } else if (val1 > too_big) {
                    d -= scaler2 * (val1 - too_big);
                }
            }
            Set(diffs, x, y, d);
        }
    }

    const TapLine* lines = lf ? kLfUnit : kFullUnit;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            At(block_diff_ac, x, y) += MaltaUnit(diffs, x, y, lines);
        }
    }
}

// butteraugli.cc:1320-1405. L2Diff returns early for a zero weight.
void L2Diff(const Plane& i0, const Plane& i1, double w, Plane& diffmap) {
    if (w == 0.0) {
        return;
    }
    for (int y = 0; y < i0.height; ++y) {
        for (int x = 0; x < i0.width; ++x) {
            const double diff = Get(i0, x, y) - Get(i1, x, y);
            At(diffmap, x, y) += w * diff * diff;
        }
    }
}

void SetL2Diff(const Plane& i0, const Plane& i1, double w, Plane& diffmap) {
    if (w == 0.0) {
        return;
    }
    for (int y = 0; y < i0.height; ++y) {
        for (int x = 0; x < i0.width; ++x) {
            const double diff = Get(i0, x, y) - Get(i1, x, y);
            Set(diffmap, x, y, w * diff * diff);
        }
    }
}

void L2DiffAsymmetric(
    const Plane& i0, const Plane& i1, double w_0gt1, double w_0lt1,
    Plane& diffmap
) {
    if (w_0gt1 == 0.0 && w_0lt1 == 0.0) {
        return;
    }
    const double vw_0gt1 = w_0gt1 * 0.8;
    const double vw_0lt1 = w_0lt1 * 0.8;
    for (int y = 0; y < i0.height; ++y) {
        for (int x = 0; x < i0.width; ++x) {
            const double val0 = Get(i0, x, y);
            const double val1 = Get(i1, x, y);

            // Primary symmetric quadratic objective.
            const double diff = val0 - val1;
            double total = At(diffmap, x, y) + diff * diff * vw_0gt1;

            // Secondary half-open quadratic objectives.
            const double fabs0 = std::fabs(val0);
            const double too_small = 0.4 * fabs0;
            const double too_big = fabs0;
            double v = 0.0;
            if (val0 < 0.0) {
                if (val1 > -too_small) {
                    v = val1 + too_small;
                } else if (val1 < -too_big) {
                    v = -val1 - too_big;
                }
            } else {
                if (val1 < too_small) {
                    v = too_small - val1;
                } else if (val1 > too_big) {
                    v = val1 - too_big;
                }
            }
            total += vw_0lt1 * v * v;
            Set(diffmap, x, y, total);
        }
    }
}

// butteraugli.cc:1110-1134.
Plane CombineChannelsForMasking(const Plane hf[2], const Plane uhf[2]) {
    Plane out = MakePlane(hf[0].width, hf[0].height);
    for (int y = 0; y < out.height; ++y) {
        for (int x = 0; x < out.width; ++x) {
            const double xdiff = (Get(uhf[0], x, y) + Get(hf[0], x, y)) * 2.5;
            const double ydiff =
                Get(uhf[1], x, y) * 0.4 + Get(hf[1], x, y) * 0.4;
            Set(out, x, y, std::sqrt(xdiff * xdiff + ydiff * ydiff));
        }
    }
    return out;
}

// butteraugli.cc:1136-1150.
Plane DiffPrecompute(const Plane& in) {
    constexpr double kMul = 6.19424080439;
    constexpr double kBiasArg = 12.61050594197;
    const double bias = kMul * kBiasArg;
    const double sqrt_bias = std::sqrt(bias);
    Plane out = MakePlane(in.width, in.height);
    for (int y = 0; y < in.height; ++y) {
        for (int x = 0; x < in.width; ++x) {
            Set(out, x, y,
                std::sqrt(kMul * std::fabs(Get(in, x, y)) + bias) - sqrt_bias);
        }
    }
    return out;
}

// butteraugli.cc:1157-1172.
void StoreMin3(double v, double& min0, double& min1, double& min2) {
    if (v < min2) {
        if (v < min0) {
            min2 = min1;
            min1 = min0;
            min0 = v;
        } else if (v < min1) {
            min2 = min1;
            min1 = v;
        } else {
            min2 = v;
        }
    }
}

// butteraugli.cc:1174-1207.
Plane FuzzyErosion(const Plane& from) {
    constexpr int kStep = 3;
    Plane out = MakePlane(from.width, from.height);
    for (int y = 0; y < from.height; ++y) {
        for (int x = 0; x < from.width; ++x) {
            double min0 = Get(from, x, y);
            double min1 = 2.0 * min0;
            double min2 = min1;
            if (x >= kStep) {
                StoreMin3(Get(from, x - kStep, y), min0, min1, min2);
                if (y >= kStep) {
                    StoreMin3(Get(from, x - kStep, y - kStep), min0, min1, min2);
                }
                if (y + kStep < from.height) {
                    StoreMin3(Get(from, x - kStep, y + kStep), min0, min1, min2);
                }
            }
            if (x + kStep < from.width) {
                StoreMin3(Get(from, x + kStep, y), min0, min1, min2);
                if (y >= kStep) {
                    StoreMin3(Get(from, x + kStep, y - kStep), min0, min1, min2);
                }
                if (y + kStep < from.height) {
                    StoreMin3(Get(from, x + kStep, y + kStep), min0, min1, min2);
                }
            }
            if (y >= kStep) {
                StoreMin3(Get(from, x, y - kStep), min0, min1, min2);
            }
            if (y + kStep < from.height) {
                StoreMin3(Get(from, x, y + kStep), min0, min1, min2);
            }
            Set(out, x, y, 0.45 * min0 + 0.3 * min1 + 0.25 * min2);
        }
    }
    return out;
}

// butteraugli.cc:1214-1266.
Plane Mask(const Plane& mask0, const Plane& mask1, Plane& diff_ac) {
    constexpr double kRadius = 2.7;
    const Plane diff0 = DiffPrecompute(mask0);
    const Plane diff1 = DiffPrecompute(mask1);
    const Plane blurred0 = Blur(diff0, kRadius);
    Plane result = FuzzyErosion(blurred0);
    const Plane blurred1 = Blur(diff1, kRadius);
    for (int y = 0; y < result.height; ++y) {
        for (int x = 0; x < result.width; ++x) {
            const double diff = Get(blurred0, x, y) - Get(blurred1, x, y);
            At(diff_ac, x, y) += 10.0 * diff * diff;
        }
    }
    return result;
}

// butteraugli.cc:1268-1275.
double MaskY(double delta) {
    constexpr double kOffset = 0.829591754942;
    constexpr double kScaler = 0.451936922203;
    constexpr double kMul = 2.5485944793;
    constexpr double kGlobalScale = 1.0 / (17.83 * 0.79079917404);
    const double c = kMul / (kScaler * delta + kOffset);
    const double retval = kGlobalScale * (1.0 + c);
    return retval * retval;
}

double MaskDcY(double delta) {
    constexpr double kOffset = 0.20025578522;
    constexpr double kScaler = 3.87449418804;
    constexpr double kMul = 0.505054525019;
    constexpr double kGlobalScale = 1.0 / (17.83 * 0.79079917404);
    const double c = kMul / (kScaler * delta + kOffset);
    const double retval = kGlobalScale * (1.0 + c);
    return retval * retval;
}

constexpr double kWUhfMalta = 1.10039032555;
constexpr double kNorm1Uhf = 71.7800275169;
constexpr double kWUhfMaltaX = 173.5;
constexpr double kNorm1UhfX = 5.0;
constexpr double kWHfMalta = 18.7237414387;
constexpr double kNorm1Hf = 4498534.45232;
constexpr double kWHfMaltaX = 6923.99476109;
constexpr double kNorm1HfX = 8051.15833247;
constexpr double kWMfMalta = 37.0819870399;
constexpr double kNorm1Mf = 130262059.556;
constexpr double kWMfMaltaX = 8246.75321353;
constexpr double kNorm1MfX = 1009002.70582;
constexpr double kWmul[9] = {
    400.0,         1.50815703118,  0.0,
    2150.0,        10.6195433239,  16.2176043152,
    29.2353797994, 0.844626970982, 0.703646627719,
};

// butteraugli.cc:1894-1952.
Plane DiffmapPsychoImage(
    const PsychoImage& pi0, const PsychoImage& pi1, const ButterRefParams& params
) {
    const int width = pi0.mf[0].width;
    const int height = pi0.mf[0].height;
    Plane block_diff_ac[3];
    Plane block_diff_dc[3];
    for (int c = 0; c < 3; ++c) {
        block_diff_ac[c] = MakePlane(width, height);
        block_diff_dc[c] = MakePlane(width, height);
    }
    Plane diffs = MakePlane(width, height);
    const double asymmetry = params.hf_asymmetry;
    const double sqrt_asymmetry = std::sqrt(asymmetry);

    MaltaDiffMap(
        pi0.uhf[1], pi1.uhf[1], kWUhfMalta * asymmetry, kWUhfMalta / asymmetry,
        kNorm1Uhf, false, diffs, block_diff_ac[1]
    );
    MaltaDiffMap(
        pi0.uhf[0], pi1.uhf[0], kWUhfMaltaX * asymmetry, kWUhfMaltaX / asymmetry,
        kNorm1UhfX, false, diffs, block_diff_ac[0]
    );
    MaltaDiffMap(
        pi0.hf[1], pi1.hf[1], kWHfMalta * sqrt_asymmetry,
        kWHfMalta / sqrt_asymmetry, kNorm1Hf, true, diffs, block_diff_ac[1]
    );
    MaltaDiffMap(
        pi0.hf[0], pi1.hf[0], kWHfMaltaX * sqrt_asymmetry,
        kWHfMaltaX / sqrt_asymmetry, kNorm1HfX, true, diffs, block_diff_ac[0]
    );
    MaltaDiffMap(
        pi0.mf[1], pi1.mf[1], kWMfMalta, kWMfMalta, kNorm1Mf, true, diffs,
        block_diff_ac[1]
    );
    MaltaDiffMap(
        pi0.mf[0], pi1.mf[0], kWMfMaltaX, kWMfMaltaX, kNorm1MfX, true, diffs,
        block_diff_ac[0]
    );

    for (int c = 0; c < 3; ++c) {
        if (c < 2) {
            // No blue-channel error is accumulated at HF.
            L2DiffAsymmetric(
                pi0.hf[c], pi1.hf[c], kWmul[c] * asymmetry, kWmul[c] / asymmetry,
                block_diff_ac[c]
            );
        }
        L2Diff(pi0.mf[c], pi1.mf[c], kWmul[3 + c], block_diff_ac[c]);
        SetL2Diff(pi0.lf[c], pi1.lf[c], kWmul[6 + c], block_diff_dc[c]);
    }

    const Plane mask0 = CombineChannelsForMasking(pi0.hf, pi0.uhf);
    const Plane mask1 = CombineChannelsForMasking(pi1.hf, pi1.uhf);
    const Plane mask = Mask(mask0, mask1, block_diff_ac[1]);

    // butteraugli.cc:1268-1318.
    Plane result = MakePlane(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const double maskval = MaskY(Get(mask, x, y));
            const double dc_maskval = MaskDcY(Get(mask, x, y));
            const double diff_dc0 = Get(block_diff_dc[0], x, y) * params.xmul;
            const double diff_ac0 = Get(block_diff_ac[0], x, y) * params.xmul;
            const double dc_total = diff_dc0 + Get(block_diff_dc[1], x, y) +
                                    Get(block_diff_dc[2], x, y);
            const double ac_total = diff_ac0 + Get(block_diff_ac[1], x, y) +
                                    Get(block_diff_ac[2], x, y);
            Set(result, x, y,
                std::sqrt(dc_total * dc_maskval + ac_total * maskval));
        }
    }
    return result;
}

// butteraugli.cc:1733-1766.
Plane SubSample2x(const Plane& in) {
    const int width = (in.width + 1) / 2;
    const int height = (in.height + 1) / 2;
    Plane out = MakePlane(width, height);
    for (int y = 0; y < in.height; ++y) {
        for (int x = 0; x < in.width; ++x) {
            At(out, x / 2, y / 2) += 0.25 * Get(in, x, y);
        }
    }
    if ((in.width & 1) != 0) {
        for (int y = 0; y < height; ++y) {
            At(out, width - 1, y) *= 2.0;
        }
    }
    if ((in.height & 1) != 0) {
        for (int x = 0; x < width; ++x) {
            At(out, x, height - 1) *= 2.0;
        }
    }
    return out;
}

// butteraugli.cc:1768-1784.
void AddSupersampled2x(const Plane& src, double weight, Plane& dest) {
    constexpr double kHeuristicMixingValue = 0.3;
    for (int y = 0; y < dest.height; ++y) {
        for (int x = 0; x < dest.width; ++x) {
            double value = Get(dest, x, y);
            value *= 1.0 - kHeuristicMixingValue * weight;
            value += weight * Get(src, x / 2, y / 2);
            Set(dest, x, y, value);
        }
    }
}

// butteraugli.cc:1792-1858: two levels only. Diffmap folds exactly one
// subsampled level, and only when that level is at least 8x8.
Plane ComparatorDiffmap(
    const Plane rgb0[3], const Plane rgb1[3], const ButterRefParams& params
) {
    const int width = rgb0[0].width;
    const int height = rgb0[0].height;
    Plane xyb0[3];
    Plane xyb1[3];
    OpsinDynamicsImage(rgb0, params.intensity_target, xyb0);
    OpsinDynamicsImage(rgb1, params.intensity_target, xyb1);
    const PsychoImage pi0 = SeparateFrequencies(xyb0);
    const PsychoImage pi1 = SeparateFrequencies(xyb1);
    Plane result = DiffmapPsychoImage(pi0, pi1, params);

    const int sub_width = (width + 1) / 2;
    const int sub_height = (height + 1) / 2;
    if (sub_width < 8 || sub_height < 8) {
        return result;
    }

    Plane sub_rgb0[3];
    Plane sub_rgb1[3];
    for (int c = 0; c < 3; ++c) {
        sub_rgb0[c] = SubSample2x(rgb0[c]);
        sub_rgb1[c] = SubSample2x(rgb1[c]);
    }
    Plane sub_xyb0[3];
    Plane sub_xyb1[3];
    OpsinDynamicsImage(sub_rgb0, params.intensity_target, sub_xyb0);
    OpsinDynamicsImage(sub_rgb1, params.intensity_target, sub_xyb1);
    const PsychoImage sub_pi0 = SeparateFrequencies(sub_xyb0);
    const PsychoImage sub_pi1 = SeparateFrequencies(sub_xyb1);
    const Plane sub_result = DiffmapPsychoImage(sub_pi0, sub_pi1, params);
    AddSupersampled2x(sub_result, 0.5, result);
    return result;
}

}  // namespace

ButterRefResult ButteraugliCpuReference(
    const float* r0, const float* g0, const float* b0, const float* r1,
    const float* g1, const float* b1, int width, int height,
    const ButterRefParams& params
) {
    ButterRefResult result;
    if (width <= 0 || height <= 0) {
        return result;
    }
    const float* first[3] = {r0, g0, b0};
    const float* second[3] = {r1, g1, b1};
    for (int c = 0; c < 3; ++c) {
        if (first[c] == nullptr || second[c] == nullptr) {
            return result;
        }
    }

    // butteraugli.cc:1974-2011 ButteraugliDiffmapSmall. Images below 8x8 are
    // edge-extended symmetrically, then the diffmap is cropped back.
    const bool small = width < 8 || height < 8;
    const int padded_width = std::max(8, width);
    const int padded_height = std::max(8, height);
    const int xborder = width < 8 ? (8 - width) / 2 : 0;
    const int yborder = height < 8 ? (8 - height) / 2 : 0;

    Plane rgb0[3];
    Plane rgb1[3];
    for (int c = 0; c < 3; ++c) {
        rgb0[c] = MakePlane(padded_width, padded_height);
        rgb1[c] = MakePlane(padded_width, padded_height);
    }
    for (int y = 0; y < padded_height; ++y) {
        const int sy =
            small ? std::min(height - 1, y > yborder ? y - yborder : 0) : y;
        for (int x = 0; x < padded_width; ++x) {
            const int sx =
                small ? std::min(width - 1, x > xborder ? x - xborder : 0) : x;
            const std::size_t index =
                static_cast<std::size_t>(sy) * static_cast<std::size_t>(width) +
                static_cast<std::size_t>(sx);
            for (int c = 0; c < 3; ++c) {
                Set(rgb0[c], x, y, static_cast<double>(first[c][index]));
                Set(rgb1[c], x, y, static_cast<double>(second[c][index]));
            }
        }
    }

    Plane diffmap = ComparatorDiffmap(rgb0, rgb1, params);
    if (small) {
        Plane cropped = MakePlane(width, height);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                Set(cropped, x, y, Get(diffmap, x + xborder, y + yborder));
            }
        }
        diffmap = std::move(cropped);
    }

    // butteraugli.cc:1954-1964 and lib/extras/metrics.cc:42-145.
    double score = 0.0;
    double sum3 = 0.0;
    double sum6 = 0.0;
    double sum12 = 0.0;
    for (double value : diffmap.data) {
        score = std::max(score, value);
        const double value2 = value * value;
        const double value3 = value2 * value;
        sum3 += value3;
        const double value6 = value3 * value3;
        sum6 += value6;
        const double value12 = value6 * value6;
        sum12 += value12;
    }
    const std::size_t count = diffmap.data.size();
    const double inverse_count = 1.0 / static_cast<double>(count);
    const double distance = std::pow(sum3 * inverse_count, 1.0 / 3.0) +
                            std::pow(sum6 * inverse_count, 1.0 / 6.0) +
                            std::pow(sum12 * inverse_count, 1.0 / 12.0);
    result.score = score;
    result.distance_p3 = distance / 3.0;
    return result;
}

}  // namespace vqm::test
