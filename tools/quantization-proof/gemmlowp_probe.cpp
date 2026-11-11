// Calls the pinned upstream scalar primitives. Input vectors come from the checker.
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
// Keep this probe scalar on every host, including hosts that enable SIMD by
// default. The platform header's guard prevents a second detection below.
#include "internal/detect_platform.h"
#undef GEMMLOWP_NEON
#undef GEMMLOWP_AVX2
#undef GEMMLOWP_SSE4
#undef GEMMLOWP_MSA
#undef GEMMLOWP_WASMSIMD
#include "fixedpoint/fixedpoint.h"

int main() {
  char op;
  std::int32_t a, b, shift, zero;
  while (std::cin >> op >> a >> b) {
    if (op == 'h') {
      std::cout << gemmlowp::SaturatingRoundingDoublingHighMul(a, b) << '\n';
    } else if (op == 'd') {
      std::cout << gemmlowp::RoundingDivideByPOT(a, b) << '\n';
    } else if (op == 'q' && std::cin >> shift >> zero) {
      const auto product = gemmlowp::SaturatingRoundingDoublingHighMul(a, b);
      const auto scaled = gemmlowp::RoundingDivideByPOT(product, shift);
      const auto offset = static_cast<std::int64_t>(scaled) + zero;
      if (offset < std::numeric_limits<std::int32_t>::min() ||
          offset > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error("zero-point addition exceeds INT32");
      }
      std::cout << std::min<std::int64_t>(127, std::max<std::int64_t>(-128, offset)) << '\n';
    } else {
      return 1;
    }
  }
  return std::cin.eof() ? 0 : 1;
}
