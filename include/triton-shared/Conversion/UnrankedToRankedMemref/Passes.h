#ifndef TRITON_SHARED_CONVERSION_UNRANKEDTORANKEDMEMREF_PASSES_H
#define TRITON_SHARED_CONVERSION_UNRANKEDTORANKEDMEMREF_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

void registerUnrankedToRankedMemrefPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_SHARED_CONVERSION_UNRANKEDTORANKEDMEMREF_PASSES_H 