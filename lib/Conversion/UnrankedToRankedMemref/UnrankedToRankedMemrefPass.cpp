#include "triton-shared/Conversion/UnrankedToRankedMemref/UnrankedToRankedMemref.h"
#include "triton-shared/Analysis/PtrAnalysis.h"
#include "UnrankedToRankedMemrefConversionPassIncGen.h.inc"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "unranked-to-ranked-memref"

using namespace mlir;

namespace mlir {
namespace triton {

class UnrankedToRankedMemrefPass
    : public PassWrapper<UnrankedToRankedMemrefPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnrankedToRankedMemrefPass)

  StringRef getArgument() const override { return "unranked-to-ranked-memref"; }
  
  StringRef getDescription() const override { 
    return "Convert unranked memrefs to ranked memrefs"; 
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }

void runOnOperation() override {
  auto module = getOperation();
  MLIRContext *context = &getContext();
  OpBuilder builder(context);

  module.walk([&](func::FuncOp funcOp) {
    auto funcType = funcOp.getFunctionType();
    SmallVector<Type, 4> newInputs;
    bool hasUnrankedMemRefs = false;
    
    // Check input arguments for unranked memrefs
    for (Type inputType : funcType.getInputs()) {
      if (auto unrankedMemRef = dyn_cast<UnrankedMemRefType>(inputType)) {
        // Use a reasonable default for function arguments (here we use a 1D dynamic shape)
        Type elementType = unrankedMemRef.getElementType();
        auto rankedMemRef = MemRefType::get({ShapedType::kDynamic}, elementType);
        newInputs.push_back(rankedMemRef);
        hasUnrankedMemRefs = true;
      } else {
        newInputs.push_back(inputType);
      }
    }

    // Update function signature
    if (hasUnrankedMemRefs) {
      auto newFuncType = FunctionType::get(context, newInputs, funcType.getResults());
      funcOp.setType(newFuncType);
      
      // Update block argument types
      for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
        funcOp.getArgument(i).setType(newInputs[i]);
      }
    }
  });

  // Handle operation results
  module.walk([&](Operation *op) {
    if (op->getNumResults() == 0)
      return;

    for (Value result : op->getResults()) {
      Type type = result.getType();
      if (auto unrankedMemRef = dyn_cast<UnrankedMemRefType>(type)) {
        // Get the element type
        Type elementType = unrankedMemRef.getElementType();

        // Try to infer the shape from the operation's context
        SmallVector<int64_t> shape;
        if (auto castOp = dyn_cast<memref::ReinterpretCastOp>(op)) {
          // Get shape from reinterpret_cast
          for (auto size : castOp.getMixedSizes()) {
            if (auto attr = dyn_cast<IntegerAttr>(size.dyn_cast<Attribute>())) {
              shape.push_back(attr.getInt());
            } else {
              shape.push_back(ShapedType::kDynamic);
            }
          }
        } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
          // Get shape from load indices
          shape.push_back(1); // Scalar load
        } else {
          // Default to dynamic shape if we can't infer
          shape.push_back(ShapedType::kDynamic);
        }

        // Create ranked memref type
        auto rankedMemRef = MemRefType::get(shape, elementType);

        // Replace the unranked memref with ranked memref
        result.setType(rankedMemRef);
      }
    }
  });
}
};

std::unique_ptr<Pass> createUnrankedToRankedMemrefPass() {
  return std::make_unique<UnrankedToRankedMemrefPass>();
}

void registerUnrankedToRankedMemrefPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return createUnrankedToRankedMemrefPass();
  });
}

} // namespace triton
} // namespace mlir
