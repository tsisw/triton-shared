#ifndef TRITON_SHARED_CONVERSION_UNRANKEDTORANKEDMEMREF_UNRANKEDTORANKEDMEMREF_TD
#define TRITON_SHARED_CONVERSION_UNRANKEDTORANKEDMEMREF_UNRANKEDTORANKEDMEMREF_TD

include "mlir/Pass/PassBase.td"

def UnrankedToRankedMemref : Pass<"unranked-to-ranked-memref"> {
  let summary = "Convert unranked memrefs to ranked memrefs";
  let description = [{
    This pass converts unranked memrefs to ranked memrefs by inferring shapes
    from the operation context.
  }];
  let dependentDialects = ["mlir::memref::MemRefDialect"];
}

#endif 