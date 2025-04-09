set(LLVM_TARGET_DEFINITIONS UnrankedToRankedMemref.td)

mlir_tablegen(UnrankedToRankedMemrefConversionPassIncGen.h.inc
  -gen-pass-decls)

add_public_tablegen_target(UnrankedToRankedMemrefConversionPassIncGen)

add_triton_library(TritonUnrankedToRankedMemref
  UnrankedToRankedMemrefPass.cpp

  DEPENDS
  UnrankedToRankedMemrefConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRPass
  MLIRMemRefDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonTransforms
  TritonSharedAnalysis
)

mlir_check_all_link_libraries(TritonUnrankedToRankedMemref)