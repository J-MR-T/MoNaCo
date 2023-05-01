add_mlir_dialect_library(MLIRAMD64
        AMD64Dialect.cpp
        AMD64Ops.cpp

        ADDITIONAL_HEADER_DIRS
		${MONACO_SOURCE_DIR}/include/AMD64

        DEPENDS
        MLIRAMD64OpsIncGen
        MLIRAMD64OpsEnumsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
        MLIRInferTypeOpInterface
    )

