set(sources
        src/csr_graph.cu
        src/ggc_rt.cu
)

hip_add_library(galois_gpu
  ${sources}
  HIPCC_OPTIONS -D_FORCE_INLINES
)

set_target_properties (galois_gpu PROPERTIES 
  INTERFACE_POSITION_INDEPENDENT_CODE On 
  POSITION_INDEPENDENT_CODE On
)
