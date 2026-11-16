file(REMOVE_RECURSE
  "lib/libascendc_kernels_npu_vec.pdb"
  "lib/libascendc_kernels_npu_vec.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/ascendc_kernels_npu_vec.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
