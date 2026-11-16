file(REMOVE_RECURSE
  "lib/libascendc_kernels_vec_test.pdb"
  "lib/libascendc_kernels_vec_test.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/ascendc_kernels_vec_test.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
