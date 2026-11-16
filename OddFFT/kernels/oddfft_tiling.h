// kernels/oddfft_tiling.h  (OddFFT)
//
// This lives in kernels/ and not in common/ for a build reason, and the name
// is prefixed for a second one.
//
// The ascendc_library() target that compiles the kernels gets NO extra include
// directories -- only the toolkit's own. A kernel can therefore include a
// header from its own directory and nothing else. An earlier version of this
// file sat in common/ and every kernel failed on the device with
// "'tiling.h' file not found", while the local syntax check passed because it
// was handing the compiler -I common. A check that is more permissive than the
// real build is not a check; scripts/chk_kernels.sh now mirrors the real
// include set.
//
// The `oddfft_` prefix is because the CANN toolkit ships tiling/tiling_api.h
// and a bare "tiling.h" on an include path is an accident waiting to happen.
//
// The tile count of one stage, defined ONCE and shared by the kernels and the
// host.
//
// WHY THIS IS SHARED AND WHY IT IS A MACRO
// The host picks blockDim from the tile count, and each kernel block picks its
// own tiles from the same count. If the two ever disagreed the tiles would be
// mis-assigned: some elements computed twice, others not at all, and the
// result would be plausibly wrong rather than obviously broken. So there must
// be exactly one definition.
//
// It is a macro and not an inline function because the same text has to be
// usable inside __aicore__ device code and in ordinary host code, and a plain
// inline function is not callable from device code without the __aicore__
// qualifier, which in turn is not valid in the host translation units. A macro
// sidesteps the question entirely: both sides expand the same arithmetic.
//
// THE RULE: j runs over [0, t). Tiles are `tile` long, except that a trailing
// remainder below 8 is ABSORBED into the last tile instead of becoming a tile
// of its own.
//
// The absorb was originally a CORRECTNESS measure: a tile shorter than 8
// floats would need a DataCopyPad below the 32-byte granule, which had not
// been shown to work, and rounding the length UP is not an option because the
// write would overrun into another tile's live data.
//
// Sub-granule DataCopyPad has since been measured and it DOES work (see
// host/plan.h). The absorb stays anyway, now as an OPTIMISATION rather than a
// requirement: a transfer costs about the same whatever its length, so one
// longer tile beats two, one of which is tiny.
// Consequence: a tile length is in [8, tile + 7] whenever t >= 8, and equals
// t itself when t < 8 -- which is now a case the device can serve.
#pragma once
#include <cstdint>

#define ODDFFT_SHIFT_COUNT(t, tile) \
    ( ((t) <= (tile)) ? 1u \
      : ( ((t) % (tile) >= 8u) ? ((t) / (tile) + 1u) : ((t) / (tile)) ) )

// The length of tile index s, given the tile count from ODDFFT_SHIFT_COUNT.
#define ODDFFT_TILE_VALID(t, tile, s, nShift) \
    ( ((s) + 1u == (nShift)) ? ((t) - (s) * (tile)) : (tile) )
