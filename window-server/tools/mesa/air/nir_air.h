/*
 * Copyright © 2026 Darwin on redwood contributors
 * SPDX-License-Identifier: MIT
 */

/*
 * AIR -> NIR: Metal shaders (the LLVM bitcode inside a .metallib) into NIR.
 * See docs/air.md of Darwin on redwood for the design and the mapping
 * tables this follows.
 */

#ifndef NIR_AIR_H
#define NIR_AIR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "compiler/shader_enums.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nir_shader nir_shader;
typedef struct nir_shader_compiler_options nir_shader_compiler_options;

/* -- the .metallib container ---------------------------------------------- */

struct air_metallib_function {
   const char *name;
   int type;               /* the container's TYPE tag, see air_metallib_type_name() */
   const void *bitcode;    /* the module, still in its 0x0B17C0DE wrapper */
   size_t bitcode_size;
};

struct air_metallib {
   unsigned version_major, version_minor;
   unsigned num_functions;
   struct air_metallib_function *functions;
};

bool air_metallib_parse(void *mem_ctx, const void *data, size_t size,
                        struct air_metallib *lib, const char **error);
const char *air_metallib_type_name(int type);

/* -- what the translated shader expects of the runtime ---------------------- */

enum air_resource_kind {
   AIR_RESOURCE_CONSTANT_BUFFER,    /* UBO at .index */
   AIR_RESOURCE_DEVICE_BUFFER,      /* SSBO at .index */
   AIR_RESOURCE_THREADGROUP_BUFFER, /* sized at dispatch */
   AIR_RESOURCE_TEXTURE,            /* sampler view at .index */
   AIR_RESOURCE_IMAGE,              /* storage image at .index */
   AIR_RESOURCE_SAMPLER,            /* sampler state at .index */
};

struct air_resource {
   enum air_resource_kind kind;
   unsigned index;   /* Metal's [[buffer(n)]], [[texture(n)]], [[sampler(n)]] */
   unsigned size;    /* air.buffer_size when given, else 0 */
   bool read, write;
   unsigned param;   /* the entry point's parameter; ~0u for an inline sampler */
   const char *name; /* air.arg_name */
   const char *type_name;

   /* A sampler the shader declares itself (Metal's constexpr sampler) rather
    * than one the API binds: the runtime has to create it from this state
    * word and bind it at .index, which the frontend assigns after the API's
    * samplers. The word's layout is not decoded yet -- see docs/air.md.
    */
   bool inline_sampler;
   uint64_t sampler_state;
};

struct air_varying {
   const char *name;  /* the AIR name that links vertex to fragment */
   unsigned slot;     /* gl_varying_slot assigned by the translator */
   unsigned components;
   bool flat, centroid, sample, no_perspective;
};

struct air_shader_info {
   gl_shader_stage stage;     /* MESA_SHADER_NONE for an air.visible function */
   const char *entry_point;
   unsigned num_resources;
   struct air_resource *resources;
   unsigned num_varyings;     /* vertex outputs or fragment inputs */
   struct air_varying *varyings;
   unsigned num_render_targets;
   unsigned threadgroup_bytes; /* static threadgroup memory */
};

/* -- the translator --------------------------------------------------------- */

struct air_constant_value {
   unsigned index;     /* the function constant's index */
   uint64_t bits;      /* its value, as the bits of its type */
};

struct air_to_nir_options {
   const struct air_constant_value *constants;
   unsigned num_constants;

   /* Translate an air.visible library function by generating an entry point
    * that calls it: argument i read from device buffer 0 at byte 16*i, the
    * return value written to device buffer 1 at offset 0, one thread. Metal's
    * own way of using these functions is to stitch them into a caller at
    * pipeline time (air.stitching_info in their metadata), which this frontend
    * does not implement; the wrapper exists so that a library function's body
    * can be run and compared at all. A shader produced this way is not an
    * entry point of the corpus and is counted separately.
    */
   bool library_wrapper;
};

/* Translates one AIR module (one function of a .metallib). entry_point may
 * be NULL: the module's own entry point is taken. On failure returns NULL
 * and sets *error (allocated under mem_ctx) to "<category>: <what>", where
 * the category is one of
 *
 *    reader       the bitcode could not be read
 *    stage        no entry point, or not one this translator knows
 *    interface    an entry-point metadata key not handled yet
 *    instruction  an LLVM instruction or form not handled yet
 *    intrinsic    an air.* (or llvm.*) function not handled yet
 *
 * so the corpus report can count them. *info, when non-NULL, describes what
 * the shader expects to be bound (allocated under mem_ctx).
 */
nir_shader *air_to_nir(void *mem_ctx, const void *bitcode, size_t size,
                       const char *entry_point,
                       const struct air_to_nir_options *options,
                       const nir_shader_compiler_options *nir_options,
                       struct air_shader_info *info, const char **error);

/* The function constants the module declares, with .bits zero: a caller
 * fills in the values it will specialize with. Returns how many there are.
 */
unsigned air_module_function_constants(void *mem_ctx, const void *bitcode, size_t size,
                                       struct air_constant_value **out);

/* What the module is, without translating it: the stage from its metadata
 * (MESA_SHADER_NONE for visible functions, -1 when unknown) and the air.*
 * functions it calls, NUL-separated in *calls.
 */
int air_module_stage(void *mem_ctx, const void *bitcode, size_t size, const char **calls,
                     const char **error);

#ifdef __cplusplus
}
#endif

#endif /* NIR_AIR_H */
