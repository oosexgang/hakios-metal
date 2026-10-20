/*
 * Copyright © 2026 Darwin on redwood contributors
 * SPDX-License-Identifier: MIT
 */

/* Shared between air_to_nir.c and air_intrinsics.c. */

#ifndef AIR_PRIVATE_H
#define AIR_PRIVATE_H

#include "air_module.h"
#include "nir_air.h"

#include "nir.h"
#include "nir_builder.h"

/* Where a pointer points. AIR's address spaces say which kind of memory;
 * which buffer, and the byte offset into it, the translator tracks itself
 * (docs/air.md section 3): no pointer in the corpus is ever stored to
 * memory or cast to an integer, so a pointer never has to become a number.
 */
enum air_ptr_class {
   AIR_PTR_NONE,
   AIR_PTR_UBO,           /* a constant buffer: .binding is its index */
   AIR_PTR_SSBO,          /* a device buffer */
   AIR_PTR_SHARED,        /* threadgroup memory */
   AIR_PTR_SCRATCH,       /* an alloca, or a program-scope variable */
   AIR_PTR_CONSTANT_DATA, /* a constant global: nir->constant_data */
   AIR_PTR_FUNCTION_CONSTANT, /* a function constant: .binding is its index */
   AIR_PTR_TEXTURE,       /* a texture handle; offset unused */
   AIR_PTR_SAMPLER,       /* a sampler handle */
};

struct air_ptr {
   enum air_ptr_class class;
   unsigned binding;
   bool is_image;                    /* a texture the shader writes: an image, not a view */
   nir_def *offset;                  /* bytes, 32-bit; NULL means zero */
   const struct air_type *pointee;   /* for sizes, and for loads without a type */
};

enum air_val_kind {
   AIR_VAL_NONE,
   AIR_VAL_SSA,   /* a scalar or vector */
   AIR_VAL_AGG,   /* a struct or array, as a tree of values */
   AIR_VAL_PTR,
};

struct air_val {
   enum air_val_kind kind;
   nir_def *def;
   struct air_ptr ptr;
   unsigned num_elems;
   struct air_val **elems;
};

struct air_ctx {
   void *mem_ctx;
   const struct air_module *m;
   const struct air_function *fn;
   nir_shader *nir;
   nir_builder b;

   const struct air_to_nir_options *options;
   struct air_shader_info *info;

   /* translated values, keyed on const struct air_value * */
   struct hash_table *values;
   /* air_block * -> nir_block * */
   struct hash_table *blocks;
   /* a cursor in each block, before its terminator, where phi stores go */
   struct hash_table *phi_cursors;
   /* phi -> nir register decl */
   struct hash_table *phi_regs;

   /* A call of another function in the module becomes a nir_call, and
    * nir_inline_functions folds it in afterwards. These two carry the state
    * that needs: the functions already translated, so a callee shared by two
    * call sites is translated once, and -- while a callee is being
    * translated -- the deref its `ret` has to store through, since a NIR call
    * returns nothing and the convention is an out-parameter at index 0.
    * ret_deref is NULL while the entry point is being translated, which is how
    * emit_return tells the two apart.
    */
   struct hash_table *fns;
   nir_deref_instr *ret_deref;

   /* What each module global resolves to, decided once (plan_globals) and
    * turned into a value in whichever function needs it (materialise_globals).
    * The two are separate because the deciding half has side effects that must
    * happen once -- it appends inline samplers to info->resources and lays out
    * shared and scratch memory -- while the making half emits an nir_def for
    * the byte offset, and a def belongs to the impl that created it. Copying
    * the entry point's values into a callee instead left the callee using the
    * caller's defs, which is a cross-function def/use: nir_repair_ssa then
    * asks the wrong function for its dominance metadata and asserts.
    */
   struct air_global_plan {
      enum air_ptr_class class;
      unsigned binding;
      unsigned offset;      /* bytes, when has_offset */
      bool has_offset;
      bool skip;            /* llvm.* lists, and anything that failed */
   } *global_plan;

   /* what the entry point's parameters are */
   unsigned num_params;
   struct air_val **params;

   /* outputs, in the order of the returned struct's fields */
   unsigned num_outputs;
   struct {
      unsigned location;      /* gl_varying_slot or gl_frag_result */
      unsigned index;         /* dual-source index */
      nir_variable *var;
   } *outputs;

   unsigned scratch_size;
   unsigned shared_size;

   /* module-scope variables, by their air_global */
   struct hash_table *globals;

   char error[256];
};

bool air_error(struct air_ctx *c, const char *fmt, ...) PRINTFLIKE(2, 3);

/* value bookkeeping */
struct air_val *air_val_ssa(struct air_ctx *c, nir_def *def);
struct air_val *air_val_ptr(struct air_ctx *c, struct air_ptr ptr);
struct air_val *air_get(struct air_ctx *c, const struct air_value *v);
void air_set(struct air_ctx *c, const struct air_value *v, struct air_val *val);

/* an SSA value, with aggregates and pointers refused */
nir_def *air_get_ssa(struct air_ctx *c, const struct air_value *v);

/* the NIR type of an AIR type: bit size and components */
unsigned air_nir_bit_size(struct air_ctx *c, const struct air_type *t);
unsigned air_nir_components(const struct air_type *t);

/* memory, by the pointer's class */
nir_def *air_load_ptr(struct air_ctx *c, struct air_ptr ptr, const struct air_type *type,
                      unsigned align);
bool air_store_ptr(struct air_ctx *c, struct air_ptr ptr, nir_def *value, unsigned align);
struct air_ptr air_ptr_add(struct air_ctx *c, struct air_ptr ptr, nir_def *bytes);

/* air.* and llvm.* calls; the result is set on the instruction by the callee */
bool air_translate_call(struct air_ctx *c, const struct air_inst *in, const char *name);
/* a call of a function defined in this module, folded in by nir_inline_functions */
bool air_call_in_module(struct air_ctx *c, const struct air_inst *in, const char *name);

#endif /* AIR_PRIVATE_H */
