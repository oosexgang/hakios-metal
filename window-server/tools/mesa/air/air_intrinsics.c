/*
 * Copyright © 2026 Darwin on redwood contributors
 * SPDX-License-Identifier: MIT
 */

/*
 * The air.* functions. There are 125 distinct declarations in the corpus but
 * their names are <operation>.<type suffix>, so this is a table of about 50
 * operations in four families (docs/metal.md 9.2). The suffix is not trusted
 * for anything: the types come from the call's own LLVM types, and the suffix
 * is only what picks the handler.
 *
 * Where Metal's definition of an operation is not certain from its name --
 * air.fast_exp could be exp or exp2 -- the choice is marked UNVERIFIED and
 * has to be settled against the CPU reference (docs/air.md section 4) before
 * any shader using it is called correct.
 */

#include "air_private.h"

#include <string.h>

#include "nir_builtin_builder.h"
#include "util/u_math.h"

/* the dot-separated parts of an air.* name, after "air." */
#define AIR_MAX_PARTS 8
struct air_name {
   unsigned num;
   char part[AIR_MAX_PARTS][32];
};

static bool
split_name(const char *name, struct air_name *out)
{
   out->num = 0;
   const char *p = name + 4; /* past "air." */
   while (*p && out->num < AIR_MAX_PARTS) {
      const char *dot = strchr(p, '.');
      size_t len = dot ? (size_t)(dot - p) : strlen(p);
      if (len >= sizeof(out->part[0]))
         return false;
      memcpy(out->part[out->num], p, len);
      out->part[out->num][len] = 0;
      out->num++;
      if (!dot)
         return true;
      p = dot + 1;
   }
   return out->num < AIR_MAX_PARTS;
}

static bool
is_signed_part(const char *s)
{
   return !strcmp(s, "s");
}

/* -- the families ------------------------------------------------------- */

static bool
math_op(struct air_ctx *c, const struct air_inst *in, const struct air_name *n, bool *handled)
{
   const char *op = n->part[0];
   unsigned argc = in->num_operands - 1;

   /* the families this file does not handle are named before the arguments
    * are touched, so that their error says so rather than complaining about
    * a texture handle used as a number
    */
   if (strstr(op, "texture") || strstr(op, "_depth_") || !strncmp(op, "sample", 6) ||
       !strncmp(op, "gather", 6) || !strncmp(op, "read_", 5) || !strncmp(op, "write_", 6)) {
      *handled = false;
      return true;
   }

   nir_def *a[4] = { 0 };
   for (unsigned i = 0; i < argc && i < 4; i++)
      if (!(a[i] = air_get_ssa(c, in->operands[1 + i])))
         return false;

   bool is_float = air_type_is_float(air_type_scalar(in->base.type));
   /* the sign of an integer operation is the last-but-one part: air.min.s.i32 */
   bool is_signed = n->num >= 3 ? is_signed_part(n->part[1]) : true;
   nir_def *r = NULL;
   *handled = true;

   /* fast math is what every one of these means: Metal's "fast" library */
   bool was_exact = c->b.exact;
   c->b.exact = false;

   if (!strcmp(op, "fast_fabs") || !strcmp(op, "fabs")) {
      r = nir_fabs(&c->b, a[0]);
   } else if (!strcmp(op, "fast_sqrt") || !strcmp(op, "sqrt")) {
      r = nir_fsqrt(&c->b, a[0]);
   } else if (!strcmp(op, "fast_rsqrt") || !strcmp(op, "rsqrt")) {
      r = nir_frsq(&c->b, a[0]);
   } else if (!strcmp(op, "fast_floor") || !strcmp(op, "floor")) {
      r = nir_ffloor(&c->b, a[0]);
   } else if (!strcmp(op, "fast_ceil") || !strcmp(op, "ceil")) {
      r = nir_fceil(&c->b, a[0]);
   } else if (!strcmp(op, "fast_round") || !strcmp(op, "round")) {
      /* Metal's round() goes away from zero at .5; rint() is the even one */
      r = nir_fround_even(&c->b, a[0]); /* UNVERIFIED: round vs rint */
   } else if (!strcmp(op, "fast_trunc") || !strcmp(op, "trunc")) {
      r = nir_ftrunc(&c->b, a[0]);
   } else if (!strcmp(op, "fast_fract") || !strcmp(op, "fract")) {
      r = nir_ffract(&c->b, a[0]);
   } else if (!strcmp(op, "fast_sin") || !strcmp(op, "sin")) {
      r = nir_fsin(&c->b, a[0]);
   } else if (!strcmp(op, "fast_cos") || !strcmp(op, "cos")) {
      r = nir_fcos(&c->b, a[0]);
   } else if (!strcmp(op, "fast_tan") || !strcmp(op, "tan")) {
      r = nir_fdiv(&c->b, nir_fsin(&c->b, a[0]), nir_fcos(&c->b, a[0]));
   } else if (!strcmp(op, "fast_atan2") || !strcmp(op, "atan2")) {
      r = nir_atan2(&c->b, a[0], a[1]);
   } else if (!strcmp(op, "fast_exp") || !strcmp(op, "exp")) {
      /* UNVERIFIED: exp or exp2. Taken as the natural exp, which is what
       * Metal's exp() is; exp2 has its own name in the language.
       */
      r = nir_fexp2(&c->b, nir_fmul_imm(&c->b, a[0], M_LOG2E));
   } else if (!strcmp(op, "fast_exp2") || !strcmp(op, "exp2")) {
      r = nir_fexp2(&c->b, a[0]);
   } else if (!strcmp(op, "fast_log") || !strcmp(op, "log")) {
      r = nir_fmul_imm(&c->b, nir_flog2(&c->b, a[0]), 1.0 / M_LOG2E);
   } else if (!strcmp(op, "fast_log2") || !strcmp(op, "log2")) {
      r = nir_flog2(&c->b, a[0]);
   } else if (!strcmp(op, "fast_pow") || !strcmp(op, "pow")) {
      r = nir_fpow(&c->b, a[0], a[1]);
   } else if (!strcmp(op, "fast_divide") || !strcmp(op, "divide")) {
      r = nir_fdiv(&c->b, a[0], a[1]);
   } else if (!strcmp(op, "saturate") || !strcmp(op, "fast_saturate")) {
      r = nir_fsat(&c->b, a[0]);
   } else if (!strcmp(op, "sign")) {
      r = is_float ? nir_fsign(&c->b, a[0]) : nir_isign(&c->b, a[0]);
   } else if (!strcmp(op, "mix")) {
      /* mix(x, y, a) = x + a * (y - x), and that is what Metal defines */
      r = nir_fadd(&c->b, a[0],
                   nir_fmul(&c->b, a[2], nir_fsub(&c->b, a[1], a[0])));
   } else if (!strcmp(op, "dot")) {
      r = nir_fdot(&c->b, a[0], a[1]);
   } else if (!strcmp(op, "fast_fmin") || !strcmp(op, "fmin") || !strcmp(op, "min")) {
      r = is_float ? nir_fmin(&c->b, a[0], a[1])
                   : (is_signed ? nir_imin(&c->b, a[0], a[1]) : nir_umin(&c->b, a[0], a[1]));
   } else if (!strcmp(op, "fast_fmax") || !strcmp(op, "fmax") || !strcmp(op, "max")) {
      r = is_float ? nir_fmax(&c->b, a[0], a[1])
                   : (is_signed ? nir_imax(&c->b, a[0], a[1]) : nir_umax(&c->b, a[0], a[1]));
   } else if (!strcmp(op, "fast_clamp") || !strcmp(op, "clamp")) {
      if (is_float)
         r = nir_fmin(&c->b, nir_fmax(&c->b, a[0], a[1]), a[2]);
      else if (is_signed)
         r = nir_imin(&c->b, nir_imax(&c->b, a[0], a[1]), a[2]);
      else
         r = nir_umin(&c->b, nir_umax(&c->b, a[0], a[1]), a[2]);
   } else if (!strcmp(op, "fast_ldexp") || !strcmp(op, "ldexp")) {
      r = nir_ldexp(&c->b, a[0], nir_i2i32(&c->b, a[1]));
   } else if (!strcmp(op, "fast_fma") || !strcmp(op, "fma")) {
      r = nir_ffma(&c->b, a[0], a[1], a[2]);
   } else if (!strcmp(op, "all")) {
      r = nir_ball(&c->b, a[0]);
   } else if (!strcmp(op, "any")) {
      r = nir_bany(&c->b, a[0]);
   } else if (!strcmp(op, "convert")) {
      /* air.convert.<dst sign>.<dst type>.<src sign>.<src type> */
      if (n->num < 5) {
         c->b.exact = was_exact;
         return air_error(c, "intrinsic: %s (a convert without both types)", op);
      }
      const struct air_type *dt = in->base.type;
      const struct air_type *st = in->operands[1]->type;
      unsigned bits = air_nir_bit_size(c, dt);
      bool dst_float = n->part[1][0] == 'f';
      bool src_float = n->part[3][0] == 'f';
      bool src_signed = n->part[3][0] == 's';
      nir_def *x = a[0];
      if (!src_float && air_type_bit_size(air_type_scalar(st)) == 1)
         x = src_signed ? nir_ineg(&c->b, nir_b2i32(&c->b, x)) : nir_b2i32(&c->b, x);
      if (dst_float && src_float)
         r = nir_f2fN(&c->b, x, bits);
      else if (dst_float)
         r = src_signed ? nir_i2fN(&c->b, x, bits) : nir_u2fN(&c->b, x, bits);
      else if (src_float)
         r = n->part[1][0] == 's' ? nir_f2iN(&c->b, x, bits) : nir_f2uN(&c->b, x, bits);
      else
         r = src_signed ? nir_i2iN(&c->b, x, bits) : nir_u2uN(&c->b, x, bits);
   } else {
      *handled = false;
   }

   c->b.exact = was_exact;
   if (!*handled)
      return true;
   if (!r)
      return false;
   air_set(c, &in->base, air_val_ssa(c, r));
   return true;
}

/* -- atomics ------------------------------------------------------------ */

static bool
atomic_op(struct air_ctx *c, const struct air_inst *in, const struct air_name *n)
{
   /* air.atomic.{global,local}.<op>[.<sign>].<type>(ptr, value?, order,
    * scope, volatile)
    */
   if (n->num < 3)
      return air_error(c, "intrinsic: an atomic without a name");
   const char *where = n->part[1];
   const char *op = n->part[2];
   bool is_signed = n->num >= 5 ? is_signed_part(n->part[3]) : true;

   struct air_val *pv = air_get(c, in->operands[1]);
   if (!pv)
      return false;
   if (pv->kind != AIR_VAL_PTR)
      return air_error(c, "intrinsic: an atomic on something that is not a pointer");
   struct air_ptr p = pv->ptr;

   bool shared = !strcmp(where, "local");
   if (shared != (p.class == AIR_PTR_SHARED))
      return air_error(c, "intrinsic: air.atomic.%s on %s memory", where,
                       p.class == AIR_PTR_SHARED ? "threadgroup" : "device");

   nir_def *off = p.offset ? p.offset : nir_imm_int(&c->b, 0);

   if (!strcmp(op, "load")) {
      nir_def *r = air_load_ptr(c, p, in->base.type, 0);
      if (!r)
         return false;
      air_set(c, &in->base, air_val_ssa(c, r));
      return true;
   }
   if (!strcmp(op, "store")) {
      nir_def *v = air_get_ssa(c, in->operands[2]);
      return v && air_store_ptr(c, p, v, 0);
   }

   nir_atomic_op aop;
   if (!strcmp(op, "add")) aop = nir_atomic_op_iadd;
   else if (!strcmp(op, "min")) aop = is_signed ? nir_atomic_op_imin : nir_atomic_op_umin;
   else if (!strcmp(op, "max")) aop = is_signed ? nir_atomic_op_imax : nir_atomic_op_umax;
   else if (!strcmp(op, "and")) aop = nir_atomic_op_iand;
   else if (!strcmp(op, "or")) aop = nir_atomic_op_ior;
   else if (!strcmp(op, "xor")) aop = nir_atomic_op_ixor;
   else if (!strcmp(op, "xchg")) aop = nir_atomic_op_xchg;
   else return air_error(c, "intrinsic: atomic %s", op);

   nir_def *v = air_get_ssa(c, in->operands[2]);
   if (!v)
      return false;
   nir_def *r;
   if (shared) {
      r = nir_shared_atomic(&c->b, v->bit_size, off, v, .atomic_op = aop);
   } else if (p.class == AIR_PTR_SSBO) {
      r = nir_ssbo_atomic(&c->b, v->bit_size, nir_imm_int(&c->b, p.binding), off, v,
                          .atomic_op = aop);
   } else {
      return air_error(c, "intrinsic: an atomic on a pointer that is not a buffer");
   }
   if (in->base.type && in->base.type->kind != AIR_TYPE_VOID)
      air_set(c, &in->base, air_val_ssa(c, r));
   return true;
}

/* -- barriers ----------------------------------------------------------- */

static bool
barrier_op(struct air_ctx *c, const struct air_inst *in)
{
   /* air.wg.barrier(i32 flags, i32 scope): the flags say which memory is
    * ordered. 2 is threadgroup memory, 1 device, 4 texture; the corpus only
    * ever passes 2 with scope 1 (the threadgroup).
    */
   uint64_t flags = 2;
   const struct air_constant *k =
      in->num_operands > 1 ? air_value_as_constant(in->operands[1]) : NULL;
   if (k && k->ckind == AIR_CONST_INT)
      flags = k->u;

   nir_variable_mode modes = 0;
   if (flags & 1)
      modes |= nir_var_mem_ssbo | nir_var_mem_global;
   if (flags & 2)
      modes |= nir_var_mem_shared;
   if (flags & 4)
      modes |= nir_var_image;
   if (!modes)
      modes = nir_var_mem_shared;

   nir_barrier(&c->b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP, .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = modes);
   return true;
}

/* -- llvm.* ------------------------------------------------------------- */

static bool
llvm_op(struct air_ctx *c, const struct air_inst *in, const char *name)
{
   if (!strncmp(name, "llvm.lifetime.", 14))
      return true; /* scope markers: nothing to emit */
   if (!strncmp(name, "llvm.memcpy.", 12)) {
      /* a copy between two of our pointers, of a constant length */
      struct air_val *dst = air_get(c, in->operands[1]);
      struct air_val *src = air_get(c, in->operands[2]);
      const struct air_constant *len = air_value_as_constant(in->operands[3]);
      if (!dst || !src)
         return false;
      if (dst->kind != AIR_VAL_PTR || src->kind != AIR_VAL_PTR)
         return air_error(c, "intrinsic: memcpy that is not between pointers");
      if (!len || len->ckind != AIR_CONST_INT)
         return air_error(c, "intrinsic: memcpy of a run-time length");
      /* in 4-byte pieces, which is what every memcpy in the corpus is a
       * multiple of (they copy float vectors out of constant buffers)
       */
      if (len->u % 4)
         return air_error(c, "intrinsic: memcpy of %u bytes", (unsigned)len->u);
      for (uint64_t off = 0; off < len->u; off += 4) {
         struct air_ptr s = air_ptr_add(c, src->ptr, nir_imm_int(&c->b, off));
         struct air_ptr d = air_ptr_add(c, dst->ptr, nir_imm_int(&c->b, off));
         static const struct air_type u32 = { .kind = AIR_TYPE_INT, .bits = 32 };
         nir_def *v = air_load_ptr(c, s, &u32, 4);
         if (!v || !air_store_ptr(c, d, v, 4))
            return false;
      }
      return true;
   }
   if (!strncmp(name, "llvm.memset.", 12)) {
      struct air_val *dst = air_get(c, in->operands[1]);
      nir_def *byte = air_get_ssa(c, in->operands[2]);
      const struct air_constant *len = air_value_as_constant(in->operands[3]);
      if (!dst || !byte)
         return false;
      if (dst->kind != AIR_VAL_PTR)
         return air_error(c, "intrinsic: memset of something that is not a pointer");
      if (!len || len->ckind != AIR_CONST_INT || len->u % 4)
         return air_error(c, "intrinsic: memset of a length that is not a multiple of 4");
      if (!nir_src_is_const(nir_src_for_ssa(byte)))
         return air_error(c, "intrinsic: memset of a run-time byte");
      uint32_t b = nir_src_as_uint(nir_src_for_ssa(byte)) & 0xff;
      uint32_t word = b | b << 8 | b << 16 | b << 24;
      for (uint64_t off = 0; off < len->u; off += 4) {
         struct air_ptr d = air_ptr_add(c, dst->ptr, nir_imm_int(&c->b, off));
         if (!air_store_ptr(c, d, nir_imm_int(&c->b, word), 4))
            return false;
      }
      return true;
   }
   if (!strcmp(name, "llvm.trap"))
      return air_error(c, "intrinsic: llvm.trap");
   return air_error(c, "intrinsic: %s", name);
}

/* -- the dispatcher ----------------------------------------------------- */

bool
air_translate_call(struct air_ctx *c, const struct air_inst *in, const char *name)
{
   if (!strncmp(name, "llvm.", 5))
      return llvm_op(c, in, name);

   if (strncmp(name, "air.", 4)) {
      /* a call of another function in the module: inlining them is the
       * translator's job and is not done yet (no system shader needs it:
       * every one is a single function after Apple's own inliner)
       */
      return air_error(c, "intrinsic: a call of %s, a function in the module", name);
   }

   struct air_name n;
   if (!split_name(name, &n))
      return air_error(c, "intrinsic: %s (a name this cannot take apart)", name);

   if (n.num >= 2 && !strcmp(n.part[0], "wg") && !strcmp(n.part[1], "barrier"))
      return barrier_op(c, in);
   if (!strcmp(n.part[0], "atomic"))
      return atomic_op(c, in, &n);

   bool handled = false;
   if (!math_op(c, in, &n, &handled))
      return false;
   if (handled)
      return true;

   /* textures: the largest family, and M3's whole subject. Named here so the
    * corpus report counts them as one thing rather than as unknown calls.
    */
   const char *op = n.part[0];
   if (strstr(op, "texture") || strstr(op, "_depth_") || !strncmp(op, "sample", 6) ||
       !strncmp(op, "gather", 6) || !strncmp(op, "read_", 5) || !strncmp(op, "write_", 6))
      return air_error(c, "intrinsic: %s (textures, not yet)", name);

   return air_error(c, "intrinsic: %s", name);
}
