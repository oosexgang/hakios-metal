/*
 * Copyright © 2026 Darwin on redwood contributors
 * SPDX-License-Identifier: MIT
 */

/*
 * air_run: draw with a Metal shader and read the pixels back.
 *
 * This is the half of M1 that compiling cannot give. The shader comes out of a
 * .metallib, through air_to_nir, into a Gallium pipe_shader_state as
 * PIPE_SHADER_IR_NIR -- which both drivers of interest accept natively: ir3
 * is a NIR compiler, and softpipe takes NIR through nir_to_tgsi. One quad is
 * drawn into a 32-bit-float render target and one pixel is printed, so the
 * result is a number and not a picture.
 *
 * On this machine the screen is softpipe, so the whole path can be run with no
 * GPU at all. On the phone the same source wants the freedreno screen that
 * tools/mesa/freedreno-darwin.patch already creates inside the port's Mesa
 * (osmesa_create_gpu_screen), and nothing else about it changes.
 *
 * usage: air_run FILE[.metallib] [FUNCTION]
 */

#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include "nir.h"
#include "nir_builder.h"
#include "nir_air.h"
#include "air_module.h"

#include "pipe/p_context.h"
#include "pipe/p_defines.h"
#include "pipe/p_screen.h"
#include "pipe/p_shader_tokens.h"
#include "pipe/p_state.h"
#include "target-helpers/inline_sw_helper.h"
#include "null/null_sw_winsys.h"
#ifdef GALLIUM_FREEDRENO
#include "freedreno/drm/freedreno_drm_public.h"
#include "util/driconf.h"
#include "util/xmlconfig.h"
#include <xf86drm.h>
#endif
#include "util/format/u_format.h"
#include "util/u_inlines.h"
#include "util/u_sampler.h"
#include "util/u_debug.h"
#include "util/u_memory.h"

#ifdef __APPLE__
#include <mach-o/dyld.h>
#include <signal.h>
/* <sys/ucontext.h>, not <ucontext.h>: the latter refuses to be included
 * without _XOPEN_SOURCE on the iOS SDK, and userland/src/inject.c takes the
 * same route for the same reason.
 */
#include <sys/ucontext.h>
#endif

#define WIDTH 4
#define HEIGHT 4

/* The texture bound for every sampled texture the shader declares: four by
 * four, 32-bit floats, each texel a number that says which texel it is. The
 * sampler is nearest with no mip filter and clamped addressing, and the
 * coordinate the vertex shader hands over is the centre of texel (1, 1), so
 * a correct sample is an exact equality and not a tolerance -- no filtering
 * weights, no wrap arithmetic, nothing to round.
 *
 * This verifies the texture operations whose sampler the API chooses (nine
 * modules of the corpus) and those that take no sampler at all (twenty-nine,
 * which fetch by integer coordinate). It cannot verify the two whose sampler
 * the shader declares itself: that needs Metal's 64-bit sampler word decoded,
 * which it is not.
 */
#define TEX_SIZE 4
#define TEX_X 1
#define TEX_Y 1

/* Slot is in the value on purpose. Binding one texture to every slot makes a
 * shader that reads the wrong slot return the right texel, which is a test
 * passing for the wrong reason -- the third time that shape of mistake has
 * turned up here, after the 2D texture bound for a 3D sampler and the colour
 * attachment nothing stored to. Looked for deliberately this time rather than
 * tripped over: slot n adds a thousand times n, so a misbinding is visible.
 */
static void
texel_value(unsigned slot, unsigned x, unsigned y, unsigned z, float out[4])
{
   out[0] = 100.0f + x + 1000.0f * slot;
   out[1] = 200.0f + y;
   out[2] = 300.0f + x + y + z;
   out[3] = 400.0f;
}

#ifdef __APPLE__
/*
 * A program that dies on this port leaves nothing behind: the kernel says
 * "Failed to send exception EXC_CORPSE_NOTIFY" and that is the end of it. So
 * print the fault before dying. These two functions are userland/src/inject.c's
 * say_the_fault()/watch_for_faults() (branch sileo2, 6683521), with the slide
 * taken from this program's own image rather than from the shared cache, since
 * the pc of interest is in air_run or in libOSMesa and not in iOS.
 */
static void
say_the_fault(int signal_number, siginfo_t *info, void *context_in)
{
   ucontext_t *context = context_in;
   uintptr_t slide = (uintptr_t)_dyld_get_image_vmaddr_slide(0);
   uintptr_t pc = 0, lr = 0, fp = 0;

   if (context != NULL && context->uc_mcontext != NULL) {
      pc = (uintptr_t)context->uc_mcontext->__ss.__pc;
      lr = (uintptr_t)context->uc_mcontext->__ss.__lr;
      fp = (uintptr_t)context->uc_mcontext->__ss.__fp;
   }
   /* who called, by the frame pointers: on arm64 each frame holds the previous
    * at [fp] and its return address at [fp + 8]
    */
   char frames[8 * 20 + 1] = "";
   size_t used = 0;
   for (unsigned i = 0; i < 8 && fp != 0 && (fp & 7) == 0; i++) {
      uintptr_t next = ((uintptr_t *)fp)[0], ret = ((uintptr_t *)fp)[1];
      if (ret == 0)
         break;
      used += (size_t)snprintf(frames + used, sizeof(frames) - used, " %#lx", ret - slide);
      fp = next > fp ? next : 0;   /* the chain only ever goes up the stack */
   }
   fprintf(stderr, "air_run: signal %d at %p, pc %#lx, lr %#lx, called from%s (unslid)\n",
           signal_number, info != NULL ? info->si_addr : NULL, pc - slide, lr - slide,
           frames);
   fflush(stderr);
   signal(signal_number, SIG_DFL);
   raise(signal_number);
}

static void
watch_for_faults(void)
{
   struct sigaction action;

   memset(&action, 0, sizeof(action));
   action.sa_sigaction = say_the_fault;
   action.sa_flags = SA_SIGINFO | SA_RESETHAND;
   sigaction(SIGSEGV, &action, NULL);
   sigaction(SIGBUS, &action, NULL);
}
#else
static void watch_for_faults(void) {}
#endif

/* the pattern cpuref.py and air2nir --eval fill their buffers with, so that
 * all three agree on what the shader was given
 */
static uint8_t
pattern_byte(unsigned i)
{
   return (i * 7 + 13) & 0xff;
}

/* Fill a buffer the shader is about to read.
 *
 * AIR_RUN_BUFFER=FILE puts that file's bytes there instead of the pattern,
 * which is what lets a test choose the arguments an operation is called on.
 * The reason it has to be a buffer and not a constant in the shader: ir3
 * folds, so a shader that computes air.fast_round(0.5) from an immediate is
 * answered by the compiler and the Adreno never runs the instruction -- the
 * test would pass without the hardware having been asked. A value loaded from
 * a buffer cannot be folded, and the harness knows what it put there.
 *
 * Returns how many bytes came from the file, so the caller can say so rather
 * than claim a fill it did not do. A file shorter than the buffer leaves the
 * rest as the pattern.
 */
static unsigned
fill_buffer(uint8_t *bits, unsigned bytes)
{
   for (unsigned k = 0; k < bytes; k++)
      bits[k] = pattern_byte(k);

   const char *path = getenv("AIR_RUN_BUFFER");
   if (!path || !*path)
      return 0;

   FILE *f = fopen(path, "rb");
   if (!f) {
      fprintf(stderr, "air_run: AIR_RUN_BUFFER=%s: %s\n", path, strerror(errno));
      return 0;
   }
   size_t got = fread(bits, 1, bytes, f);
   fclose(f);
   return got;
}

/* A vertex shader for the quad: it takes the position as attribute 0 and
 * passes it through. Written here rather than translated, because what is
 * under test is the fragment shader; when the .metallib has a vertex shader of
 * its own that is a separate run.
 */
static nir_shader *
quad_vertex_shader(const nir_shader_compiler_options *options, unsigned num_varyings)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, options, "quad");
   nir_variable *in = nir_variable_create(b.shader, nir_var_shader_in,
                                          glsl_vec4_type(), "pos_in");
   in->data.location = VERT_ATTRIB_GENERIC0;
   in->data.driver_location = 0;
   nir_variable *out = nir_variable_create(b.shader, nir_var_shader_out,
                                           glsl_vec4_type(), "pos_out");
   out->data.location = VARYING_SLOT_POS;
   out->data.driver_location = 0;
   nir_store_var(&b, out, nir_load_var(&b, in), 0xf);

   /* One constant coordinate per varying the fragment shader reads: the
    * centre of texel (TEX_X, TEX_Y), so every fragment samples the same
    * texel and the expected value is a single known one.
    */
   float c = (TEX_X + 0.5f) / TEX_SIZE;
   float d = (TEX_Y + 0.5f) / TEX_SIZE;
   for (unsigned i = 0; i < num_varyings; i++) {
      nir_variable *v = nir_variable_create(b.shader, nir_var_shader_out,
                                            glsl_vec4_type(), "varying");
      v->data.location = VARYING_SLOT_VAR0 + i;
      v->data.driver_location = 1 + i;
      v->data.interpolation = INTERP_MODE_SMOOTH;
      nir_store_var(&b, v, nir_imm_vec4(&b, c, d, c, 1.0f), 0xf);
   }
   b.shader->num_inputs = 1;
   b.shader->num_outputs = 1 + num_varyings;
   return b.shader;
}

/* What a Gallium driver expects of a shader it is handed: NIR's own system
 * values lowered to the hardware's, and I/O as intrinsics rather than
 * variables. The same three passes are in tools/mesa/air-ir3.patch; when a
 * fourth place needs them they should become one function of the frontend.
 */
static int
type_size_vec4(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

static void
finalize_for_gallium(nir_shader *nir)
{
   NIR_PASS(_, nir, nir_lower_system_values);
   NIR_PASS(_, nir, nir_lower_compute_system_values, NULL);
   NIR_PASS(_, nir, nir_lower_io, nir_var_shader_in | nir_var_shader_out,
            type_size_vec4, 0);
}

/* -- compute ---------------------------------------------------------------
 *
 * A kernel writes through pointers instead of returning anything, so what is
 * read back is a device buffer. Two rules decide whether that readback is
 * evidence, and both come from the same mistake made three times with
 * textures: a test can pass for the wrong reason, and the way to stop it is
 * to make every input distinguishable so any confusion gives a wrong number
 * rather than the right one.
 *
 *   - the grid is more than one thread, and every element of the output is
 *     printed, not just the first. With one thread, or one element looked at,
 *     a wrong workgroup index or a wrong global id reads correct.
 *   - the input buffers vary with position (the same (i * 7 + 13) & 0xff
 *     pattern), so a thread reading the wrong offset produces a wrong value
 *     and not a plausible one.
 *
 * What this does NOT yet do is give atomics a test that can fail. Seventeen of
 * the forty kernels that translate use a global atomic, and a sum written by
 * threads that never raced looks exactly like a sum written by threads that
 * did. Detecting a lost update needs the expected total to depend on every
 * thread having contributed -- which needs a kernel whose arithmetic is known,
 * i.e. one written for the purpose rather than one of Apple's. Until that
 * exists, a kernel that uses atomics and produces sensible numbers here is
 * evidence that it ran, and no evidence at all about atomicity.
 */
#define GRID 4

static int
dispatch_kernel(struct pipe_screen *screen, const struct air_shader_info *info,
                nir_shader *cs, const char *name, const char *how)
{
   /* Metal's threadgroup size is a dispatch argument, not a shader constant,
    * so the frontend leaves it variable (docs/air.md section 7.4) and whoever
    * dispatches has to fix it. air_run is that whoever. It is not optional:
    * softpipe reads the block size from TGSI_PROPERTY_CS_FIXED_BLOCK_*, and
    * with a variable size those are zero, so it runs ZERO threads and the
    * output buffer comes back exactly as it went in. That looked like a
    * kernel that did nothing, and was only visible because the buffer was
    * filled with a distinctive pattern -- had it been zeroed, a kernel named
    * InitMinMax writing zeros would have been indistinguishable from this.
    */
   cs->info.workgroup_size[0] = GRID;
   cs->info.workgroup_size[1] = GRID;
   cs->info.workgroup_size[2] = 1;
   cs->info.workgroup_size_variable = false;

   finalize_for_gallium(cs);
   struct pipe_context *pipe = screen->context_create(screen, NULL, 0);
   if (!pipe) {
      fprintf(stderr, "air_run: no context for a kernel\n");
      return 1;
   }
   printf("air_run: %s on %s (%s), as a compute kernel\n", name,
          screen->get_name(screen), how);

   /* every buffer the kernel declares; the device ones are what it writes */
   struct pipe_resource *bufs[32] = { 0 };
   unsigned sizes[32] = { 0 };
   bool writable[32] = { false };
   unsigned num_ssbo = 0, num_written = 0;
   for (unsigned i = 0; i < info->num_resources; i++) {
      const struct air_resource *r = &info->resources[i];
      unsigned bytes = r->size ? r->size : 4096;
      if (r->index >= ARRAY_SIZE(bufs))
         continue;
      uint8_t *bits = calloc(1, bytes);
      unsigned from_file = fill_buffer(bits, bytes);
      if (from_file)
         printf("air_run: buffer at index %u, %u bytes, %u of them from "
                "AIR_RUN_BUFFER\n", r->index, bytes, from_file);

      if (r->kind == AIR_RESOURCE_CONSTANT_BUFFER) {
         struct pipe_resource *b = pipe_buffer_create(screen, PIPE_BIND_CONSTANT_BUFFER,
                                                      PIPE_USAGE_DEFAULT, bytes);
         pipe_buffer_write(pipe, b, 0, bytes, bits);
         pipe->set_constant_buffer(pipe, PIPE_SHADER_COMPUTE, r->index, false,
                                   &(struct pipe_constant_buffer){
                                      .buffer = b, .buffer_size = bytes });
      } else if (r->kind == AIR_RESOURCE_DEVICE_BUFFER) {
         bufs[r->index] = pipe_buffer_create(screen, PIPE_BIND_SHADER_BUFFER,
                                             PIPE_USAGE_STAGING, bytes);
         pipe_buffer_write(pipe, bufs[r->index], 0, bytes, bits);
         sizes[r->index] = bytes;
         writable[r->index] = r->write;
         num_ssbo = MAX2(num_ssbo, r->index + 1);
         num_written += r->write ? 1 : 0;
      } else if (r->kind == AIR_RESOURCE_TEXTURE || r->kind == AIR_RESOURCE_IMAGE) {
         free(bits);
         fprintf(stderr, "air_run: %s uses a texture or an image; this harness binds those "
                         "for a draw and not yet for a dispatch\n", name);
         return 1;
      }
      free(bits);
   }
   if (!num_written) {
      fprintf(stderr, "air_run: %s writes no device buffer, so a dispatch would leave "
                      "nothing to read back\n", name);
      return 1;
   }

   struct pipe_shader_buffer sb[32] = { 0 };
   unsigned mask = 0;
   for (unsigned i = 0; i < num_ssbo; i++) {
      sb[i].buffer = bufs[i];
      sb[i].buffer_size = sizes[i];
      if (writable[i])
         mask |= 1u << i;
   }
   pipe->set_shader_buffers(pipe, PIPE_SHADER_COMPUTE, 0, num_ssbo, sb, mask);

   void *cso = pipe->create_compute_state(pipe, &(struct pipe_compute_state){
      .ir_type = PIPE_SHADER_IR_NIR, .prog = cs,
      .static_shared_mem = cs->info.shared_size });
   if (!cso) {
      fprintf(stderr, "air_run: the driver would not make a compute state of it\n");
      return 1;
   }
   pipe->bind_compute_state(pipe, cso);

   /* more than one thread, and the whole grid, so that a wrong index shows */
   struct pipe_grid_info grid = {
      .work_dim = 3,
      .block = { GRID, GRID, 1 },
      .grid = { 1, 1, 1 },
   };
   printf("air_run: dispatching %ux%ux1 threads in one workgroup; %u device buffer%s bound, "
          "%u of them written, filled with (i * 7 + 13) & 0xff\n",
          GRID, GRID, num_ssbo, num_ssbo == 1 ? "" : "s", num_written);
   pipe->launch_grid(pipe, &grid);
   pipe->flush(pipe, NULL, 0);

   for (unsigned i = 0; i < num_ssbo; i++) {
      if (!writable[i])
         continue;
      struct pipe_transfer *xfer;
      uint32_t *p = pipe_buffer_map(pipe, bufs[i], PIPE_MAP_READ, &xfer);
      if (!p) {
         fprintf(stderr, "air_run: device buffer %u could not be read back\n", i);
         return 1;
      }
      unsigned words = MIN2(sizes[i] / 4, (unsigned)(GRID * GRID));
      printf("air_run: device buffer %u after the dispatch, first %u words:\n", i, words);
      for (unsigned w = 0; w < words; w++)
         printf("  [%2u] %11u  0x%08x\n", w, p[w], p[w]);
      pipe_buffer_unmap(pipe, xfer);
   }

   pipe->destroy(pipe);
   screen->destroy(screen);
   return 0;
}

int
main(int argc, char **argv)
{
   /* --disasm: ask ir3 to print the Adreno instructions it generates. The
    * screen's name proves fd_screen_create won; it does not prove that the
    * shader itself went through ir3 rather than some later software path, and
    * the four floats look identical either way. ir3's own disassembly in the
    * log is the positive proof. IR3_SHADER_DEBUG is read once, at compiler
    * creation, so it has to be set before the screen is made.
    */
   bool want_disasm = false;
   if (argc > 1 && !strcmp(argv[1], "--disasm")) {
      want_disasm = true;
      setenv("IR3_SHADER_DEBUG", "disasm", 1);
      argv++;
      argc--;
   }
   if (argc < 2) {
      fprintf(stderr, "usage: air_run [--disasm] FILE[.metallib] [FUNCTION]\n");
      return 2;
   }

   watch_for_faults();
   glsl_type_singleton_init_or_ref();
   void *ctx = ralloc_context(NULL);

   /* the module */
   FILE *f = fopen(argv[1], "rb");
   if (!f) {
      perror(argv[1]);
      return 1;
   }
   fseek(f, 0, SEEK_END);
   size_t size = ftell(f);
   fseek(f, 0, SEEK_SET);
   void *data = ralloc_size(ctx, size);
   if (fread(data, 1, size, f) != size) {
      fprintf(stderr, "air_run: short read of %s\n", argv[1]);
      return 1;
   }
   fclose(f);

   const void *bitcode = data;
   size_t bitcode_size = size;
   const char *want = argc > 2 ? argv[2] : NULL;
   const char *name = argv[1];
   if (size >= 4 && !memcmp(data, "MTLB", 4)) {
      struct air_metallib lib;
      const char *err;
      if (!air_metallib_parse(ctx, data, size, &lib, &err)) {
         fprintf(stderr, "air_run: %s: %s\n", argv[1], err);
         return 1;
      }
      unsigned pick = lib.num_functions;
      for (unsigned i = 0; i < lib.num_functions; i++) {
         if (!want || !strcmp(want, lib.functions[i].name)) {
            pick = i;
            break;
         }
      }
      if (pick == lib.num_functions) {
         fprintf(stderr, "air_run: no function %s\n", want ? want : "(any)");
         return 1;
      }
      bitcode = lib.functions[pick].bitcode;
      bitcode_size = lib.functions[pick].bitcode_size;
      name = lib.functions[pick].name;
   }

   /* The screen: the GPU's where the kernel has one to drive, softpipe
    * otherwise. The same source therefore runs on this machine (no msm
    * device, so it falls back and draws in software) and on the phone, where
    * it opens the Adreno exactly as the port's own Mesa does
    * (osmesa_create_gpu_screen in tools/mesa/freedreno-darwin.patch).
    * AIR_RUN_SOFTWARE=1 forces software even where a GPU is present.
    */
   struct pipe_screen *screen = NULL;
   const char *how = "software";
#ifdef GALLIUM_FREEDRENO
   if (!debug_get_bool_option("AIR_RUN_SOFTWARE", false)) {
      int fd = drmOpenWithType("msm", NULL, DRM_NODE_RENDER);
      if (fd >= 0) {
         /* The driver option cache has to be real, not a zeroed struct:
          * fd_screen_create passes it straight to driParseConfigFiles, which
          * dereferences it. A zeroed one died at address 0x10 on the phone --
          * the first thing this program ever did on a machine that has a GPU,
          * and invisible on any machine that has not. This is the sequence
          * osmesa_create_gpu_screen() uses in tools/mesa/freedreno-darwin.patch,
          * which is the one CoreAnimation goes through on this device daily.
          */
         static const driOptionDescription freedreno_options[] = {
#include "freedreno/driinfo_freedreno.h"
         };
         static driOptionCache info, cache;
         static struct pipe_screen_config config;

         driParseOptionInfo(&info, freedreno_options, ARRAY_SIZE(freedreno_options));
         driParseConfigFiles(&cache, &info, 0, "msm", NULL, NULL, NULL, 0, NULL, 0);
         config.options = &cache;
         config.options_info = &info;

         screen = fd_drm_screen_create_renderonly(fd, NULL, &config);
         if (screen)
            how = "the GPU, through the kernel's msm device";
         else
            fprintf(stderr, "air_run: the msm device opened but gave no screen\n");
      }
   }
#endif
   if (!screen)
      screen = sw_screen_create(null_sw_create());
   if (!screen) {
      fprintf(stderr, "air_run: no gallium screen\n");
      return 1;
   }
   printf("air_run: %s on %s (%s)\n", name, screen->get_name(screen), how);

   const void *options =
      screen->get_compiler_options(screen, PIPE_SHADER_IR_NIR, PIPE_SHADER_FRAGMENT);
   if (!options) {
      fprintf(stderr, "air_run: this screen takes no NIR\n");
      return 1;
   }

   /* the shader */
   struct air_constant_value *fc;
   unsigned num_fc = air_module_function_constants(ctx, bitcode, bitcode_size, &fc);
   struct air_to_nir_options air_options = { .constants = fc, .num_constants = num_fc };
   struct air_shader_info info;
   const char *err;
   nir_shader *fs = air_to_nir(ctx, bitcode, bitcode_size, NULL, &air_options, options,
                               &info, &err);
   if (!fs) {
      fprintf(stderr, "air_run: %s: %s\n", name, err);
      return 1;
   }
   if (info.stage == MESA_SHADER_COMPUTE)
      return dispatch_kernel(screen, &info, fs, name, how);
   if (info.stage != MESA_SHADER_FRAGMENT) {
      fprintf(stderr, "air_run: %s is a %s shader; only fragment shaders are drawn here\n",
              name, gl_shader_stage_name(info.stage));
      return 1;
   }
   /* The render target's format follows the shader's own output type. A Metal
    * fragment shader returning uint4 writes integers, and reading those back
    * through a float target gives the right bits read the wrong way -- which
    * looks exactly like a translation bug and is not one.
    */
   enum pipe_format rt_format = PIPE_FORMAT_R32G32B32A32_FLOAT;
   bool rt_is_float = true;
   /* Which attachment to read back: the lowest one the shader writes, not
    * target 0. clearFragment04 writes FRAG_RESULT_DATA1 alone, and target 0
    * then holds the clear colour -- a correct result that looks exactly like
    * a shader which stored nothing.
    */
   unsigned read_rt = ~0u;
   nir_foreach_variable_with_modes(var, fs, nir_var_shader_out) {
      if (var->data.location < FRAG_RESULT_DATA0)
         continue;
      unsigned idx = var->data.location - FRAG_RESULT_DATA0;
      if (idx >= read_rt)
         continue;
      read_rt = idx;
      enum glsl_base_type base = glsl_get_base_type(glsl_without_array(var->type));
      if (base != GLSL_TYPE_FLOAT && base != GLSL_TYPE_FLOAT16 &&
          base != GLSL_TYPE_DOUBLE) {
         /* unsigned, not signed: an LLVM integer carries no signedness, so
          * both this and tools/air/cpuref.py print the bits unsigned and the
          * two can be compared
          */
         rt_format = PIPE_FORMAT_R32G32B32A32_UINT;
         rt_is_float = false;
      } else {
         rt_format = PIPE_FORMAT_R32G32B32A32_FLOAT;
         rt_is_float = true;
      }
   }
   if (read_rt == ~0u) {
      fprintf(stderr, "air_run: %s writes no colour attachment\n", name);
      return 1;
   }

   /* A shader may declare a render target and never store to it:
    * texturedFragment3DDepth returns {undef, depth} and its only real output
    * is FRAG_RESULT_DEPTH. Reading the colour then gives the clear value,
    * which is a number that means nothing -- worse than no number, because it
    * looks like a result. This harness binds no depth attachment, so say so.
    */
   bool stores_colour = false;
   nir_foreach_function_impl(impl, fs) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_deref)
               continue;
            nir_deref_instr *d = nir_src_as_deref(intr->src[0]);
            if (d && d->deref_type == nir_deref_type_var &&
                d->var->data.location == (int)(FRAG_RESULT_DATA0 + read_rt))
               stores_colour = true;
         }
      }
   }
   if (!stores_colour) {
      fprintf(stderr, "air_run: %s stores nothing to a colour attachment -- its output is "
                      "depth or stencil, which this harness does not read back\n", name);
      return 1;
   }

   finalize_for_gallium(fs);

   struct pipe_context *pipe = screen->context_create(screen, NULL, 0);

   /* One target per render target the shader writes: a shader whose outputs
    * outnumber the framebuffer's attachments draws nothing at all, which
    * looks like a translation that produced no stores.
    */
   unsigned num_rt = MAX2(info.num_render_targets, read_rt + 1);
   if (num_rt > PIPE_MAX_COLOR_BUFS) {
      fprintf(stderr, "air_run: %u render targets, more than this can bind\n", num_rt);
      return 1;
   }
   struct pipe_resource templ = {
      .target = PIPE_TEXTURE_2D,
      .format = rt_format,
      .width0 = WIDTH,
      .height0 = HEIGHT,
      .depth0 = 1,
      .array_size = 1,
      .bind = PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW,
   };
   struct pipe_resource *target = NULL;
   struct pipe_framebuffer_state fb = {
      .width = WIDTH, .height = HEIGHT, .nr_cbufs = num_rt,
   };
   for (unsigned i = 0; i < num_rt; i++) {
      struct pipe_resource *res = screen->resource_create(screen, &templ);
      fb.cbufs[i] = pipe->create_surface(pipe, res,
         &(struct pipe_surface){ .format = templ.format });
      if (i == read_rt)
         target = res;
   }
   pipe->set_framebuffer_state(pipe, &fb);

   pipe->set_viewport_states(pipe, 0, 1, &(struct pipe_viewport_state){
      .scale = { WIDTH / 2.0f, HEIGHT / 2.0f, 1.0f },
      .translate = { WIDTH / 2.0f, HEIGHT / 2.0f, 0.0f },
   });

   void *blend = pipe->create_blend_state(pipe, &(struct pipe_blend_state){
      .rt[0].colormask = PIPE_MASK_RGBA,
   });
   pipe->bind_blend_state(pipe, blend);
   void *dsa = pipe->create_depth_stencil_alpha_state(pipe,
      &(struct pipe_depth_stencil_alpha_state){ 0 });
   pipe->bind_depth_stencil_alpha_state(pipe, dsa);
   void *rast = pipe->create_rasterizer_state(pipe, &(struct pipe_rasterizer_state){
      .cull_face = PIPE_FACE_NONE, .half_pixel_center = 1, .bottom_edge_rule = 1,
      .depth_clip_near = 1, .depth_clip_far = 1,
   });
   pipe->bind_rasterizer_state(pipe, rast);

   /* every constant buffer the shader reads, filled with the pattern */
   for (unsigned i = 0; i < info.num_resources; i++) {
      const struct air_resource *r = &info.resources[i];
      if (r->kind != AIR_RESOURCE_CONSTANT_BUFFER && r->kind != AIR_RESOURCE_DEVICE_BUFFER)
         continue;
      unsigned bytes = r->size ? r->size : 256;
      uint8_t *bits = malloc(bytes);
      unsigned from_file = fill_buffer(bits, bytes);
      struct pipe_resource *buf = pipe_buffer_create(screen, PIPE_BIND_CONSTANT_BUFFER,
                                                     PIPE_USAGE_DEFAULT, bytes);
      pipe_buffer_write(pipe, buf, 0, bytes, bits);
      pipe->set_constant_buffer(pipe, PIPE_SHADER_FRAGMENT, r->index, false,
                                &(struct pipe_constant_buffer){
                                   .buffer = buf, .buffer_size = bytes,
                                });
      if (from_file)
         printf("air_run: %s buffer at index %u, %u bytes, %u of them from "
                "AIR_RUN_BUFFER\n",
                r->kind == AIR_RESOURCE_CONSTANT_BUFFER ? "constant" : "device",
                r->index, bytes, from_file);
      else
         printf("air_run: %s buffer at index %u, %u bytes of (i * 7 + 13) & 0xff\n",
                r->kind == AIR_RESOURCE_CONSTANT_BUFFER ? "constant" : "device",
                r->index, bytes);
      free(bits);
   }

   /* every sampled texture the shader declares, and a sampler for it */
   unsigned num_tex = 0;
   for (unsigned i = 0; i < info.num_resources; i++)
      if (info.resources[i].kind == AIR_RESOURCE_TEXTURE)
         num_tex = MAX2(num_tex, info.resources[i].index + 1);
   if (num_tex) {
      /* The texture's format follows what the shader samples into: a Metal
       * texture2d<uint> read out of a float texture gives numbers that are
       * neither right nor obviously wrong.
       */
      enum pipe_format tex_format = PIPE_FORMAT_R32G32B32A32_FLOAT;
      bool tex_is_float = true;
      /* and the target, which has to match too: a 3D sample from a 2D
       * texture can come back right by accident, which is not evidence
       */
      enum pipe_texture_target target = PIPE_TEXTURE_2D;
      enum glsl_sampler_dim dim = GLSL_SAMPLER_DIM_2D;
      nir_foreach_function_impl(impl, fs) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_tex)
                  continue;
               nir_tex_instr *t = nir_instr_as_tex(instr);
               nir_alu_type base = nir_alu_type_get_base_type(t->dest_type);
               if (base == nir_type_int) {
                  tex_format = PIPE_FORMAT_R32G32B32A32_SINT;
                  tex_is_float = false;
               } else if (base == nir_type_uint) {
                  tex_format = PIPE_FORMAT_R32G32B32A32_UINT;
                  tex_is_float = false;
               }
               dim = t->sampler_dim;
               if (dim == GLSL_SAMPLER_DIM_3D)
                  target = PIPE_TEXTURE_3D;
               else if (dim == GLSL_SAMPLER_DIM_MS)
                  target = PIPE_TEXTURE_2D;   /* refused below */
               goto have_format;
            }
         }
      }
   have_format:;
      if (dim == GLSL_SAMPLER_DIM_MS) {
         fprintf(stderr, "air_run: %s samples a multisample texture; this harness binds "
                         "none, so it is not run rather than run against the wrong thing\n",
                 name);
         return 1;
      }
      if (dim != GLSL_SAMPLER_DIM_2D && dim != GLSL_SAMPLER_DIM_3D) {
         fprintf(stderr, "air_run: %s samples a texture of a kind this harness does not "
                         "bind (dim %u)\n", name, dim);
         return 1;
      }
      unsigned depth = target == PIPE_TEXTURE_3D ? TEX_SIZE : 1;
      unsigned tex_z = target == PIPE_TEXTURE_3D ? TEX_Y : 0;

      struct pipe_resource ttempl = {
         .target = target,
         .format = tex_format,
         .width0 = TEX_SIZE, .height0 = TEX_SIZE, .depth0 = depth, .array_size = 1,
         .bind = PIPE_BIND_SAMPLER_VIEW,
      };
      struct pipe_box box = { .width = TEX_SIZE, .height = TEX_SIZE, .depth = depth };
      struct pipe_resource *texes[PIPE_MAX_SAMPLERS] = { 0 };
      for (unsigned slot = 0; slot < num_tex && slot < PIPE_MAX_SAMPLERS; slot++) {
         float texels[TEX_SIZE * TEX_SIZE * TEX_SIZE * 4];
         int32_t itexels[TEX_SIZE * TEX_SIZE * TEX_SIZE * 4];
         for (unsigned z = 0; z < depth; z++) {
            for (unsigned y = 0; y < TEX_SIZE; y++) {
               for (unsigned x = 0; x < TEX_SIZE; x++) {
                  unsigned i = ((z * TEX_SIZE + y) * TEX_SIZE + x) * 4;
                  texel_value(slot, x, y, z, &texels[i]);
                  for (unsigned c = 0; c < 4; c++)
                     itexels[i + c] = (int32_t)texels[i + c];
               }
            }
         }
         texes[slot] = screen->resource_create(screen, &ttempl);
         pipe->texture_subdata(pipe, texes[slot], 0, PIPE_MAP_WRITE, &box,
                               tex_is_float ? (const void *)texels : (const void *)itexels,
                               TEX_SIZE * 4 * sizeof(float),
                               TEX_SIZE * TEX_SIZE * 4 * sizeof(float));
      }

      void *sampler = pipe->create_sampler_state(pipe, &(struct pipe_sampler_state){
         .wrap_s = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
         .wrap_t = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
         .wrap_r = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
         .min_img_filter = PIPE_TEX_FILTER_NEAREST,
         .mag_img_filter = PIPE_TEX_FILTER_NEAREST,
         .min_mip_filter = PIPE_TEX_MIPFILTER_NONE,
      });

      struct pipe_sampler_view *views[PIPE_MAX_SHADER_SAMPLER_VIEWS] = { 0 };
      void *samplers[PIPE_MAX_SAMPLERS] = { 0 };
      for (unsigned i = 0; i < num_tex && i < PIPE_MAX_SAMPLERS; i++) {
         struct pipe_sampler_view vtempl = { 0 };
         u_sampler_view_default_template(&vtempl, texes[i], texes[i]->format);
         views[i] = pipe->create_sampler_view(pipe, texes[i], &vtempl);
         samplers[i] = sampler;
      }
      pipe->set_sampler_views(pipe, PIPE_SHADER_FRAGMENT, 0, num_tex, 0, false, views);
      pipe->bind_sampler_states(pipe, PIPE_SHADER_FRAGMENT, 0, num_tex, samplers);

      float want[4];
      texel_value(0, TEX_X, TEX_Y, tex_z, want);
      printf("air_run: %u texture%s bound: %ux%ux%u %s, texel (x,y,z) of slot n = "
             "(100+x+1000n, 200+y, 300+x+y+z, 400), nearest, clamped\n",
             num_tex, num_tex == 1 ? "" : "s", TEX_SIZE, TEX_SIZE, depth,
             util_format_short_name(tex_format));
      printf("air_run: expected %.9g %.9g %.9g %.9g  (the centre of texel %u,%u,%u sampled "
             "once)\n", (double)want[0], (double)want[1], (double)want[2], (double)want[3],
             TEX_X, TEX_Y, tex_z);
   }

   /* the quad */
   static const float verts[] = {
      -1, -1, 0, 1,   1, -1, 0, 1,   -1, 1, 0, 1,   1, 1, 0, 1,
   };
   struct pipe_resource *vbuf = pipe_buffer_create(screen, PIPE_BIND_VERTEX_BUFFER,
                                                   PIPE_USAGE_DEFAULT, sizeof(verts));
   pipe_buffer_write(pipe, vbuf, 0, sizeof(verts), verts);
   pipe->set_vertex_buffers(pipe, 1, &(struct pipe_vertex_buffer){
      .buffer.resource = vbuf, .buffer_offset = 0,
   });
   void *velem = pipe->create_vertex_elements_state(pipe, 1,
      &(struct pipe_vertex_element){
         .src_offset = 0, .src_stride = 16, .vertex_buffer_index = 0,
         .src_format = PIPE_FORMAT_R32G32B32A32_FLOAT,
      });
   pipe->bind_vertex_elements_state(pipe, velem);

   nir_shader *vs = quad_vertex_shader(options, info.num_varyings);
   finalize_for_gallium(vs);
   void *vs_cso = pipe->create_vs_state(pipe, &(struct pipe_shader_state){
      .type = PIPE_SHADER_IR_NIR, .ir.nir = vs,
   });
   pipe->bind_vs_state(pipe, vs_cso);
   void *fs_cso = pipe->create_fs_state(pipe, &(struct pipe_shader_state){
      .type = PIPE_SHADER_IR_NIR, .ir.nir = fs,
   });
   pipe->bind_fs_state(pipe, fs_cso);

   union pipe_color_union clear = { .ui = { 0, 0, 0, 0 } };
   pipe->clear(pipe, PIPE_CLEAR_COLOR, NULL, &clear, 0.0, 0);

   struct pipe_draw_info draw = {
      .mode = MESA_PRIM_TRIANGLE_STRIP, .index_bounds_valid = true,
      .max_index = 3, .instance_count = 1,
   };
   struct pipe_draw_start_count_bias sc = { .start = 0, .count = 4 };
   pipe->draw_vbo(pipe, &draw, 0, NULL, &sc, 1);
   pipe->flush(pipe, NULL, 0);

   /* the pixel */
   struct pipe_transfer *xfer;
   void *pix = pipe->texture_map(pipe, target, 0, PIPE_MAP_READ,
                                 &(struct pipe_box){ .width = WIDTH, .height = HEIGHT,
                                                     .depth = 1 }, &xfer);
   if (!pix) {
      fprintf(stderr, "air_run: the render target could not be read back\n");
      return 1;
   }
   /* Named again here, beside the numbers: the line at the top says which
    * screen was created, this one says which screen the pixel just came out
    * of. Neither is proof that no stage in between went software -- --disasm
    * is, by making ir3 print what it compiled.
    */
   printf("air_run: read back from %s (%s)%s\n", screen->get_name(screen), how,
          want_disasm ? ", ir3's instructions are above" : "");
   printf("air_run: the pixel at (0, 0) of render target %u came back as:\n",
          read_rt);
   for (unsigned c = 0; c < 4; c++) {
      if (rt_is_float)
         printf("  %.9g\n", (double)((float *)pix)[c]);
      else
         printf("  %u\n", ((uint32_t *)pix)[c]);
   }
   pipe->texture_unmap(pipe, xfer);

   printf("air_run: drew %ux%u into %s, %u render target%s\n",
          WIDTH, HEIGHT, util_format_short_name(rt_format), num_rt,
          num_rt == 1 ? "" : "s");

   pipe->destroy(pipe);
   screen->destroy(screen);
   ralloc_free(ctx);
   glsl_type_singleton_decref();
   return 0;
}
