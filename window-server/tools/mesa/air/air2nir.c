/*
 * Copyright © 2026 Darwin on redwood contributors
 * SPDX-License-Identifier: MIT
 */

/*
 * air2nir: the frontend from the command line, as spirv2nir is for SPIR-V.
 *
 *   air2nir FILE [FUNCTION]          translate and print the NIR
 *   air2nir --ll FILE [FUNCTION]     print what the bitcode reader read, as LLVM assembly
 *   air2nir --list FILE.metallib     the functions of a container
 *   air2nir --report FILE...         one line per module, for tools/air/corpus.py
 *
 * FILE is a .metallib (then FUNCTION picks one; without it, every function)
 * or one module's bitcode, wrapped or not. --report forks for every module,
 * so that an assertion in one is a line in the report and not the end of it.
 */

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include "nir.h"
#include "nir_air.h"
#include "air_module.h"
#include "util/ralloc.h"

static const nir_shader_compiler_options default_options = {
   .lower_fdiv = false,
};

static void *
map_file(const char *path, size_t *size)
{
   int fd = open(path, O_RDONLY);
   if (fd < 0) {
      fprintf(stderr, "air2nir: %s: %s\n", path, strerror(errno));
      return NULL;
   }
   struct stat st;
   fstat(fd, &st);
   *size = st.st_size;
   void *p = mmap(NULL, st.st_size ? st.st_size : 1, PROT_READ, MAP_PRIVATE, fd, 0);
   close(fd);
   return p == MAP_FAILED ? NULL : p;
}

struct module_ref {
   const char *name;
   int container_type;
   const void *bitcode;
   size_t size;
};

/* every module of FILE (one, unless it is a container) */
static unsigned
modules_of(void *ctx, const char *path, const char *only, struct module_ref **out)
{
   size_t size;
   const uint8_t *data = map_file(path, &size);
   if (!data)
      return 0;
   if (size >= 4 && !memcmp(data, "MTLB", 4)) {
      struct air_metallib lib;
      const char *err;
      if (!air_metallib_parse(ctx, data, size, &lib, &err)) {
         fprintf(stderr, "air2nir: %s: %s\n", path, err);
         return 0;
      }
      *out = rzalloc_array(ctx, struct module_ref, lib.num_functions ? lib.num_functions : 1);
      unsigned n = 0;
      for (unsigned i = 0; i < lib.num_functions; i++) {
         if (only && strcmp(only, lib.functions[i].name))
            continue;
         (*out)[n++] = (struct module_ref) {
            lib.functions[i].name, lib.functions[i].type,
            lib.functions[i].bitcode, lib.functions[i].bitcode_size,
         };
      }
      return n;
   }
   *out = rzalloc(ctx, struct module_ref);
   (*out)[0] = (struct module_ref) { path, -1, data, size };
   return 1;
}

static const char *
stage_name(int stage)
{
   switch (stage) {
   case MESA_SHADER_VERTEX: return "vertex";
   case MESA_SHADER_FRAGMENT: return "fragment";
   case MESA_SHADER_COMPUTE: return "kernel";
   case MESA_SHADER_NONE: return "visible";
   default: return "unknown";
   }
}

static void
one_line(const char *s, FILE *fp)
{
   for (; *s; s++)
      fputc(*s == '\n' || *s == '\t' ? ' ' : *s, fp);
}

/* tools/air/corpus.py reads these lines: tab-separated key=value */
static void
report_one(const struct module_ref *mod)
{
   void *ctx = ralloc_context(NULL);
   const char *err = NULL, *calls = NULL;

   printf("module=%s\tcontainer=%s", mod->name,
          mod->container_type >= 0 ? air_metallib_type_name(mod->container_type) : "-");

   struct air_module *m = air_module_read(ctx, mod->bitcode, mod->size, &err);
   if (!m) {
      printf("\treader=fail\terror=");
      one_line(err, stdout);
      printf("\n");
      ralloc_free(ctx);
      return;
   }
   printf("\treader=ok");

   int stage = air_module_stage(ctx, mod->bitcode, mod->size, &calls, &err);
   printf("\tstage=%s\tcalls=", stage_name(stage));
   for (const char *c = calls; c && *c; c += strlen(c) + 1)
      printf("%s%s", c == calls ? "" : ",", c);

   /* Function constants are values the API supplies at pipeline creation. The
    * report has no API, so it compiles with every one set to zero and says so:
    * a module that needs values is not a module the translator cannot handle.
    */
   struct air_constant_value *fc;
   unsigned num_fc = air_module_function_constants(ctx, mod->bitcode, mod->size, &fc);
   struct air_to_nir_options options = { .constants = fc, .num_constants = num_fc };
   if (num_fc)
      printf("\tfc=%u", num_fc);

   struct air_shader_info info;
   nir_shader *nir = air_to_nir(ctx, mod->bitcode, mod->size, NULL, &options, &default_options,
                                &info, &err);
   if (!nir) {
      printf("\ttranslate=fail\terror=");
      one_line(err, stdout);
      printf("\n");
      ralloc_free(ctx);
      return;
   }
   nir_validate_shader(nir, "after air_to_nir");
   unsigned instrs = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block)
            instrs++;
      }
   }
   printf("\ttranslate=ok\tnir_instrs=%u\n", instrs);
   ralloc_free(nir);
   ralloc_free(ctx);
}

int
main(int argc, char **argv)
{
   enum { TRANSLATE, LL, LIST, REPORT } mode = TRANSLATE;
   int i = 1;
   if (i < argc && !strcmp(argv[i], "--ll"))
      mode = LL, i++;
   else if (i < argc && !strcmp(argv[i], "--list"))
      mode = LIST, i++;
   else if (i < argc && !strcmp(argv[i], "--report"))
      mode = REPORT, i++;
   if (i >= argc) {
      fprintf(stderr, "usage: air2nir [--ll | --list | --report] FILE [FUNCTION]\n");
      return 2;
   }

   glsl_type_singleton_init_or_ref();
   void *ctx = ralloc_context(NULL);
   int status = 0;

   if (mode == REPORT) {
      setvbuf(stdout, NULL, _IOLBF, 0);
      for (; i < argc; i++) {
         struct module_ref *mods;
         unsigned n = modules_of(ctx, argv[i], NULL, &mods);
         for (unsigned k = 0; k < n; k++) {
            fflush(stdout);
            pid_t pid = fork();
            if (pid == 0) {
               alarm(60);
               report_one(&mods[k]);
               fflush(stdout);
               _exit(0);
            }
            int ws;
            waitpid(pid, &ws, 0);
            if (!WIFEXITED(ws) || WEXITSTATUS(ws) != 0)
               printf("module=%s\tcrash=%s %d\n", mods[k].name,
                      WIFSIGNALED(ws) ? "signal" : "exit",
                      WIFSIGNALED(ws) ? WTERMSIG(ws) : WEXITSTATUS(ws));
         }
      }
      ralloc_free(ctx);
      glsl_type_singleton_decref();
      return 0;
   }

   struct module_ref *mods;
   unsigned n = modules_of(ctx, argv[i], i + 1 < argc ? argv[i + 1] : NULL, &mods);
   if (n == 0)
      return 1;

   for (unsigned k = 0; k < n; k++) {
      const char *err = NULL;
      if (mode == LIST) {
         printf("%-60s %-12s %zu\n", mods[k].name,
                air_metallib_type_name(mods[k].container_type), mods[k].size);
         continue;
      }
      if (mode == LL) {
         struct air_module *m = air_module_read(ctx, mods[k].bitcode, mods[k].size, &err);
         if (!m) {
            fprintf(stderr, "air2nir: %s: %s\n", mods[k].name, err);
            status = 1;
            continue;
         }
         air_module_print(m, stdout);
         continue;
      }
      struct air_constant_value *fc;
      unsigned num_fc = air_module_function_constants(ctx, mods[k].bitcode, mods[k].size, &fc);
      struct air_to_nir_options options = { .constants = fc, .num_constants = num_fc };
      struct air_shader_info info;
      nir_shader *nir = air_to_nir(ctx, mods[k].bitcode, mods[k].size, NULL, &options,
                                   &default_options, &info, &err);
      if (!nir) {
         fprintf(stderr, "air2nir: %s: %s\n", mods[k].name, err);
         status = 1;
         continue;
      }
      nir_print_shader(nir, stdout);
   }

   ralloc_free(ctx);
   glsl_type_singleton_decref();
   return status;
}
