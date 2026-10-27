# Metal on the Adreno: the plan, and where it stands

Read `docs/ru/metal.md` (why Metal is not simply missing, and what is on each side of the wall) and `docs/air.md` (the
shader compiler) first. This file is the order of work, with how each step is checked and where.

## The three parts

| Part | What it is | State |
|---|---|---|
| **A. The shader compiler** | Apple's shader bitcode (AIR, in a `.metallib`) into Mesa's NIR, so that ir3 (the compiler behind the OpenGL ES that already runs) makes Adreno code | largely done: 321 of the system's 456 shader modules reach ir3, 54 have run on the phone's GPU (`docs/air.md`, `docs/air-report.md`) |
| **B. The device** | what Metal loads: a bundle with a subclass of Metal's private `_MTLDevice`, and the objects it hands out (buffers, libraries, pipeline states, queues, command buffers, encoders: 53 classes and about 1200 methods in Apple's own) | step 1 is written (below); nothing past it |
| **C. The kernel side** | submitting work to the GPU for a Metal client, with memory and fences | the GPU already has a user client for Mesa (`drivers/gpu/hakios.gpu.cpp`, `HakiosGPUUserClient`); it is not connected to B |

What "real Metal" would be, and why games are out of reach, is argued in `docs/ru/metal.md` section 9.4: that has not
changed. What this plan aims at is the part that is within reach: **Metal clients of the system itself** (the window
server's compositor, CoreImage, the camera's viewfinder) running on the Adreno through B and C, with shaders from A.

## The order, and how each step is checked

| # | Step | Checked by | Where |
|---|---|---|---|
| 1 | **A device that exists.** `metal=1` makes the (so far empty) `IOAcceleratorES` service name `HakiosMetal`; the bundle makes `HakiosMetalDevice`, a `_MTLDevice` subclass; `MTLCreateSystemDefaultDevice()` returns it | `metal_test` (`RC_JOBS=rc.jobs.metal`) prints the device and `HakiosMetal:` lines say which messages Metal sent that the device did not answer | QEMU, no GPU needed |
| 2 | **Learn what Metal asks first**, from step 1's "not implemented" lines: properties, family support, queue, buffer | the log, in order | QEMU |
| 3 | **Buffers and textures** over the GPU memory the kernel hands out (`hakios.gpu.cpp`: `kMsmGemNew`), `IOSurface`-backed | a program writes a buffer and reads it back | phone |
| 4 | **A library and a compute pipeline**: `newLibraryWithData:` takes a `.metallib`, the function goes through A, the kernel runs | the CPU reference in `tools/air` and then the phone | phone |
| 5 | **Command queue, command buffer, compute encoder, commit and wait**, on fences | a compute kernel dispatched through the Metal API gives the numbers `tools/air/cpuref.py` expects | phone |
| 6 | **Render pipeline and render command encoder** for the three vertex and 271 fragment shaders A already handles | a triangle through Metal | phone |
| 7 | **Present**: a drawable, and the frame reaching the screen without being dropped | the frame counters (below) | phone |

Without a phone, steps 1 and 2 are all that can be done and checked. Steps 3 to 7 each need the GPU.

## Step 1: what is in the tree

- `xnu/files/iokit/Kernel/MiniIOMobileFramebuffer.cpp`: with boot-arg `metal=1` the `IOAcceleratorES` service gets the
  properties `MetalPluginName` = `HakiosMetal` and `MetalPluginClassName` = `HakiosMetalDevice`. Without the argument
  nothing changes: **the properties stay off by default, on purpose**. With them a program that asks for Metal gets a
  device, and CoreAnimation would pick Metal for the screen; until that device can draw, the screen without it is the
  working one.
- `userland/src/HakiosMetal.m`, `userland/src/HakiosMetal.plist`: the bundle, built into `/redwood/lib/HakiosMetal.bundle` and
  linked at `/System/Library/Extensions/HakiosMetal.bundle` (`userland/build.sh`).
- `userland/src/metal_test.m` and `userland/rc.jobs.metal`: the check that already existed.

## What steps 1 and 2 found (QEMU, 27 October 2026, `metal=1`, a clean image)

Run in a loop of rebuild-image, boot, read the log; each pass gave the next requirement:

1. **Metal found the service, loaded the bundle and made the class**: `HakiosMetal: the class HakiosMetalDevice is registered`
   in `backboardd` and `mediaserverd`, then `a device for the accelerator port N (made)`: `-initWithAcceleratorPort:` was
   called and `_MTLDevice`'s own `-init` returned an object. So the discovery chain of `docs/ru/metal.md` section 1 is as described.
2. `MTLAddDevice` then **asserted that the class conforms to the protocol `MTLDevice`**, and, once it did, **to the private
   protocol `MTLDeviceSPI`** (neither is declared by `_MTLDevice` itself, so the class declares them).
3. With both declared, the first messages sent to the device, in this order: **`-featureProfile`**, **`-maxBufferLength`**,
   and then `backboardd` tries to **create a command queue** (`[Metal] Failed to create MTLCommandQueue`) and exits. The
   window server restarts and fails the same way (launchd: "service has crashed N times in a row"). That is what a Metal
   device that exists and cannot do anything does to the system, and the reason `metal=1` is off by default.

4. **A command queue** is now made (`HakiosMetalQueue`, a subclass of `_MTLCommandQueue`; it queues nothing). Metal then
   sends it `-setCompletionQueue:`, which is not in `MTLCommandQueue` but in the private protocol `MTLCommandQueueSPI`: found
   by taking every protocol of the process whose name begins with `MTLCommandQueue`. The window server then asks for
   `-setGPUPriority:` and `-newDepthStencilStateWithDescriptor:`.
5. **The objects a device hands out** are stub classes made from a table in `HakiosMetal.m` (depth-stencil state, sampler
   state, texture, render and compute pipeline states, event, fence): a subclass of Metal's private base where the system has
   one, the protocol family declared, every other message logged once and answered with zero. Textures and events have no
   private base class on this system (NSObject is used).
6. **The next wall**, with those in place: CoreAnimation's Metal renderer stops with
   `Assertion failed: (metallib_url != nil), function MetalContext, file ogl-metal.mm, line 3308.` It needs the URL of a
   `.metallib` (the system has `AppleMetalGLRenderer.bundle/default.metallib`, 274 shader modules, which the AIR work translates)
   and will then ask the device for a library from it, which is where the shader compiler (part A) is first needed for real.
   Where CoreAnimation takes that URL from is not known yet; the log shows the device was asked for nothing before it.

7. **Who is shown the device**, then, is a boot-arg level (the plug-in reads `sysctl kern.bootargs`): `metal=1` only the program
   `metal_test`, `metal=2` every process but the window server, `metal=3` every process. With the window server shown the device
   it stops on the missing metallib (QuartzCore.framework has a `default.pipelinelib` that names a `default.metallib` which is
   not in this image), so it cannot run on Metal here, and it has to be left on its OpenGL ES path. With `mediaserverd` shown it, it
   ends and starts again. At `metal=1` the window server, SpringBoard and mediaserverd were each logged as not shown the device
   and none of them ended in a QEMU run (the result of `metal_test` itself is not in this file yet).

So the road from here is the library (step 4 of the order below: `newLibraryWithURL:` / `…Data:` through the AIR compiler), then
buffers and textures with real memory, and the command encoders; none of that can be tried without the GPU.

## Frame skips

A dropped frame is not a Metal problem, and Metal would not cure it: the screen of this port is drawn today by
CoreAnimation's OpenGL ES renderer through Mesa (`gpu=1`), and its pace depends on the submit path, the fences and the
present. What to measure first, on the phone, is already counted: `gGPUStat` in `drivers/gpu/hakios.gpu.cpp` (submissions,
waits, what each cost, every five seconds). Whether the framebuffer stand-in (`MiniIOMobileFramebuffer.cpp`) counts
frames and drops is not known to me: it records when the first frame was shown and keeps a small copy of frames
(`record()`), and a counter of presented and dropped frames would be the first thing to add. A frame skip that is not seen
in those numbers is in the compositor, not the driver. This needs a stable GPU run on the phone first; the
`gpu=1` hang being bisected (see `docs/driver-review/README.md`) comes before any of it.

## What is not claimed

Nothing here draws. No Metal client has used this device for anything: the first one that tried (the window server) asked
for a command queue and stopped. Everything above was seen in QEMU; none of it was run on the phone.
