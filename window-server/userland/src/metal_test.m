/*
 * metal_test: what iOS here believes about Metal, and what it does with the answer.
 *
 * Nothing in iOS says a word when Metal is missing. A program asks for the system's device with
 * MTLCreateSystemDefaultDevice(); if there is none it gets nil, with no log line, no error and no
 * reason, and whatever the caller does next -- fall back, draw nothing, or stop at an assertion --
 * is all that is ever seen. This prints the whole chain instead, so that a failure somewhere above
 * can be told from a failure here.
 *
 * The chain is short, and every step of it is checked below:
 *
 *   1. Metal's MTLRegisterDevices() asks IOKit for the services of class "IOAcceleratorES" --
 *      that name, and no other, is the whole of how a GPU is found;
 *   2. of each such service it reads two string properties, "MetalPluginName" and
 *      "MetalPluginClassName";
 *   3. it loads /System/Library/Extensions/<MetalPluginName>.bundle and takes the class named by
 *      MetalPluginClassName, or the bundle's principal class;
 *   4. the class must be a subclass of Metal's own private _MTLDevice, or it is dropped;
 *   5. it is sent -initWithAcceleratorPort: with the service, and that object is the device.
 *      MTLCreateSystemDefaultDevice() hands out the one device of that list, under a dispatch_once:
 *      an empty list is an answer given once and kept, silently, for the life of the process.
 *
 * So the question "is there Metal" has exactly one answer, and no capability bit anywhere stands
 * in for it. What iOS publishes about graphics besides -- MobileGestalt's GraphicsFeatureSetClass
 * and GraphicsFeatureSetFallbacks, from the device tree's "graphics-featureset-class" and
 * "graphics-featureset-fallbacks" -- is read by CoreUI and CoreThemeDefinition to choose which
 * compiled artwork to take out of an asset catalog, not by anything that decides how to draw; it
 * is printed here to show that it says the iPhone X's "APPLE4" whether or not a device exists.
 *
 * The last part asks a framework that has a path without Metal whether that path works: CoreImage
 * runs a filter kernel over four pixels and the result is read back. It is the one thing here that
 * can answer "can this system process an image at all", which the camera and Photos both need.
 *
 *   usage: metal_test
 */
#import <CoreFoundation/CoreFoundation.h>
#import <CoreImage/CoreImage.h>
#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#import <Metal/Metal.h>
#import <objc/message.h>
#import <objc/runtime.h>
#import <signal.h>
#import <stdio.h>
#import <stdlib.h>
#import <sys/wait.h>
#import <time.h>
#import <unistd.h>

extern CFTypeRef MGCopyAnswer(CFStringRef key);

#define EXTENSIONS      "/System/Library/Extensions"

/*
 * Two of the things asked below can take the process down with them, and did: asking CoreImage for
 * its default context killed metal_test outright on this system (nothing printed, no crash report),
 * and taking a plug-in bundle's principal class runs Apple's own driver code. Each is therefore run
 * in a child of its own: whatever happens to the child, the parent says how it ended and goes on.
 * A child that wedges instead of dying is given `secs` and then killed, which is itself an answer.
 */
static void
in_a_child(const char *what, int secs, void (*body)(void))
{
	pid_t pid = fork();
	int status = 0, waited = 0;

	if (pid < 0) {
		printf("metal_test:   cannot fork to ask about %s\n", what);
		return;
	}
	if (pid == 0) {
		body();
		_exit(0);
	}
	/* waitpid without hanging for ever: a wedged child is news too */
	while (waitpid(pid, &status, WNOHANG) == 0) {
		struct timespec ts = { .tv_sec = 0, .tv_nsec = 100 * 1000 * 1000 };

		nanosleep(&ts, NULL);
		if (++waited > secs * 10) {
			kill(pid, SIGKILL);
			waitpid(pid, &status, 0);
			printf("metal_test:   %s: still going after %d s, killed (it wedges here)\n", what, secs);
			return;
		}
	}
	if (WIFSIGNALED(status)) {
		printf("metal_test:   %s: the process died of signal %d right there\n", what, WTERMSIG(status));
	} else if (WEXITSTATUS(status) != 0) {
		printf("metal_test:   %s: ended with status %d\n", what, WEXITSTATUS(status));
	}
}

/* A string property of an IOKit service, or NULL; the caller frees it. */
static char *
service_string(io_service_t service, const char *name)
{
	CFStringRef key = CFStringCreateWithCString(NULL, name, kCFStringEncodingUTF8);
	CFTypeRef value = IORegistryEntryCreateCFProperty(service, key, NULL, 0);
	char *text = NULL;

	CFRelease(key);
	if (value != NULL) {
		if (CFGetTypeID(value) == CFStringGetTypeID()) {
			text = calloc(1, 256);
			if (text != NULL && !CFStringGetCString(value, text, 256, kCFStringEncodingUTF8)) {
				free(text);
				text = NULL;
			}
		}
		CFRelease(value);
	}
	return text;
}

/* Step 1 and 2: the services Metal looks for, and the two properties it reads of each. */
static int
accelerators(void)
{
	io_iterator_t it = IO_OBJECT_NULL;
	CFMutableDictionaryRef matching = IOServiceMatching("IOAcceleratorES");
	kern_return_t kr;
	io_service_t service;
	int found = 0;

	if (matching == NULL) {
		printf("metal_test: IOServiceMatching(\"IOAcceleratorES\") failed\n");
		return 0;
	}
	/* IOServiceGetMatchingServices consumes the dictionary. */
	kr = IOServiceGetMatchingServices(MACH_PORT_NULL, matching, &it);
	if (kr != KERN_SUCCESS) {
		printf("metal_test: IOServiceGetMatchingServices failed, %#x\n", kr);
		return 0;
	}
	while ((service = IOIteratorNext(it)) != IO_OBJECT_NULL) {
		io_name_t name = { 0 };
		char *plugin = service_string(service, "MetalPluginName");
		char *class_name = service_string(service, "MetalPluginClassName");

		IORegistryEntryGetName(service, name);
		printf("metal_test:   IOAcceleratorES \"%s\": MetalPluginName %s, MetalPluginClassName %s\n",
		    name, plugin ? plugin : "(none)", class_name ? class_name : "(none)");
		free(plugin);
		free(class_name);
		IOObjectRelease(service);
		found++;
	}
	IOObjectRelease(it);
	if (found == 0) {
		printf("metal_test:   none: Metal will find no device at all\n");
	}
	return found;
}

/* What the registry does have that looks like a GPU, whatever its class is called. */
static void
gpu_services(void)
{
	static const char *const classes[] = { "RedwoodGPU", "IOGPU", "AGXAccelerator", "IOAccelerator" };
	int found = 0;

	for (size_t i = 0; i < sizeof(classes) / sizeof(classes[0]); i++) {
		CFMutableDictionaryRef matching = IOServiceMatching(classes[i]);
		io_iterator_t it = IO_OBJECT_NULL;
		io_service_t service;

		if (matching == NULL || IOServiceGetMatchingServices(MACH_PORT_NULL, matching, &it) != KERN_SUCCESS) {
			continue;
		}
		while ((service = IOIteratorNext(it)) != IO_OBJECT_NULL) {
			io_name_t name = { 0 };
			char *client = service_string(service, "IOUserClientClass");

			IORegistryEntryGetName(service, name);
			printf("metal_test:   %s \"%s\": IOUserClientClass %s\n", classes[i], name,
			    client ? client : "(none)");
			free(client);
			IOObjectRelease(service);
			found++;
		}
		IOObjectRelease(it);
	}
	if (found == 0) {
		printf("metal_test:   none\n");
	}
}

/* Steps 3 and 4: could a plug-in bundle in this image be loaded and taken for a device at all. */
static void
plugins(void)
{
	NSFileManager *files = [NSFileManager defaultManager];
	NSArray<NSString *> *entries = [files contentsOfDirectoryAtPath:@EXTENSIONS error:NULL];
	Class device_class = objc_getClass("_MTLDevice");
	int bundles = 0;

	printf("metal_test:   Metal's own base class _MTLDevice: %s\n", device_class ? "there" : "not there");
	if (entries == nil) {
		printf("metal_test:   %s is not in this image: no plug-in could be loaded\n", EXTENSIONS);
		return;
	}
	for (NSString *entry in entries) {
		if (![entry hasSuffix:@".bundle"]) {
			continue;
		}
		NSString *path = [@EXTENSIONS stringByAppendingPathComponent:entry];
		NSBundle *bundle = [NSBundle bundleWithPath:path];
		NSString *executable = [bundle objectForInfoDictionaryKey:@"CFBundleExecutable"];
		NSString *principal = [bundle objectForInfoDictionaryKey:@"NSPrincipalClass"];
		Class loaded = Nil;
		const char *verdict;

		bundles++;
		/* Taking the class is what loads the bundle's code; it is the step that fails first. */
		loaded = principal != nil ? [bundle classNamed:principal] : [bundle principalClass];
		if (loaded == Nil) {
			verdict = "the class does not load";
		} else if (device_class == Nil) {
			verdict = "loaded, but there is no _MTLDevice to compare it with";
		} else if ([loaded isSubclassOfClass:device_class]) {
			verdict = "loaded, and a _MTLDevice: Metal would take it";
		} else {
			verdict = "loaded, but not a _MTLDevice: Metal would drop it";
		}
		printf("metal_test:   %s: executable %s, principal class %s -- %s\n", [entry UTF8String],
		    executable ? [executable UTF8String] : "(none)",
		    principal ? [principal UTF8String] : "(none)", verdict);
	}
	if (bundles == 0) {
		printf("metal_test:   no .bundle in %s\n", EXTENSIONS);
	}
}

/* Step 5, as a program sees it. */
static void
the_device(void)
{
	id<MTLDevice> device = MTLCreateSystemDefaultDevice();

	if (device == nil) {
		printf("metal_test:   MTLCreateSystemDefaultDevice() = nil (and says nothing about why)\n");
		return;
	}
	printf("metal_test:   MTLCreateSystemDefaultDevice() = \"%s\", class %s\n",
	    [[device name] UTF8String], class_getName([(id)device class]));
	printf("metal_test:   registry id %llu, unified memory %d, max threads per group %lu, "
	    "Apple family 1/4 %d/%d\n", (unsigned long long)[device registryID], (int)[device hasUnifiedMemory],
	    (unsigned long)[device maxThreadsPerThreadgroup].width,
	    (int)[device supportsFamily:MTLGPUFamilyApple1], (int)[device supportsFamily:MTLGPUFamilyApple4]);
}

/* What iOS publishes about graphics besides: artwork, not rendering (see the note above). */
static void
gestalt(void)
{
	static const char *const keys[] = {
		"GraphicsFeatureSetClass", "GraphicsFeatureSetFallbacks", "ArtworkDeviceSubType",
		"HasExtendedColorDisplay", "main-screen-class", "DeviceSupportsMetal",
	};

	for (size_t i = 0; i < sizeof(keys) / sizeof(keys[0]); i++) {
		CFStringRef key = CFStringCreateWithCString(NULL, keys[i], kCFStringEncodingUTF8);
		CFTypeRef answer = MGCopyAnswer(key);
		CFStringRef text = answer ? CFCopyDescription(answer) : NULL;
		char line[256] = "(none)";

		if (text != NULL) {
			CFStringGetCString(text, line, sizeof(line), kCFStringEncodingUTF8);
			CFRelease(text);
		}
		printf("metal_test:   %s = %s\n", keys[i], line);
		if (answer != NULL) {
			CFRelease(answer);
		}
		CFRelease(key);
	}
}

/*
 * A framework that has a path without Metal: CoreImage. A filter kernel is run over four pixels
 * and the result read back, which tells whether this system can process an image at all.
 */
/* `options` is nil for the default context, or a dictionary to ask for something else. */
static NSDictionary *coreimage_options;

static void
coreimage(void)
{
	@autoreleasepool {
		CIContext *context;
		CIImage *source, *result;
		CIFilter *filter;
		CGImageRef image;
		CGRect rect = CGRectMake(0, 0, 4, 4);
		unsigned char pixels[4 * 4 * 4];
		CGColorSpaceRef space;
		CGContextRef bitmap;

		/* said before the call, not after: this is the call that was dying */
		printf("metal_test:   asking for a CIContext (%s)...\n",
		    coreimage_options == nil ? "the default one" : "software renderer");
		context = [CIContext contextWithOptions:coreimage_options];
		if (context == nil) {
			printf("metal_test:   no CIContext: CoreImage cannot work here\n");
			return;
		}
		/*
		 * Which of CoreImage's two backends the default context took. iOS 16.5 has both:
		 * CI::GLContext (with CI::GLMainProgram, CI::GLTextureManager) and CI::MetalContext
		 * (with CI::MetalDAG, CI::MetalKernel); -[CIContext initWithOptions:] chooses between
		 * them by CIGLIsUsable(). -isOpenGLBased and -device are private, so they are asked for
		 * only if they are there.
		 */
		printf("metal_test:   CIContext: %s\n", class_getName([context class]));
		/*
		 * Sent through a typed objc_msgSend, not -performSelector:, because -isOpenGLBased
		 * returns a BOOL and not an object: ARC would take the return value for one.
		 */
		SEL gl_sel = @selector(isOpenGLBased);
		if ([context respondsToSelector:gl_sel]) {
			BOOL (*ask)(id, SEL) = (BOOL (*)(id, SEL))objc_msgSend;
			printf("metal_test:   backend: %s\n", ask(context, gl_sel)
			    ? "OpenGL ES (CI::GLContext)" : "Metal (CI::MetalContext)");
		} else {
			printf("metal_test:   backend: -isOpenGLBased is not there to ask\n");
		}
		SEL dev_sel = @selector(device);
		if ([context respondsToSelector:dev_sel]) {
			id (*ask_device)(id, SEL) = (id (*)(id, SEL))objc_msgSend;
			id device = ask_device(context, dev_sel);
			printf("metal_test:   the context's Metal device: %s\n",
			    device ? class_getName([device class]) : "nil");
		}
		/* orange, inverted by a kernel: 1, 0.5, 0 -> 0, 0.5, 1 */
		source = [[CIImage imageWithColor:[CIColor colorWithRed:1.0 green:0.5 blue:0.0]]
		    imageByCroppingToRect:rect];
		filter = [CIFilter filterWithName:@"CIColorInvert"];
		if (filter == nil) {
			printf("metal_test:   no CIColorInvert filter\n");
			return;
		}
		[filter setValue:source forKey:kCIInputImageKey];
		result = [filter valueForKey:kCIOutputImageKey];
		if (result == nil) {
			printf("metal_test:   the filter gave no image\n");
			return;
		}
		printf("metal_test:   running the filter kernel...\n");
		image = [context createCGImage:result fromRect:rect];
		if (image == NULL) {
			printf("metal_test:   CoreImage did not render: a filter kernel cannot run here\n");
			return;
		}
		memset(pixels, 0, sizeof(pixels));
		space = CGColorSpaceCreateDeviceRGB();
		bitmap = CGBitmapContextCreate(pixels, 4, 4, 8, 4 * 4, space,
		    kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
		if (bitmap != NULL) {
			CGContextDrawImage(bitmap, rect, image);
			printf("metal_test:   CoreImage rendered: first pixel %u %u %u %u (inverted orange is about 0 128 255 255)\n",
			    pixels[0], pixels[1], pixels[2], pixels[3]);
			CGContextRelease(bitmap);
		} else {
			printf("metal_test:   CoreImage rendered, but the pixels could not be read back\n");
		}
		CGColorSpaceRelease(space);
		CGImageRelease(image);
	}
}

int
main(void)
{
	/* every line as it happens: a call that kills the process must not take the log with it */
	setvbuf(stdout, NULL, _IONBF, 0);
	printf("metal_test: the services Metal looks for (IOAcceleratorES)\n");
	accelerators();
	printf("metal_test: what the registry has that is a GPU\n");
	gpu_services();
	printf("metal_test: the device a program gets\n");
	the_device();
	printf("metal_test: what iOS publishes about graphics (artwork, not rendering)\n");
	gestalt();
	printf("metal_test: CoreImage, which has a path without Metal\n");
	in_a_child("the default CIContext", 30, coreimage);
	coreimage_options = @{ @"kCIContextUseSoftwareRenderer": @YES };
	in_a_child("a software-renderer CIContext", 30, coreimage);
	/*
	 * Last on purpose: taking a bundle's principal class loads that bundle's code and runs its
	 * initialisers -- for AGXMetalA11.bundle, Apple's own driver for the A11's GPU, which this
	 * image does carry.
	 */
	printf("metal_test: the plug-in bundles of %s (loads their code: kept last)\n", EXTENSIONS);
	in_a_child("loading the plug-in bundles", 60, plugins);
	return 0;
}
