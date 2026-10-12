/*
 * What Mesa's freedreno takes from libdrm's xf86drm.h, for a system without Linux's DRM: the
 * declarations here, the functions in redwood_drm.c, which carry the requests to the kernel's
 * GPU driver (the service RedwoodGPU; iokit/Kernel/RedwoodGPU.cpp) as calls of its user client.
 * tools/mk-mesa.sh puts both into Mesa's tree, where they are the "libdrm" freedreno is built
 * against. The request numbers and their structures are Linux's (Mesa's include/drm-uapi).
 */
#ifndef REDWOOD_XF86DRM_H
#define REDWOOD_XF86DRM_H

#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#include "drm-uapi/drm.h"

#define DRM_NODE_PRIMARY        0
#define DRM_NODE_RENDER         2
#define DRM_CLOEXEC             O_CLOEXEC
#define DRM_RDWR                O_RDWR

typedef struct _drmVersion {
	int version_major, version_minor, version_patchlevel;
	int name_len;
	char *name;
	int date_len;
	char *date;
	int desc_len;
	char *desc;
} drmVersion, *drmVersionPtr;

int drmOpenWithType(const char *name, const char *busid, int type);
int drmIoctl(int fd, unsigned long request, void *arg);
int drmCommandWrite(int fd, unsigned long index, const void *data, unsigned long size);
int drmCommandWriteRead(int fd, unsigned long index, void *data, unsigned long size);
drmVersionPtr drmGetVersion(int fd);
void drmFreeVersion(drmVersionPtr version);
int drmGetCap(int fd, uint64_t capability, uint64_t *value);

/* buffers and fences as file descriptors: there are none here */
int drmPrimeHandleToFD(int fd, uint32_t handle, uint32_t flags, int *prime_fd);
int drmPrimeFDToHandle(int fd, int prime_fd, uint32_t *handle);
int drmSyncobjCreate(int fd, uint32_t flags, uint32_t *handle);
int drmSyncobjDestroy(int fd, uint32_t handle);
int drmSyncobjReset(int fd, const uint32_t *handles, uint32_t count);
int drmSyncobjFDToHandle(int fd, int obj_fd, uint32_t *handle);
int drmSyncobjHandleToFD(int fd, uint32_t handle, int *obj_fd);
int drmSyncobjImportSyncFile(int fd, uint32_t handle, int sync_file_fd);
int drmSyncobjExportSyncFile(int fd, uint32_t handle, int *sync_file_fd);

/* Not libdrm's: a buffer's memory in this process, in place of mmap() on the device (freedreno_bo.c is patched to call these). */
void *redwood_drm_map(int fd, uint32_t handle, uint64_t size);
void redwood_drm_unmap(int fd, uint32_t handle, void *address);

#endif
