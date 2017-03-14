#include "xcl_tools.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <CL/cl.h>

void XCLSetup(char *vendor, char *device, char *binary, struct XCLWorld* world) {
  PopulatePlatform(vendor, world);
  if (world->status != CL_SUCCESS) {
    fprintf(stderr, "Failed to populate vendor platform: %s\n", vendor);
    exit(EXIT_FAILURE);
  }
  PopulateDevice(device, world);
  if (world->status != CL_SUCCESS) {
    fprintf(stderr, "Failed to populate device: %s\n", device);
    exit(EXIT_FAILURE);
  }
  PopulateContext(world);
  if (world->status != CL_SUCCESS) {
    fprintf(stderr, "Failed to populate context\n");
    exit(EXIT_FAILURE);
  }
  PopulateQueue(world);
  if (world->status != CL_SUCCESS) {
    fprintf(stderr, "Failed to populate queue\n");
    exit(EXIT_FAILURE);
  }
  PopulateProgram(binary, world);
  if (world->status != CL_SUCCESS) {
    fprintf(stderr, "Failed to populate binary %s\n", binary);
    exit(EXIT_FAILURE);
  }
}

void XCLTeardown(struct XCLWorld* world) {
  if (world->vendor_name) {
    free(world->vendor_name);
  }
  if (world->device_name) {
    free(world->device_name);
  }
  if (world->binary_name) {
    free(world->binary_name);
  }
  if (world->binary_data) {
    free(world->binary_data);
  }
  if (world->program) {
    clReleaseProgram(world->program);
  }
  if (world->queue) {
    clReleaseCommandQueue(world->queue);
  }
  if (world->context) {
    clReleaseContext(world->context);
  }
  if (world->device) {
    clReleaseDevice(world->device);
  }
}

void PopulatePlatform(char *vendor, struct XCLWorld* world) {
  cl_uint num_platforms;
  world->status = clGetPlatformIDs(0, NULL, &num_platforms);
  if (world->status != CL_SUCCESS) { return; } 
  cl_platform_id* platforms = (cl_platform_id *) malloc(num_platforms * sizeof(cl_platform_id));
  world->status = clGetPlatformIDs(num_platforms, platforms, NULL);
  if (world->status != CL_SUCCESS) { return; } 
  for (cl_uint pid = 0; pid < num_platforms; ++pid) {
    size_t vendor_sz;
    const cl_platform_id plt = platforms[pid];
    world->status = clGetPlatformInfo(plt, CL_PLATFORM_VENDOR, 0, NULL, &vendor_sz);
    if (world->status != CL_SUCCESS) { return; } 
    char * vendor_name = (char *) malloc(vendor_sz * sizeof(char));
    world->status = clGetPlatformInfo(plt, CL_PLATFORM_VENDOR, vendor_sz, vendor_name, NULL);
    if (world->status != CL_SUCCESS) { return; } 
    if (strstr(vendor_name, vendor)) {
      world->platform = plt;
      world->vendor_name = vendor_name;
      break;
    } else {
      free(vendor_name);
    }
  }
  free(platforms);
}

void PopulateDevice(char *device, struct XCLWorld* world) {
  cl_uint num_devices;
  world->status = clGetDeviceIDs(world->platform, CL_DEVICE_TYPE_ACCELERATOR, 0, NULL, &num_devices);
  if (world->status != CL_SUCCESS) { return; }
  cl_device_id* devices = (cl_device_id *) malloc(num_devices * sizeof(cl_device_id));
  world->status = clGetDeviceIDs(world->platform, CL_DEVICE_TYPE_ACCELERATOR, num_devices, devices, NULL);
  if (world->status != CL_SUCCESS) { return; }
  for (cl_uint did = 0; did < num_devices; ++did) {
    size_t name_sz;
    const cl_device_id dev = devices[did];
    world->status = clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, NULL, &name_sz);
    if (world->status != CL_SUCCESS) { return; }
    char * device_name = (char *) malloc(name_sz * sizeof(char));
    world->status = clGetDeviceInfo(dev, CL_DEVICE_NAME, name_sz, device_name, NULL);
    if (world->status != CL_SUCCESS) { return; } 
    if (strstr(device_name, device)) {
      world->device = dev;
      world->device_name = device_name;
      break;
    } else {
      free(device_name);
    }
  }
  free(devices);
}

void PopulateContext(struct XCLWorld* world) {
  world->context = clCreateContext(0, 1, &world->device, NULL, NULL, &world->status);
}

void PopulateQueue(struct XCLWorld* world) {
  world->queue = clCreateCommandQueue(world->context, world->device, CL_QUEUE_PROFILING_ENABLE, &world->status);
}

void PopulateProgram(char *binary, struct XCLWorld* world) {
  world->binary_name = strdup(binary);
  size_t fsize = LoadFile(binary, NULL);
  if (fsize == 0) {
    fprintf(stderr, "Failed to open file %s\n", binary);
    exit(EXIT_FAILURE);
  }
  world->binary_data = (char *) malloc(fsize);
  if (LoadFile(binary, world->binary_data) != fsize) {
    fprintf(stderr, "Failed to read file %s\n", binary);
    free(world->binary_data);
    exit(EXIT_FAILURE);
  }


}

size_t LoadFile(char *binary, char *dest) {
  FILE *f = fopen(binary, "rb");
  if (!f) { return 0; }
  fseek(f, 0, SEEK_END);
  size_t size = ftell(f);
  if (dest) {
    fseek(f, 0, SEEK_SET);
    if (size != fread(dest, sizeof(char), size, f)) {
      fclose(f);
      return 0;
    }
    fclose(f);
    dest[size] = '\0';
  }
  return size+1;
}
