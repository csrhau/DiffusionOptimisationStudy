#include "xcl_world.h"

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
    fprintf(stderr, "Failed to populate binary %s: %d\n", binary, world->status);
    exit(EXIT_FAILURE);
  }
}

void XCLTeardown(struct XCLWorld* world) {
  if (world->vendor_name) {
    free(world->vendor_name);
    world->vendor_name = NULL;
  }
  if (world->device_name) {
    free(world->device_name);
    world->device_name = NULL;
  }
  if (world->binary_name) {
    free(world->binary_name);
    world->binary_name = NULL;
  }
  if (world->binary_data) {
    free(world->binary_data);
    world->binary_data = NULL;
  }
  if (world->program) {
    clReleaseProgram(world->program);
    world->program = 0;
  }
  if (world->queue) {
    clReleaseCommandQueue(world->queue);
    world->queue = 0;
  }
  if (world->context) {
    clReleaseContext(world->context);
    world->context = 0;
  }
  if (world->device) {
    clReleaseDevice(world->device);
    world->device = 0;
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
  world->binary_data = (unsigned char *) malloc(fsize + 1);
  if (LoadFile(binary, world->binary_data) != fsize) {
    fprintf(stderr, "Failed to read file %s\n", binary);
    free(world->binary_data);
    exit(EXIT_FAILURE);
  }
  const unsigned char *program_start = (const unsigned char *) world->binary_data;
  world->program = clCreateProgramWithBinary(world->context, 1, &world->device, &fsize, &program_start, NULL, &world->status);
  if (world->status != CL_SUCCESS) { return; } 
  if (!world->program) {
    world->status = CL_BUILD_PROGRAM_FAILURE;
  }
}

size_t LoadFile(char *binary, unsigned char *data) {
  FILE *f = fopen(binary, "rb");
  if (!f) { return 0; }
  fseek(f, 0, SEEK_END);
  size_t size = ftell(f);
  if (data) {
    fseek(f, 0, SEEK_SET);
    if (size != fread(data, sizeof(unsigned char), size, f)) {
      fclose(f);
      return 0;
    }
    fclose(f);
    data[size] = '\0';
  }
  return size;
}
