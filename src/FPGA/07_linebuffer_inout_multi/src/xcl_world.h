#pragma once

#include <CL/cl.h>

#define QUOTE(name) #name
#define STR(macro) QUOTE(macro)
#define VENDOR_STRING STR(TARGET_VENDOR)

struct XCLWorld {
  char *vendor_name;
  char *device_name;
  char *binary_name;
  unsigned char *binary_data;
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_int status;
};

void XCLSetup(char *vendor, char *device, char *binary, struct XCLWorld* world);
void XCLTeardown(struct XCLWorld* world);

// Should be static
void PopulatePlatform(char *vendor, struct XCLWorld* world);
void PopulateDevice(char *device, struct XCLWorld* world);
void PopulateContext(struct XCLWorld* world);
void PopulateQueue(struct XCLWorld* world);
void PopulateProgram(char *binary, struct XCLWorld* world);
size_t LoadFile(char *binary, unsigned char *data);
