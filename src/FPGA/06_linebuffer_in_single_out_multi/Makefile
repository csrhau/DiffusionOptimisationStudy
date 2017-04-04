HOST_BINARY = deqn3d.x
FPGA_BINARY = deqn3d.xclbin

SRC_DIR = src
KERNEL_DIR = $(SRC_DIR)/kernels
TEMPLATE_DIR = $(SRC_DIR)/templates
BUILD_DIR = build

XDEVICE  = xilinx:adm-pcie-ku3:1ddr:3.0
EMULATOR = xilinx:pcie-hw-em:7v3:1.0

CC = $(XILINX_SDACCEL)/lnx64/tools/gcc/bin/gcc
CCFLAGS = -g -Wall -Wextra -pedantic -O0 -std=gnu99 
CPPFLAGS += -DKXVER=3 -DFPGA_DEVICE -I$(XILINX_SDACCEL)/runtime/include/1_2
CPPFLAGS += -DTARGET_DEVICE=$(TARGET_DEVICE) -DTARGET_VENDOR=Xilinx
CPPFLAGS += -DTARGET_BINARY=$(FPGA_BINARY)
LDFLAGS = -L$(XILINX_SDACCEL)/runtime/lib/x86_64
LDLIBS = -lxilinxopencl

XCLCC = xocc
XCLFLAGS = --xdevice ${XDEVICE} $(KERNEL_DEFS) $(KERNEL_INCS) -t $(XCLTARGET) -I ./src

C_SRC_FILES = $(wildcard $(SRC_DIR)/*.c) 
KERNEL_SRC_FILES = $(wildcard $(KERNEL_DIR)/*.cl)

OBJ_FILES = $(C_SRC_FILES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
DEP_FILES = $(OBJ_FILES:.o=.d)

.DEFAULT: sw_emu 
.PHONY: sw_emu 
sw_emu: host_binary $(BUILD_DIR)/sw_emu_$(FPGA_BINARY) $(BUILD_DIR)/sw_emu_runscript.sh

.PHONY: hw_emu 
hw_emu: host_binary $(BUILD_DIR)/hw_emu_$(FPGA_BINARY) $(BUILD_DIR)/hw_emu_runscript.sh

.PHONY: hw
hw: host_binary $(BUILD_DIR)/hw_$(FPGA_BINARY) $(BUILD_DIR)/hw_runscript.sh

.PHONY: host_binary
host_binary: $(BUILD_DIR)/$(HOST_BINARY)

.PHONY: all
all: sw_emu hw_emu hw

.PHONY: emulation
emulation: sw_emu hw_emu

$(BUILD_DIR)/$(HOST_BINARY): $(OBJ_FILES) | $(BUILD_DIR)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@ 

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CCFLAGS) $(INCLUDES) -MMD -c $< -o $@

$(BUILD_DIR)/%_$(FPGA_BINARY): XCLTARGET=$*
$(BUILD_DIR)/%_$(FPGA_BINARY): $(KERNEL_SRC_FILES) src/common.h | $(BUILD_DIR)
	$(XCLCC) $(XCLFLAGS) -o $@ $(KERNEL_SRC_FILES) 

$(BUILD_DIR)/sw_emu_runscript.sh: $(TEMPLATE_DIR)/runscript.m4 | $(BUILD_DIR)
	m4 -DBINARY=$(HOST_BINARY) -DXCLBIN=sw_emu_$(FPGA_BINARY) -DDEVICE=$(EMULATOR) $< > $@
	chmod u+x $@

$(BUILD_DIR)/hw_emu_runscript.sh: $(TEMPLATE_DIR)/runscript.m4 | $(BUILD_DIR)
	m4 -DBINARY=$(HOST_BINARY) -DXCLBIN=hw_emu_$(FPGA_BINARY) -DDEVICE=$(EMULATOR) $< > $@
	chmod u+x $@

$(BUILD_DIR)/hw_runscript.sh: $(TEMPLATE_DIR)/runscript.m4 | $(BUILD_DIR)
	m4 -DBINARY=$(HOST_BINARY) -DXCLBIN=hw_$(FPGA_BINARY) -DDEVICE=$(XDEVICE) $< > $@
	chmod u+x $@

$(BUILD_DIR):
	@mkdir -p $@

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
	rm -f sdaccel_profile_summary.csv
	rm -f sdaccel_profile_summary.html
	rm -rf _xocc_*.dir

-include $(DEP_FILES) # has to be below all: otherwise it subsumes the default target
