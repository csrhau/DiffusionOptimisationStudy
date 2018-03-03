PRODUCT = deqn3d.x
SRC_DIR = src
BUILD_DIR = build
DEP_DIR = $(BUILD_DIR)

CPP_HDR_FILES = $(wildcard $(SRC_DIR)/*.h)
CPP_SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES = $(CPP_SRC_FILES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
DEP_FILES = $(CPP_SRC_FILES:$(SRC_DIR)/%.cpp=$(DEP_DIR)/%.d)

CXXFLAGS = $(COMPILER_CXXFLAGS) -std=c++11 -Werror -pedantic -O0 -g
LDFLAGS = $(COMPILER_LDFLAGS)
LDLIBS = $(COMPILER_LDLIBS)


###########################
# Compiler Specific Flags #
###########################

ifeq ($(findstring g++, $(CXX)), g++)
	TOOLCHAIN=GNU
endif

ifeq ($(findstring pgc++, $(CXX)), pgc++)
	TOOLCHAIN=PGI
endif

ifeq ($(TOOLCHAIN), GNU)
	COMPILER_CXXFLAGS = -fopenacc -Wall -Wextra
	COMPILER_LDFLAGS  = -fopenacc
	DEPFLAGS = -MMD -MF$(BUILD_DIR)/$*.d
endif

ifeq ($(TOOLCHAIN), PGI)
	ta ?= host # Default to single-threaded host impl (works everywhere)
	COMPILER_CXXFLAGS = -acc -Minfo=accel -ta=$(ta)
	COMPILER_LDFLAGS  = -acc -Bstatic_pgi $(EXPLICIT_LIBRARY_PATH)
	# PGI Doesn't by default respect LIBRARY_PATH or LD_LIBRARY_PATH
	COMPILER_LDFLAGS += $(addprefix -L, $(subst :, , $(LIBRARY_PATH)))
	DEPFLAGS = -MMD$(BUILD_DIR)/$*.d
endif


####################################
# Simulation Size Parameterization #
####################################

ifneq ($(timesteps),)
CPPFLAGS +=-DTIMESTEPS=$(timesteps)
endif

ifneq ($(imax),)
CPPFLAGS +=-DIMAX=$(imax)
endif

ifneq ($(jmax),)
CPPFLAGS +=-DJMAX=$(jmax)
endif

ifneq ($(kmax),)
CPPFLAGS +=-DKMAX=$(kmax)
endif


##############
# Make Rules #
##############

.DEFAULT: all
.PHONY: all
all: $(BUILD_DIR)/$(PRODUCT)

.PHONY: ibm-p8
ibm-p8: ta=tesla:cc35,lineinfo,nordc,debug imax=100 jmax=100 kmax=100 timesteps=100
ibm-p8: $(BUILD_DIR)/$(PRODUCT)

$(BUILD_DIR)/$(PRODUCT): $(OBJ_FILES) | $(BUILD_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR) ; $(COMPILER_WARNINGS)
	$(CXX) $(CPPFLAGS) $(DEPFLAGS) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR):
	@mkdir -p $@

.PHONY: usage
usage:
	@echo "Simulation Parameters"
	@echo "====================="
	@echo "    imax=<int>"
	@echo "    jmax=<int>"
	@echo "    kmax=<int>"
	@echo "    timesteps=<int>"
	@echo ""
	@echo "Compiler Options"
	@echo "================"
	@echo "PGI:"
	@echo "    ta=host|multicore|tesla[:tesla-options]"
	@echo ""
	@echo "Usage examples"
	@echo "=============="
	@echo "PGI + CUDA:    make ta=tesla:cc60,lineinfo,nordc,debug {imax,jmax,kmax,timesteps}=100"

.PHONY: clean
clean:
	rm -rf $(PRODUCT)
	rm -rf $(OBJ_FILES)
	rm -rf $(DEP_FILES)
	rm -rf $(BUILD_DIR)


###################################
# Automatic Dependency Generation #
###################################

-include $(DEP_FILES) # has to be after 'all', otherwise becomes default target
