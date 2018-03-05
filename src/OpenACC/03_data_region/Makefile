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


##########################
# OpenACC Specific Flags #
##########################

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
ifeq ($(cuda),)
	ACC_ARCHITECTURE = -ta=$(ta)
else
	ACC_ARCHITECTURE = -ta=tesla:cuda$(cuda),nordc,debug
	ACC_LINKFLAGS = -Mcuda=$(cuda)
endif
	COMPILER_CXXFLAGS = -acc -Minfo=accel $(ACC_ARCHITECTURE)
	COMPILER_LDFLAGS  = -acc $(ACC_LINKFLAGS)
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
	@echo "    cuda=7.5|8.0|9.0|9.1"
	@echo "    ta=host|multicore"
	@echo ""
	@echo "Usage examples"
	@echo "=============="
	@echo "GNU:               make {imax,jmax,kmax,timesteps}=100"
	@echo "PGI + CUDA 9.0:    make cuda=9.0 {imax,jmax,kmax,timesteps}=100"
	@echo "PGI + CUDA 9.1:    make cuda=9.1 {imax,jmax,kmax,timesteps}=100"

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
