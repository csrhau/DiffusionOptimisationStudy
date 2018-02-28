PRODUCT = deqn3d.x
SRC_DIR = src
BUILD_DIR = build

CXXFLAGS=-std=c++11 -g -Wall -Wextra -Werror -pedantic -O3

CPP_HDR_FILES = $(wildcard $(SRC_DIR)/*.h)
CPP_SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES = $(CPP_SRC_FILES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
DEP_FILES = $(OBJ_FILES:.o=.d)


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


###########################
# Compiler Specific Flags #
###########################

ifeq ($(CXX), g++)
$(warning WARNING: GNU compilers do not support multithreading on host CPUs)
CXXFLAGS += -fopenacc
LDFLAGS += -fopenacc
endif

.DEFAULT: all
.PHONY: all
all: $(BUILD_DIR)/$(PRODUCT)

$(BUILD_DIR)/$(PRODUCT): $(OBJ_FILES) | $(BUILD_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCLUDES) -MMD -c $< -o $@

$(BUILD_DIR):
	@mkdir -p $@

.PHONY: clean
clean:
	rm -rf $(PRODUCT)
	rm -rf $(OBJ_FILES)
	rm -rf $(DEP_FILES)
	rm -rf $(BUILD_DIR)

-include $(DEP_FILES) # has to be below all: otherwise it subsumes the default target
