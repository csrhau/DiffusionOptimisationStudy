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
	COMPILER_CXXFLAGS = -acc -Minfo=accel -Bdynamic -ta=$(ta)
	COMPILER_LDFLAGS  = -acc -Bdynamic
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
	@echo "PGI specific variables:"
	@echo "    ta=[host,multicore,tesla]"
	@echo ""
	@echo "PGI Usage Example: make timesteps=20 ta=tesla"

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
