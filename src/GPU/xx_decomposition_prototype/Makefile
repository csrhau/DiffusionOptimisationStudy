PRODUCT = deqn3d.x
SRC_DIR = src
TEST_DIR = test
BUILD_DIR = build
GTEST_BASE_DIR ?= googletest

NVCCFLAGS=-lineinfo -std=c++11 -O3
CXXFLAGS=-g -Wall -Wextra -pedantic -std=c++11

LDLIBS = -lcuda -lcudart
LDFLAGS = # paths to libraries; -L/path/to/library

CXX_HDR_FILES = $(wildcard $(SRC_DIR)/*.h)
CXX_SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp) 
CXX_OBJ_FILES = $(CXX_SRC_FILES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
CXX_DEP_FILES = $(CXX_OBJ_FILES:.o=.d)

CU_HDR_FILES = $(wildcard $(SRC_DIR)/*.cuh)
CU_SRC_FILES = $(wildcard $(SRC_DIR)/*.cu) 
CU_OBJ_FILES = $(CU_SRC_FILES:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.cu.o)
CU_DEP_FILES = $(CU_OBJ_FILES:.o=.d)

.DEFAULT: all
.PHONY: all
all: $(BUILD_DIR)/$(PRODUCT)

$(BUILD_DIR)/$(PRODUCT): $(CU_OBJ_FILES) $(CXX_OBJ_FILES)
	$(NVCC) $(LDFLAGS) $^ $(LDLIBS) -o $@ 

$(BUILD_DIR)/%.cu.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCCINCLUDES) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCLUDES) -MMD -c $< -o $@

$(BUILD_DIR):
	@mkdir -p $@

.PHONY: tests
tests: $(BUILD_DIR)/tests.x

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)

.PHONY: clobber
clobber: clean
	rm -rf $(GTEST_BASE_DIR)

######################################
# GoogleTest framework and testcases #
######################################
GTEST_DIR = $(GTEST_BASE_DIR)/googletest
GMOCK_DIR = $(GTEST_BASE_DIR)/googlemock
GTEST_HEADERS = $(wildcard $(GTEST_DIR)/include/gtest/*.h)\
								$(wildcard $(GTEST_DIR)/include/gtest/internal/*.h)
GMOCK_HEADERS = $(GMOCK_DIR)/include/gmock/*.h \
								$(GMOCK_DIR)/include/gmock/internal/*.h \
								$(GTEST_HEADERS)
GTEST_SRCS = $(wildcard $(GTEST_DIR)/src/*.cc)\
             $(wildcard $(GTEST_DIR)/src/*.h) \
					   $(GTEST_HEADERS)
GMOCK_SRCS = $(wildcard $(GMOCK_DIR)/src/*.cc) \
						 $(GMOCK_HEADERS)
TEST_SRC_FILES = $(wildcard $(TEST_DIR)/*.cpp)
TEST_OBJ_FILES = $(TEST_SRC_FILES:$(TEST_DIR)/%.cpp=$(BUILD_DIR)/%.o)
TEST_DEP_FILES = $(TEST_OBJ_FILES:.o=.d)

$(GTEST_BASE_DIR):
	git clone https://github.com/google/googletest.git

ifeq ("$(wildcard $(GTEST_BASE_DIR))","")
$(BUILD_DIR)/tests.x:
	@echo "ERROR! Please run 'make googletest' or set the GTEST_BASE_DIR variable first"
else
$(BUILD_DIR)/tests.x: CPPFLAGS += -isystem $(GTEST_DIR)/include -isystem $(GMOCK_DIR)/include
$(BUILD_DIR)/tests.x: LDFLAGS += -lpthread
$(BUILD_DIR)/tests.x: $(TEST_OBJ_FILES) $(filter-out $(BUILD_DIR)/main.cu.o, $(CU_OBJ_FILES)) $(CXX_OBJ_FILES) $(BUILD_DIR)/gmock_main.a | $(BUILD_DIR) 
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $^ $(LD_LIBS) -o $@
endif

$(BUILD_DIR)/gtest-all.o: $(GTEST_DIR)/src/gtest-all.cc $(GTEST_SRCS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) -I$(GMOCK_DIR) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/gmock-all.o: $(GMOCK_DIR)/src/gmock-all.cc $(GMOCK_SRCS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) -I$(GMOCK_DIR) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/gmock_main.o: $(GMOCK_DIR)/src/gmock_main.cc $(GMOCK_SRCS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) -I$(GMOCK_DIR) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/gmock_main.a :$(BUILD_DIR)/gtest-all.o $(BUILD_DIR)/gmock-all.o $(BUILD_DIR)/gmock_main.o
		$(AR) $(ARFLAGS) $@ $^

$(BUILD_DIR)/%_unittest.o: $(TEST_DIR)/%_unittest.cpp $(GTEST_HEADERS) $(GMOCK_HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) -I$(SRC_DIR) -I$(GMOCK_DIR) $(CXXFLAGS) -MMD -c $< -o $@ 

-include $(CU_DEP_FILES)
-include $(CXX_DEP_FILES)
-include $(TEST_DEP_FILES)
