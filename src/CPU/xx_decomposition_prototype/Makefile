PRODUCT = binary.x
SRC_DIR = src
TEST_DIR = test
BUILD_DIR = build
GTEST_BASE_DIR ?= googletest

CC=gcc
CCFLAGS = -g -std=gnu11 -pedantic -Wall -Wextra -O3
CPPFLAGS = # Preprocessor flags; -DEXPERIMENT_ONE
LDLIBS = -lrt
LDFLAGS = # paths to libraries; -L/path/to/library

C_HDR_FILES = $(wildcard $(SRC_DIR)/*.h)
C_SRC_FILES = $(wildcard $(SRC_DIR)/*.c) 
OBJ_FILES = $(C_SRC_FILES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
DEP_FILES = $(OBJ_FILES:.o=.d)

.DEFAULT: all
.PHONY: all
all: product

.PHONY: product
product: $(BUILD_DIR)/$(PRODUCT)

$(BUILD_DIR)/$(PRODUCT): $(OBJ_FILES)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@ 

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CCFLAGS) $(INCLUDES) -MMD -c $< -o $@

$(BUILD_DIR): 
	mkdir -p $@

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
	git clone git@github.com:google/googletest.git

ifeq ("$(wildcard $(GTEST_BASE_DIR))","")
$(BUILD_DIR)/tests.x:
	@echo "ERROR! Please run 'make googletest' or set the GTEST_BASE_DIR variable first"
else
$(BUILD_DIR)/tests.x: CPPFLAGS += -isystem $(GTEST_DIR)/include -isystem $(GMOCK_DIR)/include
$(BUILD_DIR)/tests.x: LDFLAGS += -lpthread
$(BUILD_DIR)/tests.x: $(TEST_OBJ_FILES) $(filter-out $(BUILD_DIR)/main.o, $(OBJ_FILES)) $(BUILD_DIR)/gmock_main.a | $(BUILD_DIR) 
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $^ -o $@
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

-include $(DEP_FILES)
-include $(TEST_DEP_FILES)
