PRODUCT = deqn3d.x
SRC_DIR = src
BUILD_DIR = build

CXXFLAGS=g -Wall -Wextra -pedantic -O3
NVCCFLAGS=-lineinfo -std=c++11

CU_HDR_FILES = $(wildcard $(SRC_DIR)/*.cuh)
CU_SRC_FILES = $(wildcard $(SRC_DIR)/*.cu) 
CU_OBJ_FILES = $(CU_SRC_FILES:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.cu.o)
CU_DEP_FILES = $(CU_OBJ_FILES:.o=.d)

.DEFAULT: all
.PHONY: all
all: $(BUILD_DIR)/$(PRODUCT)

$(BUILD_DIR)/$(PRODUCT): $(CU_OBJ_FILES)
	$(NVCC) $(LDFLAGS) $^ $(LDLIBS) -o $@ 

$(BUILD_DIR)/%.cu.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCCINCLUDES) -c $< -o $@

$(BUILD_DIR):
	@mkdir -p $@

.PHONY: clean
clean:
	rm -rf $(PRODUCT)
	rm -rf $(CU_OBJ_FILES)
	rm -rf $(CU_DEP_FILES)
	rm -rf $(BUILD_DIR)

-include $(CU_DEP_FILES) # has to be below all: otherwise it subsumes the default target
