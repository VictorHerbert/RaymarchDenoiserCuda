# ===========================================================================
#                           Compiler and Flags
# ===========================================================================

NVCC = nvcc
CXX = $(NVCC)

CXXFLAGS_LK = -w -G -g -O3 -std=c++17 -arch=sm_75 -I./include
CXXFLAGS = $(CXXFLAGS_LK) -dc

MKDIR = mkdir
RM = rm

# ===========================================================================
#                               Directories
# ===========================================================================

SRC_DIR     = src
BUILD_DIR   = build
INCLUDE_DIR = include

# ===========================================================================
#                        Source and Object Files
# ===========================================================================

SRC        = $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*.cu)
OBJ        = $(SRC:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
OBJ        := $(OBJ:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)

# ===========================================================================
#                                 Target
# ===========================================================================

TARGET = $(BUILD_DIR)/main

# ===========================================================================
#                               Build Rules
# ===========================================================================

$(TARGET): $(OBJ)
	@echo "Linking $< into $@"
	@$(NVCC) $(CXXFLAGS_LK) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Recompiling $< into $@"
	@$(NVCC) $(CXXFLAGS) -M -MT $@ $< > $(BUILD_DIR)/$*.d
	@$(NVCC) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@echo "Recompiling $< into $@"
	@$(NVCC) $(CXXFLAGS) -M -MT $@ $< > $(BUILD_DIR)/$*.d
	@$(NVCC) $(CXXFLAGS) -c $< -o $@

-include $(ALL_OBJ:.o=.d)

# ===========================================================================
#                                 Tasks
# ===========================================================================

test: $(TARGET)
	@./$(TARGET) -t

run_no_args: $(TARGET)
	@./$(TARGET)

memcheck: $(TARGET)
	compute-sanitizer --tool memcheck --show-backtrace=yes --log-file $(BUILD_DIR)/memcheck.log ./$(TARGET) -t

doxygen:
	doxygen Doxyfile

all: memcheck doxygen

.PHONY: all clean test

# ===========================================================================
#                                  Clean
# ===========================================================================

test_clean: $(TARGET)
	@$(RM) -rf test/*
	@./$(TARGET) -t

clean:
	$(RM) -rf $(BUILD_DIR)/*
	$(RM) -rf test/*
	$(MKDIR) -p test



