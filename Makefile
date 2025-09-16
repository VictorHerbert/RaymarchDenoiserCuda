# Compiler and flags
NVCC = nvcc
CXX = $(NVCC)
CXXFLAGS_LK = -w -G -g -O0 -std=c++17 -arch=sm_75 -I./include -I./include/imgui -I./include/imgui/backends
CXXFLAGS = $(CXXFLAGS_LK) -dc
LDFLAGS =  -lglfw -lGL -ldl -lpthread

# Directories
SRC_DIR = src
BUILD_DIR = build
INCLUDE_DIR = include

# ImGui source files (relative to SRC_DIR or INCLUDE_DIR, adjust accordingly)
IMGUI_SRC = \
	$(INCLUDE_DIR)/imgui/imgui.cpp \
	$(INCLUDE_DIR)/imgui/imgui_draw.cpp \
	$(INCLUDE_DIR)/imgui/imgui_tables.cpp \
	$(INCLUDE_DIR)/imgui/imgui_widgets.cpp \
	$(INCLUDE_DIR)/imgui/imgui_demo.cpp \
	$(INCLUDE_DIR)/imgui/backends/imgui_impl_glfw.cpp \
	$(INCLUDE_DIR)/imgui/backends/imgui_impl_opengl3.cpp

# Convert ImGui sources to build object files
IMGUI_OBJ = $(patsubst $(INCLUDE_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(IMGUI_SRC))

# Source files from your src directory
SRC = $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*.cu)
OBJ = $(SRC:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
OBJ := $(OBJ:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)

# Combine all objects: src + ImGui
ALL_OBJ = $(OBJ) $(IMGUI_OBJ)

BLENDER = "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"

# Targets
TARGET = $(BUILD_DIR)/main

# Default target
all: $(TARGET)

run: $(TARGET)
	@mkdir -p build/sample
	@./$(TARGET) -r sample/cornell/32/Render.png -s build/sample/cornell32.png

test: $(TARGET)
	@mkdir -p build/test
	@./$(TARGET) -t

window:$(TARGET)
	@./$(TARGET) -gui

debug: $(TARGET)
	cuda-gdb -ex=run -ex=quit ./$(TARGET)

sanitize: $(TARGET)
	compute-sanitizer --tool memcheck --show-backtrace=yes ./$(TARGET) -gui

prof:
	@nsys profile -o build/prof ./$(TARGET) -t
	nsight-sys build/prof.nsys-rep

# Link main including ImGui objects
$(TARGET): $(ALL_OBJ)
	@$(NVCC) $(CXXFLAGS_LK) -o $@ $^ $(LDFLAGS)

# Compile rules with dependency generation for .cpp in src and include directories
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	@$(NVCC) $(CXXFLAGS) -M -MT $@ $< > $(BUILD_DIR)/$*.d
	@$(NVCC) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	@$(NVCC) $(CXXFLAGS) -M -MT $@ $< > $(BUILD_DIR)/$*.d
	$(NVCC) $(CXXFLAGS) -c $< -o $@

# Compile ImGui .cpp files from include directory
$(BUILD_DIR)/%.o: $(INCLUDE_DIR)/%.cpp
	@mkdir -p $(dir $@)
	@$(NVCC) $(CXXFLAGS) -M -MT $@ $< > $(BUILD_DIR)/$*.d
	@$(NVCC) $(CXXFLAGS) -c $< -o $@

render: scenes/cornell.blend
	$(BLENDER) -b scenes/cornell.blend -P scripts/setup_passes.py -- 1
	$(BLENDER) -b scenes/cornell.blend -P scripts/setup_passes.py -- 4
	$(BLENDER) -b scenes/cornell.blend -P scripts/setup_passes.py -- 8
	$(BLENDER) -b scenes/cornell.blend -P scripts/setup_passes.py -- 16
	$(BLENDER) -b scenes/cornell.blend -P scripts/setup_passes.py -- 32
	$(BLENDER) -b scenes/cornell.blend -P scripts/setup_passes.py -- 8192

.PHONY: render all clean run debug sanitize

# Clean
clean:
	@rm -rf $(BUILD_DIR)/*
	@mkdir -p $(BUILD_DIR)

# Include auto-generated dependency files
-include $(ALL_OBJ:.o=.d)
