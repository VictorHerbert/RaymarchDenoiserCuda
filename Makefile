# Compiler and flags
NVCC = nvcc
CXX = $(NVCC)
CXXFLAGS_LK = -w -G -g -O0 -std=c++17 -arch=sm_75 -I./include -I./include/imgui -I./include/imgui/backends
CXXFLAGS = $(CXXFLAGS_LK) -dc


ifeq ($(OS),Windows_NT)
    LDFLAGS = -lglfw3dll -lopengl32
	MKDIR = wsl mkdir
else
    LDFLAGS =  -lglfw -lGL -ldl -lpthread
	MKDIR = mkdir
endif


# Directories
SRC_DIR 	= src
BUILD_DIR	= build
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

SRC			= $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*.cu)
OBJ			= $(SRC:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
OBJ			:= $(OBJ:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)
IMGUI_OBJ 	= $(patsubst $(INCLUDE_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(IMGUI_SRC))
ALL_OBJ 	= $(OBJ) $(IMGUI_OBJ)

BLENDER = "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"

# Targets
TARGET = $(BUILD_DIR)/main

all: $(TARGET)

window:$(TARGET)
	@./$(TARGET) -gui

debug: $(TARGET)
	cuda-gdb -ex=run -ex=quit ./$(TARGET)

sanitize: $(TARGET)
	compute-sanitizer --tool memcheck --show-backtrace=yes ./$(TARGET) -gui

# Link main including ImGui objects
$(TARGET): $(ALL_OBJ)
	@$(NVCC) $(CXXFLAGS_LK) -o $@ $^ $(LDFLAGS)

# Compile rules with dependency generation for .cpp in src and include directories
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@$(NVCC) $(CXXFLAGS) -M -MT $@ $< > $(BUILD_DIR)/$*.d
	@$(NVCC) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@$(NVCC) $(CXXFLAGS) -M -MT $@ $< > $(BUILD_DIR)/$*.d
	$(NVCC) $(CXXFLAGS) -c $< -o $@

# Compile ImGui .cpp files from include directory
$(BUILD_DIR)/%.o: $(INCLUDE_DIR)/%.cpp
	@$(NVCC) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)/*
	$(MKDIR) -p build/imgui/backends

.PHONY: render all clean run debug sanitize