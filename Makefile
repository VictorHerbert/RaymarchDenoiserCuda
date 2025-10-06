# =========================
# Compiler and Flags
# =========================

NVCC = nvcc
CXX = $(NVCC)

CXXFLAGS_LK = -w -G -g -O0 -std=c++17 -arch=sm_75 -I./include -I./include/imgui -I./include/imgui/backends
CXXFLAGS = $(CXXFLAGS_LK) -dc

ifeq ($(OS),Windows_NT)
    LDFLAGS = -lglfw3dll -lopengl32
    MKDIR = wsl mkdir
else
    LDFLAGS = -lglfw -lGL -ldl -lpthread
    MKDIR = mkdir
endif

RM = rm

# =========================
# Directories
# =========================

SRC_DIR     = src
BUILD_DIR   = build
INCLUDE_DIR = include

# =========================
# ImGui Source Files
# =========================

IMGUI_SRC = \
    $(INCLUDE_DIR)/imgui/imgui.cpp \
    $(INCLUDE_DIR)/imgui/imgui_draw.cpp \
    $(INCLUDE_DIR)/imgui/imgui_tables.cpp \
    $(INCLUDE_DIR)/imgui/imgui_widgets.cpp \
    $(INCLUDE_DIR)/imgui/imgui_demo.cpp \
    $(INCLUDE_DIR)/imgui/backends/imgui_impl_glfw.cpp \
    $(INCLUDE_DIR)/imgui/backends/imgui_impl_opengl3.cpp

# =========================
# Source and Object Files
# =========================
SRC        = $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*.cu)
OBJ        = $(SRC:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
OBJ        := $(OBJ:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)
IMGUI_OBJ  = $(patsubst $(INCLUDE_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(IMGUI_SRC))
ALL_OBJ    = $(OBJ) $(IMGUI_OBJ)

BLENDER = "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"

# =========================
# Target
# =========================
TARGET = $(BUILD_DIR)/main

# =========================
# Build Rules
# =========================
all: $(TARGET)

test: $(TARGET)
	@./$(TARGET) -t

window:
	@./$(TARGET) -gui

$(TARGET): $(ALL_OBJ)
	$(NVCC) $(CXXFLAGS_LK) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@$(NVCC) $(CXXFLAGS) -M -MT $@ $< > $(BUILD_DIR)/$*.d
	@$(NVCC) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@$(NVCC) $(CXXFLAGS) -M -MT $@ $< > $(BUILD_DIR)/$*.d
	$(NVCC) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(INCLUDE_DIR)/%.cpp
	@$(NVCC) $(CXXFLAGS) -c $< -o $@

doxygen:
	doxygen Doxyfile


# =========================
# Clean
# =========================
clean:
	$(RM) -rf $(BUILD_DIR)/*
	$(MKDIR) -p build/imgui/backends
	$(MKDIR) -p build/output

.PHONY: render all clean run
