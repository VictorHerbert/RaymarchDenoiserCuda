#ifndef WINDOW_H
#define WINDOW_H

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>

#include "image.cuh"

int window();
void renderUI();
GLuint textureFromBuffer(float3* image, int2 shape);

#endif