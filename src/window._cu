#include "window.cuh"

#include "third_party/helper_math.h"
#include "image.cuh"
#include "filter.cuh"
#include "math.h"
#include "raymarch.cuh"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>

#include <iostream>
#include <chrono>

GLuint image_texture;
int display_w, display_h;

// Boilerplate from reference code
int window(){
    if (!glfwInit()){
        throw std::runtime_error("Failed to init GLFW");
        return -1;
    }
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Denoiser", NULL, NULL);

    if (!window){
        throw std::runtime_error("Failed to create window");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Setup Dear ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();

        glfwGetFramebufferSize(window, &display_w, &display_h);

        renderUI();

        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // Cleanup
    glDeleteTextures(1, &image_texture);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

bool raymarchCPU = false;
bool denoiseCPU = false;

int kersize = 2;
int depth = 1;
bool cpu = false;

DenoiseParams denoiseParams = {5, .5, .5, .5, .5};
float minSigma = 0.05;
float maxSigma = 2.0f;

auto previousFrameCheckpoint = std::chrono::high_resolution_clock::now();

Camera camera = {
    {0,-.5,-2}, // .pos
    {0,0,1},    // forward
    {1,1},      // plane
    1           // dist
};


Image render_img("render/cornell/1/render.png");
Image albedo_img("render/cornell/1/albedo.png");
Image normal_img("render/cornell/1/normal.png");

int2 shape = render_img.shape;

Image denoised_img(shape);

CudaVector<uchar3> render_cv(render_img.vecBuffer);
CudaVector<uchar3> albedo_cv(albedo_img.vecBuffer);
CudaVector<uchar3> normal_cv(normal_img.vecBuffer);
CudaVector<uchar3> denoised_cv(totalSize(shape));

void renderUI() {
    using clock = std::chrono::high_resolution_clock;

    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(0, 0));

    const float aspect_ratio = (float)shape.x / (float)shape.y; // image ratio
    ImGui::SetNextWindowSizeConstraints(
        ImVec2(200, 200),
        ImVec2(display_w, display_h),
        [](ImGuiSizeCallbackData* data) {
            float aspect = *((float*)data->UserData);
            float width = data->DesiredSize.x;
            float height = data->DesiredSize.y;

            if (width / height > aspect)
                width = height * aspect;
            else
                height = width / aspect;

            data->DesiredSize = ImVec2(width, height);
        },
        (void*)&aspect_ratio
    );

    ImGui::SetNextWindowSize(ImVec2(display_w, display_h));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::Begin("Background", nullptr,
        ImGuiWindowFlags_NoDecoration |
        ImGuiWindowFlags_NoInputs |         // ignore mouse clicks
        ImGuiWindowFlags_NoBringToFrontOnFocus);

// -------------------------------------------------------------------------------

    auto frameCheckpoint = clock::now();
    std::chrono::duration<double, std::milli> frameTime = (frameCheckpoint - previousFrameCheckpoint);
    previousFrameCheckpoint = frameCheckpoint;

// -------------------------------------------------------------------------------

    static std::vector<Solid> solids = {
        {Light,     {0,1,0}, {.3,.3,.3}, {0,0,1}},

        {Box,      {0,-1,0},  {1,0.01,1},   {1,1,1}},
        {Box,    {0,0.5,0},   {1,0.01,1},   {1,1,1}},
        {Box,   {0,0,0.5},   {1,1,0.01},   {1,1,1}},
        {Box,   {-0.5,0,0},  {0.01,1,1},   {1,0,0}},
        {Box,  {0.5,0,0},   {0.01,1,1},   {0,1,0}},

        {Box,       {-0.15,-0.25,0.15},  {0.2,0.25,0.2},   {0,0,1}},
        {Box,       {0.2,-0.35,-0.1},    {0.15,0.35,0.15}, {1,1,0}}
    };

    static CudaVector<Solid> scene(solids.size());
    static CudaVector<uchar3> render(totalSize(shape)), normal(totalSize(shape)), albedo(totalSize(shape));
    static CPUVector<uchar3> render_cpu(totalSize(shape));

    scene.from(solids);
    raymarchSceneGPU(camera, {scene.size(), scene.data()}, {shape, render.data(), normal.data(), albedo.data(), nullptr});
    render.copyTo(render_cpu);
    
    GLuint img_texture = textureFromBuffer(render_cpu.data(), shape);
    
    /*waveletfilterGPU(
        {shape, render_cv.data(), normal_cv.data(), albedo_cv.data(), denoised_cv.data()},
        denoiseParams
    );
    denoised_cv.copyTo(denoised_img.vecBuffer);

    static GLuint img_texture;
    img_texture = textureFromBuffer(denoised_img.vecBuffer.data(), shape);*/


// -------------------------------------------------------------------------------
    std::chrono::duration<double, std::milli> processingTime = clock::now() - frameCheckpoint;
// -------------------------------------------------------------------------------

    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImGui::Image((ImTextureID)(intptr_t) img_texture, avail, ImVec2(0, 0), ImVec2(1, 1)); // fit image
    ImGui::End();
    ImGui::PopStyleVar(2);

    // --- Menu window remains unchanged ---
    ImGui::SetNextWindowPos(ImVec2(800, 30), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(300, 0));

// -------------------------------------------------------------------------------

    ImGui::Begin("Menu");

    ImGui::SeparatorText("Performance");
    ImGui::Text("FPS: %.1f", 1000/frameTime.count());
    ImGui::Text("UI Overhead: %.2f %%", (1-processingTime.count()/frameTime.count())*100);
    ImGui::Text("Frame time: %.2f/%.2f ms", processingTime.count(), frameTime.count());

    ImGui::SeparatorText("Viewport");
    ImGui::InputInt2("Resolution", (int*) &shape, .1);

    const char* items[] = { "Render", "Denoised", "Albedo", "Normal"};
    static int item_selected_idx = 0;

    const char* combo_preview_value = items[item_selected_idx];
    if (ImGui::BeginCombo("Channel", combo_preview_value)){
        for (int n = 0; n < IM_ARRAYSIZE(items); n++){
            if (ImGui::Selectable(items[n], item_selected_idx == n))
                item_selected_idx = n;
        }
        ImGui::EndCombo();
    }

    ImGui::SeparatorText("Scene");

    /*if(ImGui::CollapsingHeader("Camera")){
        ImGui::DragFloat3("Pos", (float*) &camera.pos, .1);
        ImGui::DragFloat3("Dir", (float*) &camera.forward, .1);
        ImGui::DragFloat2("Plane", (float*) &camera.plane, .1);
        ImGui::DragFloat("Dist", (float*) &camera.dist, .1);
    };
    camera.forward = normalize(camera.forward);*/
    float dummy []= {.5,.5,.5};

    const char* objTypeNames[] = {"Light", "Sphere", "Box"};

    /*for(int i = 0; i < scene.size(); i++){
        ImGui::PushID(i);
        if(ImGui::CollapsingHeader(objTypeNames[solids[i].type])){
            ImGui::DragFloat3("Pos", (float*) &solids[i].pos, .1);
            ImGui::DragFloat3("Scale", (float*) &solids[i].scale, .1);
            ImGui::ColorEdit3("Color", (float*) &solids[i].col, ImGuiColorEditFlags_Float);
        }
        ImGui::PopID();
    }*/

    ImGui::SeparatorText("Setup");

    if(ImGui::CollapsingHeader("Raymarch")){
        ImGui::Checkbox("CPU Raymarch", &raymarchCPU);
        ImGui::Checkbox("Anti-aliasing", (bool*) dummy);
    }
    if(ImGui::CollapsingHeader("Denoising")){
        ImGui::Checkbox("CPU Denoise", &denoiseCPU);
        ImGui::Spacing();
        ImGui::SliderInt("Iterations", &denoiseParams.depth, 0, 10);
        ImGui::Spacing();
        ImGui::SliderFloat("Sigma Color", &denoiseParams.sigmaColor, .05, .2);
        ImGui::SliderFloat("Sigma Albedo", &denoiseParams.sigmaAlbedo, minSigma, maxSigma);
        ImGui::SliderFloat("Sigma Normal", &denoiseParams.sigmaNormal, minSigma, maxSigma);
    }

    ImGui::End();
    ImGui::Render();
}

GLuint textureFromBuffer(uchar3* image, int2 shape){
    GLuint tex_id;
    glGenTextures(1, &tex_id);
    glBindTexture(GL_TEXTURE_2D, tex_id);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, shape.x, shape.y, 0, GL_RGB, GL_UNSIGNED_BYTE, image);

    return tex_id;
}
