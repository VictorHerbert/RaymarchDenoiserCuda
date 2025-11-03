#include "image.cuh"
#include "filter.cuh"
#include "test.cuh"

#include <vector>


FuncVector registered_funcs;
const std::string OUTPUT_PATH =  "test/";

void test() {
    if(registered_funcs.empty()){
        std::cout << "No tests found" << std::endl;
        return;
    }
    std::cout << registered_funcs.size() << " tests found" << std::endl;
    for (auto& [name, func] : registered_funcs) {
        try {
            auto start = std::chrono::high_resolution_clock::now();
            func();
            auto end = std::chrono::high_resolution_clock::now();
            double frameTime = std::chrono::duration<double, std::milli>(end - start).count();

            std::cout << "TEST " << name << ": passed with " << frameTime << " ms" <<std::endl;
        }
        catch (const std::runtime_error& e) {
            std::cout << "TEST " << name << ": failed \nREASON: " << e.what() << std::endl;
        }
        catch (...) {
            std::cout << "TEST " << name << ": failed" << std::endl;
        }
    }
}

SKIP(image_open){
    Image image("render/cornell/1/render.png");
}

SKIP(image_open_save){
    Image image("render/cornell/1/render.png");
    image.save(OUTPUT_PATH + "image.png");
}

SKIP(filter_cpu){
    Image render_img("render/cornell/1/render.png");
    Image albedo_img("render/cornell/1/albedo.png");
    Image normal_img("render/cornell/1/normal.png");

    int2 shape = render_img.shape;

    Image denoised_img(shape);

    waveletfilterCPU(
        {shape, render_img.vecBuffer.data(), normal_img.vecBuffer.data(), albedo_img.vecBuffer.data(), denoised_img.vecBuffer.data()},
        {5, 0, .1f, .1f, .1f, .1f}
    );

    denoised_img.save(OUTPUT_PATH + "filter_cpu.png");
}

SKIP(filter_gpu){
    Image render_img("render/cornell/1/render.png");
    Image albedo_img("render/cornell/1/albedo.png");
    Image normal_img("render/cornell/1/normal.png");

    int2 shape = render_img.shape;

    Image denoised_img(shape);

    CudaVector<uchar3> render(render_img.vecBuffer);
    CudaVector<uchar3> albedo(albedo_img.vecBuffer);
    CudaVector<uchar3> normal(normal_img.vecBuffer);
    CudaVector<uchar3> denoised(totalSize(shape));

    waveletfilterGPU(
        {shape, render.data(), normal.data(), albedo.data(), denoised.data()},
        {2, 0, .1f, .1f, .1f, .1f}
    );
    denoised.copyTo(denoised_img.vecBuffer);

    denoised_img.save(OUTPUT_PATH + "filter_gpu.png");
}

SKIP(video_gpu){
    int2 shape = {1920, 1080};
    int pixelCount = totalSize(shape);

    CudaVector<uchar3> render(pixelCount), albedo(pixelCount), normal(pixelCount), denoised(pixelCount);
    Framebuffer frame = {shape, render.data(), albedo.data(), normal.data(), denoised.data()};
    Image denoised_img(shape);

    DenoiseParams params = {2, 0, .1f, .1f, .1f, .1f};

    for(int frameIdx = 1; frameIdx < 5; frameIdx++){
        auto start = std::chrono::high_resolution_clock::now();
        render.from(Image("render/sponza/render/" + std::to_string(frameIdx) + ".png").vecBuffer);
        albedo.from(Image("render/sponza/albedo/" + std::to_string(frameIdx) + ".png").vecBuffer);
        normal.from(Image("render/sponza/normal/" + std::to_string(frameIdx) + ".png").vecBuffer);
        auto bp1 = std::chrono::high_resolution_clock::now();

        waveletfilterGPU(frame, params);

        cudaMemcpy(denoised_img.vecBuffer.data(), frame.denoised, sizeof(uchar3) * totalSize(shape), cudaMemcpyDeviceToHost);
        denoised_img.save(OUTPUT_PATH + "video_gpu" + std::to_string(frameIdx) + ".png");

        auto end = std::chrono::high_resolution_clock::now();

        double frameTime = std::chrono::duration<double, std::milli>(end - start).count();
        double diskTime = std::chrono::duration<double, std::milli>(bp1 - start).count();
        double cudaTime = std::chrono::duration<double, std::milli>(end - bp1).count();

        printf("Frame %i: disk %f kernel %f frame %f ms\n", frameIdx, diskTime, cudaTime, frameTime);
    }
}


TEST(video_gpu_stream){
    const int2 shape = {1920, 1080};
    const dim3 blockSize(16, 16);
    int frameCount = 20;
    int streamCount = 5;
    streamCount = min(frameCount, streamCount);
    

    int frameOutput[streamCount];

    dim3 gridSize((shape.x + 15) / 16, (shape.y + 15) / 16);
    int pixelCount = totalSize(shape);
    int byteCount = sizeof(Pixel) * pixelCount;

    float scaleSigma = 3*255*2;
    DenoiseParams params = {2, 0, .1f*scaleSigma, .1f*scaleSigma, .1f*scaleSigma, .1f*scaleSigma};

    CudaFramebuffer frames[streamCount];
    cudaStream_t streams[streamCount];

    for(int i = 0; i < streamCount; i++){
        frames[i].allocate(shape);
        cudaStreamCreate(&streams[i]);
        frameOutput[i] = -1;
    }

    for(int frameIdx = 0; frameIdx < frameCount + streamCount; frameIdx++){
        int streamIdx = frameIdx%streamCount;
        int videoIdx = (frameIdx-streamCount)%streamCount;
        auto& frame = frames[streamIdx];
        auto stream = streams[streamIdx];

        if(frameIdx >= streamCount){
            cudaStreamSynchronize(streams[streamIdx]);

            saveImage(
                OUTPUT_PATH + "video_gpu_stream" + std::to_string(videoIdx + 1) + ".png",
                frame.denoisedCPU, shape);
        }
        if(frameIdx < frameCount){
            frame.openImages("render/sponza/$type$/" + std::to_string(frameIdx+1) + ".png");

            waveletLevelsKernel<<<gridSize, blockSize, 0, stream>>>(frame, params);

            cudaMemcpyAsync(frame.denoisedCPU, frame.denoised, sizeof(Pixel) * totalSize(shape), cudaMemcpyDeviceToHost, stream);
        }
    }

    for(int i = 0; i < streamCount; i++)
        cudaStreamDestroy(streams[i]);
}

