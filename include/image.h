#ifndef IMAGE_H
#define IMAGE_H

#include "utils.h"
#include "vector.h"
#include "gbuffer.h"

#include "cuda_runtime.h"
#include <string>
#include <vector>

/**
 * @brief Represents an image in memory.
 * 
 * Contains the pixel data and the dimensions of the image.
 */
struct Image {
    int3 shape;   /**< Dimensions of the image (width, height, depth or channels). */
    byte* data;   /**< Pointer to the raw image data. */

    /**
     * @brief Constructs an image with the specified shape.
     * 
     * Allocates memory for the image data based on the shape.
     * 
     * @param shape Dimensions of the image.
     */
    Image(int3 shape);

    /**
     * @brief Constructs an image from existing data.
     * 
     * @param data Pointer to existing image data.
     * @param shape Dimensions of the image.
     */
    Image(byte* data, int3 shape);

    /**
     * @brief Loads an image from a file.
     * 
     * @param filename Path to the image file.
     * @param channels Number of channels to load (e.g., 1 for grayscale, 3 for RGB).
     */
    Image(std::string filename, int channels);

    /**
     * @brief Destructor.
     * 
     * Frees the allocated image data if necessary.
     */
    ~Image();

    /**
     * @brief Saves the image to a file.
     * 
     * @param filename Path where the image will be saved.
     */
    void save(std::string filename);

    /**
     * @brief Static method to save image data to a file without creating an Image object.
     * 
     * @param filename Path where the image will be saved.
     * @param data Pointer to the image data.
     * @param shape Dimensions of the image.
     */
    static void save(std::string filename, byte* data, int3 shape);
};


#endif