#ifndef VIDEO_H
#define VIDEO_H

#include <string>

void decodeVideo(std::string filepath, void (*callback)(uchar3*, int2));

#endif