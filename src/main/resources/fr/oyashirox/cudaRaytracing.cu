__device__ void normalize(float vector[]) {
    float norm = norm3df(vector[0], vector[1], vector[2]);
    vector[0] /= norm;
    vector[1] /= norm;
    vector[2] /= norm;
}

__device__ void sphereNormal(float center[], float position[], float result[]) {
    result[0] = position[0] - center[0];
    result[1] = position[1] - center[1];
    result[2] = position[2] - center[2];
    normalize(result);
}

__device__ void convertToRay(
        float u, float v, float* rayDirection,
        float* cameraPos, float* cameraLowerLeft, float* cameraHorizontal, float* cameraVertical) {
    rayDirection[0] =  cameraLowerLeft[0] + (u * cameraHorizontal[0]) + (v * cameraVertical[0]) - cameraPos[0];
    rayDirection[1] =  cameraLowerLeft[1] + (u * cameraHorizontal[1]) + (v * cameraVertical[1]) - cameraPos[1];
    rayDirection[2] =  cameraLowerLeft[2] + (u * cameraHorizontal[2]) + (v * cameraVertical[2]) - cameraPos[2];
//    float norm = norm3df(rayDirection[0], rayDirection[1], rayDirection[2]);
//    rayDirection[0] /= norm;
//    rayDirection[1] /= norm;
//    rayDirection[2] /= norm;
}

__device__ void hit(float rayPos[], float rayDir[], float* worldSphere, int sphereNumber, float result[]) {
    int i;
    float spherePos[3] = {0.0, 0.0, 0.0};
    float radius;
    float rayOffset[3] = {0.0, 0.0, 0.0};
    float a;
    float b;
    float c;
    float delta;
    float distance;
    float position[3];
    float normal[3] = {0.0, 0.0, 0.0};

    for(i = 0; i < sphereNumber; ++i) {
        spherePos[0] = worldSphere[i * 4 + 0];
        spherePos[1] = worldSphere[i * 4 + 1];
        spherePos[2] = worldSphere[i * 4 + 2];
        radius = worldSphere[i * 4 + 3];

        rayOffset[0] = rayPos[0] - spherePos[0];
        rayOffset[1] = rayPos[1] - spherePos[1];
        rayOffset[2] = rayPos[2] - spherePos[2];

        a = rayDir[0] * rayDir[0] + rayDir[1] * rayDir[1] + rayDir[2] * rayDir[2];
        b = 2.0 * (rayDir[0] * rayOffset[0] + rayDir[1] * rayOffset[1] + rayDir[2] * rayOffset[2]);
        c = (rayOffset[0] * rayOffset[0] + rayOffset[1] * rayOffset[1] + rayOffset[2] * rayOffset[2]) - (radius * radius);
        delta = b * b - 4 * a * c;

        if(delta >= 0) {
            distance = (-b - sqrtf(delta)) / (2.0 * a);
            if(distance > 0) {
                position[0] = rayPos[0] + rayDir[0] * distance;
                position[1] = rayPos[1] + rayDir[1] * distance;
                position[2] = rayPos[2] + rayDir[2] * distance;
                sphereNormal(spherePos, position, normal);
                break;
            }
        }
    }

    //return
    result[0] = normal[0];
    result[1] = normal[1];
    result[2] = normal[2];
}

extern "C"
// don't forget to compile with "nvcc -ptx cudaRaytracing.cu -o cudaRaytracing.ptx
// And to move the ptx file in the resources !
/** @param width image width
    @param height image height
    @param colors r,g,b for each pixel
    @param world sphere coordinates: x, y, z, radius, 4 floats per sphere
    @param numberOfSphere number of sphere in the world*/
__global__ void raytracing(
    int width, int height, float* colors,
    float* cameraPos, float* cameraLowerLeft, float* cameraHorizontal, float* cameraVertical,
    float* world, int numberOfSphere) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // index is the pixel number as in a 1D array
    int stride = blockDim.x * gridDim.x;
    int x = index % width;
    int y = index / width;
    float u = (float)x / width;
    float v = 1.0 - ((float)y / height);
    float normal[3];

    float rayDirection[3] = {0.0, 0.0, 0.0}; // direction x, y, z

    for (int i = index; i < width * height; i += stride) {
        convertToRay(u, v, rayDirection, cameraPos, cameraLowerLeft, cameraHorizontal, cameraVertical);
        hit(cameraPos, rayDirection, world, numberOfSphere, normal);
        if(normal[0] != 0.0 || normal[1] != 0.0 || normal[2] != 0.0 ) {
            colors[index * 3] = normal[0];
            colors[index * 3 + 1] = normal[1];
            colors[index * 3 + 2] = normal[2];
        } else {
            colors[index * 3] = 0.0;
            colors[index * 3 + 1] = 0.0;
            colors[index * 3 + 2] = 0.0;
        }

    }
}