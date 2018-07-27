package fr.oyashirox

import fr.oyashirox.math.Image
import fr.oyashirox.math.Vector
import fr.oyashirox.raytracing.Camera
import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.*
import jcuda.driver.JCudaDriver.*
import java.awt.Desktop
import java.awt.image.BufferedImage
import java.nio.file.Paths
import javax.imageio.ImageIO

fun main(args: Array<String>) {
       val camera = Camera(
            lookfrom = Vector(-2.0, 2.0, 1.0),
            lookat = Vector(0.0, 0.0, -1.0)
    )

    // Host memory
    val width = 1920
    val height = 1080
    val colors = FloatArray(width * height * 3) { 0.5f }
    val cameraPos = camera.lookfrom.toFloatArray()
    val cameraLowerLeft = camera.lowerLeft.toFloatArray()
    val cameraHorizontal = camera.horizontal.toFloatArray()
    val cameraVertical = camera.vertical.toFloatArray()
    val world = floatArrayOf(
            0.0f, 0.0f, -1.0f, 0.5f
    )
    val numberOfSphere = 1

    //cuda init
    var status: Int
    val filePath = object : Any() {}.javaClass
            .getResource("cudaRaytracing.ptx")
            .path.drop(1) // remove leading / (no idea why it is here)

    JCudaDriver.setExceptionsEnabled(true)
    // Initialize the driver and create a context for the first device.
    cuInit(0)
    val device = CUdevice()
    cuDeviceGet(device, 0)
    val context = CUcontext()
    status = cuCtxCreate(context, 0, device)
    println("status cuCtxCreate: $status")

    val module = CUmodule()
    status = JCudaDriver.cuModuleLoad(module, filePath)
    println("status cuModuleLoad: $status")

    val function = CUfunction()
    status = JCudaDriver.cuModuleGetFunction(function, module, "raytracing")
    println("status cuModuleGetFunction: $status")

    // Allocate the device input data, and copy the
    // host input data to the device
    val deviceCameraPos = CUdeviceptr()
    cuMemAlloc(deviceCameraPos, (3 * Sizeof.FLOAT).toLong())
    cuMemcpyHtoD(deviceCameraPos, Pointer.to(cameraPos),
            (3 * Sizeof.FLOAT).toLong())
    val deviceCameraLowerLeft = CUdeviceptr()
    cuMemAlloc(deviceCameraLowerLeft, (3 * Sizeof.FLOAT).toLong())
    cuMemcpyHtoD(deviceCameraLowerLeft, Pointer.to(cameraLowerLeft),
            (3 * Sizeof.FLOAT).toLong())
    val deviceCameraHorizontal = CUdeviceptr()
    cuMemAlloc(deviceCameraHorizontal, (3 * Sizeof.FLOAT).toLong())
    cuMemcpyHtoD(deviceCameraHorizontal, Pointer.to(cameraHorizontal),
            (3 * Sizeof.FLOAT).toLong())
    val deviceCameraVertical = CUdeviceptr()
    cuMemAlloc(deviceCameraVertical, (3 * Sizeof.FLOAT).toLong())
    cuMemcpyHtoD(deviceCameraVertical, Pointer.to(cameraVertical),
            (3 * Sizeof.FLOAT).toLong())

    val deviceWorld = CUdeviceptr()
    cuMemAlloc(deviceWorld, (numberOfSphere * 4 * Sizeof.FLOAT).toLong())
    cuMemcpyHtoD(deviceWorld, Pointer.to(world),
            (numberOfSphere * 4 * Sizeof.FLOAT).toLong())

    // Allocate device output memory
    val deviceColorsOutput = CUdeviceptr()
    cuMemAlloc(deviceColorsOutput, (width * height * 3 * Sizeof.FLOAT).toLong())

    // Set up the kernel parameters: A pointer to an array
    // of pointers which point to the actual values.
    val kernelParameters = Pointer.to(
            Pointer.to(intArrayOf(width)),
            Pointer.to(intArrayOf(height)),
            Pointer.to(deviceColorsOutput),
            Pointer.to(deviceCameraPos),
            Pointer.to(deviceCameraLowerLeft),
            Pointer.to(deviceCameraHorizontal),
            Pointer.to(deviceCameraVertical),

            Pointer.to(deviceWorld),
            Pointer.to(intArrayOf(numberOfSphere))
            )

    // Call the kernel function.
    val blockSizeX = 256
    val gridSizeX = Math.ceil((width * height).toDouble() / blockSizeX).toInt()

    status = cuLaunchKernel(function,
            gridSizeX, 1, 1, // Grid dimension
            blockSizeX, 1, 1, // Block dimension
            0, null // Kernel- and extra parameters
            , // Shared memory size and stream
            kernelParameters, null
    )
    cuCtxSynchronize()

    println("status cuLaunchKernel: $status")

    cuMemcpyDtoH(Pointer.to(colors), deviceColorsOutput,
            (width * height * 3 * Sizeof.FLOAT).toLong())



    val completeImage = Image(width, height)
    completeImage.setData(colors)
    val bufferedImage = BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB)
    bufferedImage.setRGB(0, 0, width, height, completeImage.getColorArray().map { it.argbColor }.toIntArray(), 0, width)

    val file = Paths.get(".", "generated.png").toAbsolutePath().normalize().toFile()
    println("Image saved to : ${file.absolutePath}")
    ImageIO.write(bufferedImage, "png", file)
    Desktop.getDesktop().open(file)

    // Clean up.
    cuMemFree(deviceCameraPos)
    cuMemFree(deviceCameraLowerLeft)
    cuMemFree(deviceCameraHorizontal)
    cuMemFree(deviceCameraVertical)
    cuMemFree(deviceWorld)
    cuMemFree(deviceColorsOutput)
}
