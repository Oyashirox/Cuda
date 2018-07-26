package fr.oyashirox

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.*
import jcuda.driver.JCudaDriver.*
import kotlin.system.measureTimeMillis

fun main(args: Array<String>) {
    val numElements = 1 shl 20

    // Allocate and fill the host input data
    val hostInputA = FloatArray(numElements)
    val hostInputB = FloatArray(numElements)
    for (i in 0 until numElements) {
        hostInputA[i] = i.toFloat()
        hostInputB[i] = i.toFloat()
    }

    var result = FloatArray(0)
    var time = measureTimeMillis {
        result = executeOnGPU(hostInputA, hostInputB)
    }
    var passed = checkResult(result)
    println("Test GPU ${if(passed) "SUCCESS" else "FAILED"} in $time ms for $numElements elements")

    time = measureTimeMillis {
        result = executeOnCPU(hostInputA, hostInputB)
    }
    passed = checkResult(result)
    println("Test CPU ${if(passed) "SUCCESS" else "FAILED"} in $time ms for $numElements elements")

}

fun checkResult(result: FloatArray): Boolean {
    // Verify the result
    var passed = true
    for (i in 0 until result.size) {
        val expected = (i + i).toFloat()
        if (Math.abs(result[i] - expected) > 1e-5) {
            println(
                    "At index " + i + " found " + result[i] +
                            " but expected " + expected)
            passed = false
            break
        }
    }

   return passed
}

fun executeOnGPU(hostInputA: FloatArray, hostInputB: FloatArray): FloatArray {
    val numElements = hostInputA.size
    val filePath = object : Any() {}.javaClass
            .getResource("cudaKernel.ptx")
            .path.drop(1) // remove leading / (no idea why it is here)

    // Initialize the driver and create a context for the first device.
    cuInit(0)
    val device = CUdevice()
    cuDeviceGet(device, 0)
    val context = CUcontext()
    cuCtxCreate(context, 0, device)

    val module = CUmodule()
    JCudaDriver.cuModuleLoad(module, filePath)
    val function = CUfunction()
    JCudaDriver.cuModuleGetFunction(function, module, "add")

    // Allocate the device input data, and copy the
    // host input data to the device
    val deviceInputA = CUdeviceptr()
    cuMemAlloc(deviceInputA, (numElements * Sizeof.FLOAT).toLong())
    cuMemcpyHtoD(deviceInputA, Pointer.to(hostInputA),
            (numElements * Sizeof.FLOAT).toLong())
    val deviceInputB = CUdeviceptr()
    cuMemAlloc(deviceInputB, (numElements * Sizeof.FLOAT).toLong())
    cuMemcpyHtoD(deviceInputB, Pointer.to(hostInputB),
            (numElements * Sizeof.FLOAT).toLong())

    // Allocate device output memory
    val deviceOutput = CUdeviceptr()
    cuMemAlloc(deviceOutput, (numElements * Sizeof.FLOAT).toLong())

    // Set up the kernel parameters: A pointer to an array
    // of pointers which point to the actual values.
    val kernelParameters = Pointer.to(
            Pointer.to(intArrayOf(numElements)),
            Pointer.to(deviceInputA),
            Pointer.to(deviceInputB),
            Pointer.to(deviceOutput)
    )

    // Call the kernel function.
    val blockSizeX = 256
    val gridSizeX = Math.ceil(numElements.toDouble() / blockSizeX).toInt()

    cuLaunchKernel(function,
            gridSizeX, 1, 1, // Grid dimension
            blockSizeX, 1, 1, // Block dimension
            0, null // Kernel- and extra parameters
            , // Shared memory size and stream
            kernelParameters, null
    )
    cuCtxSynchronize()

    // Allocate host output memory and copy the device output
    // to the host.
    val hostOutput = FloatArray(numElements)
    cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput,
            (numElements * Sizeof.FLOAT).toLong())

    // Clean up.
    cuMemFree(deviceInputA)
    cuMemFree(deviceInputB)
    cuMemFree(deviceOutput)

    return hostOutput
}

fun executeOnCPU(hostInputA: FloatArray, hostInputB: FloatArray): FloatArray  {
    val numElements = hostInputA.size
    val result = FloatArray(numElements)
    for (i in 0 until numElements) {
        result[i] = hostInputA[i] + hostInputB[i]
    }
    return result
}