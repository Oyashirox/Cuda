package fr.oyashirox

import jcuda.runtime.JCuda
import jcuda.Pointer

fun main(args: Array<String>) {
    val pointer = Pointer()
    JCuda.cudaMalloc(pointer, 4)
    println("Pointer: $pointer")
    JCuda.cudaFree(pointer)
}