package fr.oyashirox.raytracing

import fr.oyashirox.math.Vector
import fr.oyashirox.math.times

@Suppress("unused", "NOTHING_TO_INLINE")
data class Ray(val origin: Vector, val direction: Vector) {
    inline fun positionAt(t: Double) = origin + t*direction
}