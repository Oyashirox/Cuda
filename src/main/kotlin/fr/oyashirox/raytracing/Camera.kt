package fr.oyashirox.raytracing

import fr.oyashirox.math.Vector
import fr.oyashirox.math.times
import kotlin.math.PI
import kotlin.math.tan

@Suppress("unused", "MemberVisibilityCanBePrivate", "CanBeParameter")
/**
 * Camera using right hand coordinates: X going right, Y going up, -Z going into the screen.
 *
 * */
class Camera(
        val lookfrom: Vector = Vector(),
        val lookat: Vector = Vector(0.0, 0.0, -1.0),
        val up: Vector = Vector(0.0, 1.0, 0.0),
        val verticalFov: Double = 90.0,
        val ratio: Double = 16.0 / 9.0) {

    val lowerLeft: Vector
    val horizontal: Vector
    val vertical: Vector

    init {
        val theta = verticalFov * PI / 180.0
        val halfHeight = tan(theta / 2)
        val halfWidth = halfHeight * ratio
        val direction = lookat - lookfrom

        // Local coordinate system
        val u = (direction cross up).normalize() // Compute the right vector (x)
        val w = -direction.normalize() // Compute the depth vector (z)
        val v = w cross u // compute the up vector (y)

        lowerLeft = lookfrom - halfWidth * u - halfHeight * v - w
        horizontal = 2.0 * halfWidth * u
        vertical = 2.0 * halfHeight * v
    }

    /**
     * create a ray
     *
     * @param u horizontal position on screen `[0,1]`. 0 is left, 1 is right
     * @param v vertical position on screen `[0,1]`. 0 is bottom, 1 is top (right hand coordinates, remember ?)
     * */
    fun trace(u: Double, v: Double): Ray {
        val direction = lowerLeft + u * horizontal + v * vertical - lookfrom
        return Ray(lookfrom, direction)
    }

}