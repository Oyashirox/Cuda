package fr.oyashirox.math

import java.util.*

class Random {
    private val random = SplittableRandom()

    fun unitSphere(): Vector {
        var vector: Vector
        do {
            // Generate a random point in [-1,1] for all component
            vector = 2.0 * Vector(random.nextDouble(), random.nextDouble(), random.nextDouble()) - Vector(1.0, 1.0, 1.0)
        } while (vector.length() >= 1.0)
        return vector
    }
}