package fr.oyashirox.math

@Suppress("NOTHING_TO_INLINE", "MemberVisibilityCanBePrivate")
class Image(val width: Int, val height: Int) {
    val data = Array(width * height) { Vector() }

    inline operator fun get(x: Int, y: Int) = data[x + y * width]
    inline operator fun set(x: Int, y: Int, color: Color) {
        data[x + y * width] = color.toVector()
    }
    inline operator fun set(x: Int, y: Int, vector: Vector) {
        data[x + y * width] = vector
    }

    operator fun plusAssign(o: Image) {
        for (x in 0 until width) {
            for (y in 0 until height) {
                this[x, y] = this[x, y] + o[x, y]
            }
        }
    }

    fun setData(array: FloatArray) {
        data.forEachIndexed { index, vector ->
            if (index * 3 + 3 > array.size) return

            vector.x = array[index * 3].toDouble()
            vector.y = array[index * 3 + 1].toDouble()
            vector.z = array[index * 3 + 2].toDouble()
        }
    }

    /** @return a color array gamma corrected of size [width] x [height] */
    fun getColorArray() = Array(width * height) {
        data[it].toColor().gamma2()
    }
}