@file:Suppress("NOTHING_TO_INLINE")

package fr.oyashirox.math

@Suppress("unused", "MemberVisibilityCanBePrivate")
data class Color(val r: Int, val g: Int, val b: Int, val a: Int = 0xFF) {
    constructor(r: Byte, g: Byte, b: Byte, a: Byte = 255.toByte()) : this(r.toInt(), g.toInt(), b.toInt(), a.toInt())
    constructor(r: Double, g: Double, b: Double, a: Double = 1.0) : this(
            (r * 255).toInt(),
            (g * 255).toInt(),
            (b * 255).toInt(),
            (a * 255).toInt()
    )

    var argbColor: Int = (a shl 24) or (r shl 16) or (g shl 8) or b

    inline operator fun plus(o: Color) = Color(r + o.r, g + o.g, b + o.b)

    fun toVector() = Vector((r / 255.0), (g / 255.0), (b / 255.0))

    fun gamma2() = toVector().sqrt().toColor()

}

inline operator fun Double.times(v: Color) = Color(
        this * (v.r / 255.0),
        this * (v.g / 255.0),
        this * (v.b / 255.0)
)

inline operator fun Vector.times(v: Color) = Color(
        this.x * (v.r / 255.0),
        this.y * (v.g / 255.0),
        this.z * (v.b / 255.0)
)

