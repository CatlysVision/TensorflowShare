package com.example.tensorflowshare.classifier

import android.graphics.RectF

data class Recognition(
    var id: String,
    var title: String,
    var confidence: Float,
    var location: RectF
) {

    override fun toString(): String {
        var resultString = ""
        resultString += "[$id] "

        resultString += "$title "

        resultString += String.format("(%.1f%%) ", confidence * 100.0f)

        resultString += "$location "
        return resultString.trim()
    }
}