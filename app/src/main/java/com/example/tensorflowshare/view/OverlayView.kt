package com.example.tensorflowshare.view

import android.content.Context
import android.graphics.Canvas
import android.util.AttributeSet
import android.view.View

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyle: Int = 0
) :
    View(context, attrs, defStyle) {

    private var callback: DrawCallback? = null

    public fun setDrawCallbac(callback: DrawCallback) {
        this.callback = callback
    }

    public interface DrawCallback {
        fun onDraw(canvas: Canvas)
    }

    override fun onDraw(canvas: Canvas?) {
        callback?.onDraw(canvas!!)
    }

}