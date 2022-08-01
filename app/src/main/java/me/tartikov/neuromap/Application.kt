package me.tartikov.neuromap

class Application : android.app.Application() {
    init {
        instance = this
    }
    companion object {
        lateinit var instance: Application
            private set
    }
}