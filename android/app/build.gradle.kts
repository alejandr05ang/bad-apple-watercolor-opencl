plugins {
    id("com.android.application")
}

android {
    namespace = "com.grupo9.acuarela"
    compileSdk = 36
    ndkVersion = "30.0.16248370"

    defaultConfig {
        applicationId = "com.grupo9.acuarela"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        ndk { abiFilters += "arm64-v8a" }   // Pixel 8 (OpenCV compilado solo para esta ABI)

        externalNativeBuild {
            cmake {
                // OpenCV para Android compilado con WITH_OPENCL=ON (ver android/README.md)
                val opencv = System.getenv("OPENCV_ANDROID_DIR")
                    ?: "${System.getProperty("user.home")}/aplicaciones/Librerias/opencv-android-opencl/sdk/native/jni"
                arguments += listOf("-DOpenCV_DIR=$opencv", "-DANDROID_STL=c++_static")
                cppFlags += "-std=c++17"
            }
        }
    }

    buildTypes {
        // El código nativo se compila optimizado también en debug: el benchmark
        // CPU vs GPU no tiene sentido con -O0.
        debug {
            externalNativeBuild { cmake { arguments += "-DCMAKE_BUILD_TYPE=Release" } }
        }
        release {
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "4.1.2"
        }
    }

    // El video y los cuadros se toman directamente de la carpeta recursos/ del repo
    sourceSets {
        getByName("main") {
            assets.srcDirs("../../recursos/video", "../../recursos/cuadros")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}
