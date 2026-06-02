cmake_minimum_required(VERSION 3.18)
project(PathtracerSpectralRealtime LANGUAGES CXX CUDA)

# --- Standard et Paramètres du Compilateur ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75 86) # 75 pour Turing, 86 pour RTX 30xx.

# Options de compilation pour la performance
set(PATH_TRACER_CUDA_OPTIONS --use_fast_math)
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND PATH_TRACER_CUDA_OPTIONS -O3)
endif()

# Trouver le CUDAToolkit
find_package(CUDAToolkit REQUIRED)

# Vérification du compilateur (le setup original est pour MSVC sous Windows)
if(MSVC)
    message(STATUS "Compilation avec MSVC")
else()
    message(WARNING "Ce projet est principalement configuré pour MSVC sous Windows. La compilation pourrait échouer sur d'autres plateformes.")
endif()

# Option pour activer/désactiver la compilation des tests
option(BUILD_TESTING "Compiler les tests unitaires" ON)

#=============================================================================
# CONFIGURATION MOTEUR
#=============================================================================

# --- Bibliothèque statique du moteur (pathtracer_lib) ---
# Compiler la logique du moteur dans une bibliothèque statique permet de la partager
# entre l'application principale et les tests, évitant la duplication de code.
add_library(pathtracer_lib STATIC
        src/cuda_kernels.cu
        src/scene_setup.cpp
        src/image_io.cpp
)

# Rendre les en-têtes publics pour que les autres cibles puissent les trouver.
target_include_directories(pathtracer_lib PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
)

# Lier la bibliothèque aux dépendances CUDA nécessaires.
# curand est requis pour les fonctions de curand_kernel.h.
target_link_libraries(pathtracer_lib PRIVATE
        CUDA::cudart
        CUDA::curand
)

target_compile_options(pathtracer_lib PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${PATH_TRACER_CUDA_OPTIONS}>
)

# Activer la compilation séparée pour CUDA. C'est crucial lorsque le code `__device__`
# est réparti sur plusieurs fichiers ou bibliothèques.
set_target_properties(pathtracer_lib PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)


# --- Exécutable Principal (Le Rendu) ---
add_executable(PathtracerSpectralRealtime
        src/main.cpp
)

# Lier l'exécutable à la bibliothèque du moteur.
target_link_libraries(PathtracerSpectralRealtime PRIVATE
        pathtracer_lib
)

# Avertissements du compilateur pour l'exécutable principal
if(MSVC)
    target_compile_options(PathtracerSpectralRealtime PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /MP>)
else()
    target_compile_options(PathtracerSpectralRealtime PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
endif()

# Commande post-build pour copier la DLL du runtime CUDA à côté de l'exécutable.
# Cela rend l'application portable sans avoir à modifier le PATH système.
if(WIN32)
    add_custom_command(
            TARGET PathtracerSpectralRealtime POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:CUDA::cudart>
            $<TARGET_FILE_DIR:PathtracerSpectralRealtime>
    )
endif()


#=============================================================================
# CONFIGURATION DES TESTS
#=============================================================================

if(BUILD_TESTING)
    # Activer CTest pour lancer les tests
    enable_testing()

    # Utiliser FetchContent pour télécharger et configurer Google Test
    include(FetchContent)
    FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.14.0 # Utiliser une version stable
    )

    # Pour Windows : s'assurer que gtest utilise le même runtime CRT que le projet
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)


    # --- Exécutable de Test ---
    add_executable(run_tests
            test/test_main.cpp
            test/test_vec3.cpp
            test/test_spectrum.cpp
            test/test_camera.cpp
            test/test_intersections.cu
            test/test_materials.cu
    )

    # La compilation séparée est aussi nécessaire pour les tests contenant du code CUDA.
    set_target_properties(run_tests PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
    )

    # Lier l'exécutable de test à la bibliothèque du moteur et à Google Test.
    target_link_libraries(run_tests PRIVATE
            pathtracer_lib
            gtest
    )

    # Ajouter l'exécutable de test à CTest pour qu'il puisse être lancé via `ctest`
    add_test(NAME GTest_run_tests COMMAND run_tests)

    # Avertissements du compilateur pour les tests
    if(MSVC)
        target_compile_options(run_tests PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /MP>)
    else()
        target_compile_options(run_tests PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
    endif()
endif()
