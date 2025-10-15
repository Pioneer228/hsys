function(get_cuda_gpu_arch OUTPUT_VAR)
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE GPU_ARCH
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(GPU_ARCH)
        string(REPLACE "\n" ";" GPU_ARCH_LIST "${GPU_ARCH}")
        list(SORT GPU_ARCH_LIST)
        list(REVERSE GPU_ARCH_LIST)
        list(GET GPU_ARCH_LIST 0 HIGHEST_GPU_ARCH)
        string(REGEX REPLACE "([0-9]+)\\.([0-9]+)" "\\1\\2" HIGHEST_GPU_ARCH "${HIGHEST_GPU_ARCH}")
        set(${OUTPUT_VAR} "${HIGHEST_GPU_ARCH}" PARENT_SCOPE)
    else()
        set(${OUTPUT_VAR} "0" PARENT_SCOPE)
    endif()
endfunction()

function(set_target_minimum_cuda_arch TARGET MINIMUM_CUDA_ARCH)
    get_cuda_gpu_arch(GPU_ARCH)
    if(GPU_ARCH GREATER "0")
        if(GPU_ARCH GREATER_EQUAL ${MINIMUM_CUDA_ARCH})
            set_target_properties(${TARGET} PROPERTIES CUDA_ARCHITECTURES native)
            message(STATUS "Using native compilation (GPU arch: sm_${GPU_ARCH})")
        else()
            set_target_properties(${TARGET} PROPERTIES CUDA_ARCHITECTURES ${MINIMUM_CUDA_ARCH})
            message(STATUS "GPU is too old (sm_${GPU_ARCH}), using sm_${MINIMUM_CUDA_ARCH} as minimum")
        endif()
    else()
        set_target_properties(${TARGET} PROPERTIES CUDA_ARCHITECTURES ${MINIMUM_CUDA_ARCH})
        message(STATUS "No NVIDIA GPU detected, generating PTX for sm_${MINIMUM_CUDA_ARCH}")
    endif()
endfunction()
