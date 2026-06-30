# 비-Metal GPU(ROCm/CUDA/Vulkan) 가속 노트

rakuko-forge 가 R9700/gfx1201(ROCm)에서 sam3.cpp 를 가속하려고 분석한 결과.

## 이 브랜치가 한 것

**일반 GPU 백엔드 선택**(커밋 d736ac8): 업스트림 `sam3_load_model` 은 `use_gpu` 시 **Metal 만** init.
`ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)` + `ggml_backend_dev_init` 로 임의 GPU(HIP/CUDA/
Vulkan)를 잡도록 추가. 실측: R9700/gfx1201 을 "ROCm0 GPU backend" 로 정상 인식.

## 남은 블로커: WIN_PART / WIN_UNPART 가 ggml-cuda 미구현

GPU 백엔드로 SAM3 ViT 인코더 그래프를 돌리면:
```
ggml_cuda_graph_evaluate_and_capture: op not supported node_19 (WIN_PART)
GGML_ASSERT(ok) failed  (ggml-cuda.cu)
```
- `GGML_OP_WIN_PART`/`GGML_OP_WIN_UNPART`(SAM ViT 윈도우 어텐션)는 **CPU 백엔드에만** forward 가 있고
  ggml-cuda(HIP 포함) 에는 없음. SAM3 ViT(`sam3_build_vit_graph` → `sam3_vit_block_forward`)의 4개 호출
  (sam3.cpp:3744/3799, 6850/6862)이 유일한 미지원 op. 다른 그래프(디코더/프롬프트/마스크)는 GPU OK.

## 진행 상황 (2026-07-01)

- ✅ **WIN_PART/WIN_UNPART CUDA 커널 구현 완료** — ggml 포크(fox1245/ggml `feat/win-part-cuda`)에
  추가. ggml 서브모듈을 이 포크로 재지정. **실측: SAM3 ViT 인코더 graph compute CPU 44s → GPU 1.84s.**
- ⚠️ **다음 갭: FLASH_ATTN_EXT** — WIN_PART 통과 후 `ggml_cuda_flash_attn_ext`(fattn.cu)가 gfx1201
  에서 `BEST_FATTN_KERNEL_NONE` → abort. ggml 의 CUDA flash-attention 이 RDNA4(gfx1201) 헤드구성에
  적합 커널이 없음. sam3 ViT 어텐션 4곳(sam3.cpp:3790/4010/4411/5490)이 `ggml_flash_attn_ext` 사용.
  → **이걸 풀어야 SAM3 가 완전 GPU.** 옵션: (A) ggml-cuda FA 를 gfx1201 에 맞게(어려움) (B) 비-FA
  어텐션 경로(softmax(QKᵀ)V, 지원 op) 추가 (C) **인코더만 backend_sched(GPU+CPU) → FA 만 CPU 폴백**(권장).

## (구) 해결 옵션 — WIN_PART 용 (참고)

### A) ggml-cuda 에 WIN_PART/WIN_UNPART 커널 추가 — **권장(근본·PR가능)**
CPU forward(`ggml-cpu/ops.cpp: ggml_compute_forward_win_part_f32`)는 단순 패딩-윈도우 재배치라
CUDA 커널로 1:1 포팅 가능(출력 원소당 입력 복사 or 0). 등록: `ggml_cuda_compute_forward` 스위치 +
`ggml_backend_cuda_device_supports_op` 에 두 op 추가. **sam3.cpp 변경 0**, 모든 경로 GPU 동작.
단 ggml(서브모듈=PABannier/ggml) 수정 → ggml 포크 필요.

### B) ViT compute 사이트만 ggml_backend_sched(GPU+CPU) 폴백 — sam3.cpp 내부지만 비자명
미지원 op 만 CPU 로 자동 폴백. 단 SAM3 인코더 3사이트(6083/6285/7323)는 `galloc`·출력 텐서를
`state.galloc`/`state.vit_output`(ctx0 소유)로 **영속**시켜 downstream 에서 재사용 → transient sched 와
비호환. sched 전환 시 출력을 persistent 버퍼로 copy-out(EdgeTAM 경로처럼) 하는 재설계 동반 필요.

## 현재 동작

`forge_seg` 는 **기본 CPU**(`--gpu` opt-in). SAM 은 등록당 1회의 가벼운 작업이라 CPU(~44s 인코딩)로도
실용. GPU 가속이 필요하면 옵션 A 를 권장.
